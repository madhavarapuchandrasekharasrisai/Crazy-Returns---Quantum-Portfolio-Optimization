import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from num2words import num2words
from joblib import Parallel, delayed
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Set seaborn style for beautiful charts (used where Plotly isn't)
sns.set_style("whitegrid")
sns.set_palette("deep")

# Optimization
from scipy.optimize import minimize

# Try to import cvxpy but allow fallback
try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

# Quantum imports
import pennylane as qml

# -------------------------
# Configuration / Defaults
# -------------------------
TRADING_DAYS = 252

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"]
DEFAULT_PERIOD = "3y"
DEFAULT_INTERVAL = "1d"
DEFAULT_K = 4
DEFAULT_RISK_AVERSION = 10.0
DEFAULT_PENALTY_A = 5.0
DEFAULT_RF = 0.02
DEFAULT_QAOA_P = 2
DEFAULT_QAOA_STEPS = 120
DEFAULT_QAOA_LR = 0.1
DEFAULT_PROJECTION_YEARS = 5
DEFAULT_MONTE_CARLO_PATHS = 1000  # For advanced projections

# -------------------------
# Data utilities (with caching)
# -------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_prices(tickers, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        # single ticker
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all")
    return prices

@st.cache_data
def returns_and_cov(prices: pd.DataFrame):
    rets = np.log(prices / prices.shift(1)).dropna()
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    mu_annual = mu_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS
    # Ensure ordering as array
    return rets, mu_annual.values, cov_annual.values

# -------------------------
# Evaluation helpers
# -------------------------
def bitstring_to_weights(x):
    x = np.array(x, dtype=int)
    if x.sum() == 0:
        return np.zeros_like(x, dtype=float)
    return x / x.sum()

def portfolio_stats(weights, mu, Sigma, rf=0.0):
    weights = np.array(weights, dtype=float)
    if weights.sum() > 0:
        wnorm = weights / weights.sum()
    else:
        wnorm = weights
    port_ret = float(wnorm @ mu)
    port_var = float(wnorm @ Sigma @ wnorm)
    port_std = math.sqrt(max(port_var, 1e-12))
    sharpe = (port_ret - rf) / (port_std if port_std > 0 else 1e-12)
    return {
        "return": port_ret,
        "variance": port_var,
        "volatility": port_std,
        "sharpe": sharpe,
    }

def projected_return(annual_return, years, num_paths=DEFAULT_MONTE_CARLO_PATHS):
    """
    Advanced Monte Carlo projection: Simulate multiple paths with random variations.
    Returns mean cumulative return as percentage, plus 5th and 95th percentiles for range.
    """
    if years <= 0:
        return 0.0, 0.0, 0.0
    # Simulate paths with Gaussian noise (simple model)
    daily_returns = np.random.normal(annual_return / TRADING_DAYS, (annual_return / TRADING_DAYS)**0.5, (num_paths, years * TRADING_DAYS))
    cumulative = np.exp(np.log(1 + daily_returns).cumsum(axis=1)) - 1
    end_returns = cumulative[:, -1] * 100
    mean_ret = np.mean(end_returns)
    low_ret = np.percentile(end_returns, 5)
    high_ret = np.percentile(end_returns, 95)
    return mean_ret, low_ret, high_ret

def cost_qubo(x, Q):
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)

def calculate_risk_reduction(baseline_vol, optimized_vol):
    if baseline_vol == 0:
        return 0.0
    reduction = (baseline_vol - optimized_vol) / baseline_vol * 100
    return reduction

# -------------------------
# QUBO construction
# -------------------------
def build_qubo(mu, Sigma, k, risk_aversion=DEFAULT_RISK_AVERSION, A=DEFAULT_PENALTY_A, mu_shift=0.0):
    """
    Build QUBO matrix Q and constant term for:
      objective ≈ - (mu + mu_shift)^T x + risk_aversion * x^T Sigma x + A (sum x - k)^2
    mu_shift is used to shift returns (e.g. subtract rf for Sharpe-like proxy).
    """
    mu = np.array(mu, dtype=float) + float(mu_shift)
    Sigma = np.array(Sigma, dtype=float)
    n = mu.shape[0]
    Q = np.zeros((n, n), dtype=float)

    # Risk term: lambda * x^T Sigma x
    for i in range(n):
        Q[i, i] += risk_aversion * Sigma[i, i]
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * risk_aversion * Sigma[i, j]

    # Return term: -mu^T x (diagonal)
    for i in range(n):
        Q[i, i] += -mu[i]

    # Cardinality penalty: A (sum x - k)^2
    for i in range(n):
        Q[i, i] += A * 1.0
        Q[i, i] += -2.0 * A * k
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * A

    const = A * (k ** 2)
    return Q, const

def qubo_to_ising(Q):
    Q = np.array(Q, dtype=float)
    n = Q.shape[0]
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    const = 0.0

    for i in range(n):
        const += Q[i, i] / 2.0
        h[i] += -Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            const += Q[i, j] / 4.0
            J[i, j] += Q[i, j] / 4.0
            h[i] += -Q[i, j] / 4.0
            h[j] += -Q[i, j] / 4.0

    return h, J, const

# -------------------------
# QAOA Solver (PennyLane) with caching
# -------------------------
@st.cache_resource
def get_qaoa_solver(Q, p, lr, steps):
    return QAOAPortfolioSolver(Q, p, lr, steps)

class QAOAPortfolioSolver:
    def __init__(self, Q, p=2, lr=0.1, steps=120, seed=42):
        self.Q = np.array(Q, dtype=float)
        self.n = self.Q.shape[0]
        self.p = p
        self.lr = lr
        self.steps = steps
        self.rng = np.random.default_rng(seed)

        # Use default.qubit (statevector) for expectation optimization
        self.dev = qml.device("default.qubit", wires=self.n, shots=None)

        # Build Ising coefficients & Hamiltonian
        self.h, self.J, self.const = self._qubo_to_ising(self.Q)
        self.H = self._build_hamiltonian(self.h, self.J)

        # Build qnode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            gammas = params[: self.p]
            betas = params[self.p : self.p * 2]

            # Uniform superposition
            for w in range(self.n):
                qml.Hadamard(wires=w)

            for layer in range(self.p):
                gamma = float(gammas[layer])
                # Apply RZ for linear terms (h)
                for i in range(self.n):
                    if abs(self.h[i]) > 1e-12:
                        qml.RZ(2.0 * gamma * float(self.h[i]), wires=i)
                # Apply ZZ (MultiRZ) for J
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        if abs(self.J[i, j]) > 1e-12:
                            qml.MultiRZ(2.0 * gamma * float(self.J[i, j]), wires=[i, j])

                # Mixer
                beta = float(betas[layer])
                for w in range(self.n):
                    qml.RX(2.0 * beta, wires=w)

            return qml.expval(self.H)

        self.circuit = circuit

    @staticmethod
    def _qubo_to_ising(Q):
        n = Q.shape[0]
        h = np.zeros(n)
        J = np.zeros((n, n))
        const = 0.0
        for i in range(n):
            const += Q[i, i] / 2.0
            h[i] += -Q[i, i] / 2.0
        for i in range(n):
            for j in range(i + 1, n):
                const += Q[i, j] / 4.0
                J[i, j] += Q[i, j] / 4.0
                h[i] += -Q[i, j] / 4.0
                h[j] += -Q[i, j] / 4.0
        return h, J, const

    @staticmethod
    def _build_hamiltonian(h, J):
        coeffs = []
        ops = []
        n = len(h)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                coeffs.append(h[i])
                ops.append(qml.PauliZ(i))
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    coeffs.append(J[i, j])
                    ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:
            coeffs = [0.0]
            ops = [qml.PauliZ(0)]
        return qml.Hamiltonian(coeffs, ops)

    def optimize(self, progress_callback=None):
        params = 0.01 * self.rng.standard_normal(size=(2 * self.p,))
        opt = qml.AdamOptimizer(stepsize=self.lr)

        best_val = float("inf")
        best_params = params.copy()

        for step in range(self.steps):
            val = float(self.circuit(params))
            params = opt.step(self.circuit, params)
            if val < best_val:
                best_val = val
                best_params = params.copy()
            if progress_callback:
                progress_callback(step + 1, self.steps)
        return best_params, best_val

    def sample_best_bitstring(self, params, n_shots=2048):
        # Use shot-based device to sample bitstrings
        dev = qml.device("default.qubit", wires=self.n, shots=n_shots)

        @qml.qnode(dev)
        def sampler(params):
            gammas = params[: self.p]
            betas = params[self.p : self.p * 2]
            for w in range(self.n):
                qml.Hadamard(wires=w)
            for layer in range(self.p):
                gamma = float(gammas[layer])
                for i in range(self.n):
                    if abs(self.h[i]) > 1e-12:
                        qml.RZ(2.0 * gamma * float(self.h[i]), wires=i)
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        if abs(self.J[i, j]) > 1e-12:
                            qml.MultiRZ(2.0 * gamma * float(self.J[i, j]), wires=[i, j])
                beta = float(betas[layer])
                for w in range(self.n):
                    qml.RX(2.0 * beta, wires=w)
            return qml.sample(wires=range(self.n))

        samples = sampler(params)
        samples = np.array(samples, dtype=int)
        # Unique bitstrings
        uniq, counts = np.unique(samples, axis=0, return_counts=True)
        best_x = None
        best_cost = float("inf")
        for x in uniq:
            c = cost_qubo(x, self.Q)
            if c < best_cost:
                best_cost = c
                best_x = x
        return best_x, best_cost

# -------------------------
# Classical baseline with parallelization
# -------------------------
def project_to_simplex(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.maximum(y, 0)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(y - theta, 0)
    return w

def markowitz_continuous(mu, Sigma, risk_aversion=DEFAULT_RISK_AVERSION, allow_short=False):
    mu = np.array(mu, dtype=float)
    Sigma = np.array(Sigma, dtype=float)
    n = len(mu)

    if not HAS_CVXPY:
        lam = float(risk_aversion)
        eps = 1e-6
        A = lam * Sigma + eps * np.eye(n)
        try:
            w = np.linalg.solve(A, mu)
        except np.linalg.LinAlgError:
            w = np.ones(n) / n
        if allow_short:
            if abs(w.sum()) > 1e-12:
                return w / (w.sum())
            return w
        w = project_to_simplex(w)
        return w

    # cvxpy formulation
    w = cp.Variable(n)
    # Ensure Sigma is PSD for cvxpy; wrap as constant
    obj = 0.5 * risk_aversion * cp.quad_form(w, Sigma) - mu @ w
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints += [w >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            return np.ones(n) / n
    if w.value is None:
        return np.ones(n) / n
    return np.array(w.value, dtype=float)

def discretize_top_k(weights, k):
    k = int(k)
    if k <= 0:
        return (weights * 0).astype(int)
    idx = np.argsort(-weights)[:k]
    x = np.zeros_like(weights, dtype=int)
    x[idx] = 1
    return x

# -------------------------
# Efficient Frontier helper with parallelization
# -------------------------
def compute_frontier_point(lam, mu, Sigma):
    w = markowitz_continuous(mu, Sigma, risk_aversion=lam, allow_short=False)
    stats = portfolio_stats(w, mu, Sigma)
    return stats["return"], stats["volatility"]

@st.cache_data
def efficient_frontier(mu, Sigma, points=50):
    lams = np.linspace(0.1, 50, points)
    results = Parallel(n_jobs=-1)(delayed(compute_frontier_point)(lam, mu, Sigma) for lam in lams)
    rets, vols = zip(*results)
    return np.array(vols), np.array(rets)

# -------------------------
# Enhanced Visualization Helpers (using Plotly for interactivity)
# -------------------------
def plot_projected_growth_interactive(stats_dict, years, mode, tickers):
    year_range = np.arange(0, years + 1)
    fig = go.Figure()

    for label, stats in stats_dict.items():
        annual_ret = stats["return"]
        growth = [(1 + annual_ret) ** t - 1 for t in year_range]
        growth_pct = [g * 100 for g in growth]
        fig.add_trace(go.Scatter(x=year_range, y=growth_pct, mode='lines+markers', name=label))

    fig.update_layout(
        title=f"Projected Portfolio Growth Comparison Over {years} Years ({mode} Mode - Based on {len(tickers)} Assets)",
        xaxis_title="Years",
        yaxis_title="Cumulative Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

def plot_risk_comparison_interactive(stats_dict, mode, tickers):
    labels = list(stats_dict.keys())
    vols = [stats["volatility"] * 100 for stats in stats_dict.values()]
    fig = px.bar(x=labels, y=vols, title=f"Risk Comparison: Normal vs Optimized Portfolios ({mode} Mode - Volatility Decreased After Optimization)",
                 labels={'x': 'Portfolio Type', 'y': 'Volatility (Risk) (%)'})
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

def plot_weights_comparison_interactive(stats_dict, tickers, mode):
    data = []
    for label, stats in stats_dict.items():
        weights = stats.get("weights", np.ones(len(tickers)) / len(tickers)) * 100
        data.append(go.Bar(name=label, x=tickers, y=weights))

    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', title=f"Portfolio Weights Comparison Across Assets ({mode} Mode)",
                      xaxis_title="Assets", yaxis_title="Weight (%)", legend=dict(orientation="h"))
    fig.update_xaxes(tickangle=45)
    return fig

def plot_risk_return_scatter_interactive(stats_dict, mode):
    data = []
    for label, stats in stats_dict.items():
        data.append({
            'Risk (%)': stats["volatility"] * 100,
            'Return (%)': stats["return"] * 100,
            'Sharpe Ratio': stats["sharpe"],
            'Portfolio': label
        })
    df = pd.DataFrame(data)
    fig = px.scatter(df, x='Risk (%)', y='Return (%)', size='Sharpe Ratio', color='Portfolio',
                     title=f"Risk vs Return Scatter Plot ({mode} Mode - Bubble Size Shows Sharpe Ratio)",
                     hover_data=['Sharpe Ratio'])
    return fig

def plot_efficient_frontier_interactive(mu, Sigma, stats_dict, mode):
    vols, rets = efficient_frontier(mu, Sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols * 100, y=rets * 100, mode='lines', name='Efficient Frontier', line=dict(dash='dash')))

    for label, stats in stats_dict.items():
        fig.add_trace(go.Scatter(x=[stats["volatility"] * 100], y=[stats["return"] * 100], mode='markers', name=label, marker=dict(size=12)))

    fig.update_layout(title=f"Efficient Frontier with Portfolio Points ({mode} Mode)",
                      xaxis_title="Volatility (Risk) (%)", yaxis_title="Expected Return (%)",
                      legend=dict(orientation="h"))
    return fig

def plot_risk_vs_reward_line_interactive(stats_dict, mode):
    data = []
    for label, stats in stats_dict.items():
        data.append({
            'Risk (%)': stats["volatility"] * 100,
            'Reward (%)': stats["return"] * 100,
            'Portfolio': label
        })
    df = pd.DataFrame(data).sort_values('Risk (%)')
    fig = px.line(df, x='Risk (%)', y='Reward (%)', color='Portfolio', markers=True,
                  title=f"Risk vs Reward Line Chart ({mode} Mode - Lines Show Tradeoffs)")
    return fig

# -------------------------
# PDF Export Helper
# -------------------------
def generate_pdf_report(stats_dict, charts, mode, projection_years):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 100, f"Portfolio Optimization Report ({mode} Mode)")
    c.drawString(100, height - 120, f"Projected over {projection_years} years")
    # Add more content or embed images if possible (simplified here)
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# Streamlit UI (Updated with simple headings for non-financial users)
# -------------------------
st.set_page_config(page_title=" Crazy Returns", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UX (colors, spacing, fonts)
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stAlert {
        border-radius: 4px;
    }
    .explanation {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💠 Crazy Returns - Quantum Portfolio Optimization")


# Sidebar: Organized into expanders for better navigation with simple labels
with st.sidebar:
    st.header("🛠️ Setup Your Choices")

    with st.expander("Pick Stocks and Time Period", expanded=True):
        tickers_text = st.text_input("Stocks to Consider (separate with commas)", value=",".join(DEFAULT_TICKERS), help="Like AAPL for Apple, MSFT for Microsoft")
        period = st.selectbox("How Far Back to Look at Prices", ["1y", "2y", "3y", "5y"], index=2, help="Choose how many years of past data to use")
        interval = st.selectbox("Price Check Frequency", ["1d", "1wk", "1mo"], index=0, help="Daily, weekly, or monthly prices")

    with st.expander("Tuning for Safety and Growth", expanded=True):
        k = st.number_input("How Many Stocks to Pick", min_value=1, max_value=50, value=DEFAULT_K, help="Choose a small number for focused choices")
        risk_aversion = st.slider("How Much to Avoid Risk (Higher = Safer)", 0.0, 50.0, float(DEFAULT_RISK_AVERSION), step=0.5, help="High value means prefer low-risk options")
        penalty_A = st.slider("Stick to Chosen Number of Stocks", 0.0, 50.0, float(DEFAULT_PENALTY_A), step=0.5, help="Higher makes it stricter")
        rf_annual = st.number_input("Safe Investment Rate (like savings)", value=float(DEFAULT_RF), step=0.005, help="Rate from very safe options like bonds")

    with st.expander("Future Growth Guess Settings", expanded=True):
        projection_years = st.number_input("Years Ahead to Guess Growth", min_value=1, max_value=30, value=DEFAULT_PROJECTION_YEARS, help="How many years to predict ahead")
        monte_carlo_paths = st.number_input("Number of Growth Simulations", min_value=100, max_value=5000, value=DEFAULT_MONTE_CARLO_PATHS, help="More simulations = better guess of possible outcomes")

    with st.expander("Advanced Quantum Settings (Optional)", expanded=False):
        qaoa_layers = st.slider("Quantum Layers (for better accuracy)", 1, 5, value=int(DEFAULT_QAOA_P), help="Higher = more precise but slower")
        qaoa_steps = st.slider("Optimization Steps (for fine-tuning)", 20, 500, value=int(DEFAULT_QAOA_STEPS), step=10, help="More steps = better results, but takes longer")
        qaoa_lr = st.select_slider("Learning Speed", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=float(DEFAULT_QAOA_LR), help="How fast the system learns")

    st.divider()
    st.header("Choose Optimization Style")
    mode_options = {
        "Mean-Variance": "Balance Growth and Safety",
        "Sharpe Ratio": "Best Growth per Risk",
        "Minimum Variance": "Lowest Risk Possible"  # New mode added
    }
    selected_mode = st.radio("Pick a Style:", list(mode_options.keys()), format_func=lambda x: mode_options[x], help="Choose how to balance money growth and safety")

    if selected_mode == "Sharpe Ratio":
        with st.expander("Extra Settings for Growth per Risk", expanded=True):
            sharpe_proxy_gamma = st.slider("Risk Control in Quantum Guess", 0.0, 100.0, float(DEFAULT_RISK_AVERSION), step=0.5, help="Adjust how much to penalize risk")

    run_btn = st.button("Start Optimization", help="Click to analyze and suggest portfolio")

    # Personalization: Save preferences
    if 'saved_prefs' not in st.session_state:
        st.session_state['saved_prefs'] = {}
    if st.button("Save My Choices"):
        st.session_state['saved_prefs'] = {
            'tickers': tickers_text,
            'period': period,
            'k': k
        }
        st.success("Choices saved! Reload to use them.")
    if st.button("Load Saved Choices"):
        if st.session_state['saved_prefs']:
            tickers_text = st.session_state['saved_prefs'].get('tickers', ",".join(DEFAULT_TICKERS))
            period = st.session_state['saved_prefs'].get('period', DEFAULT_PERIOD)
            k = st.session_state['saved_prefs'].get('k', DEFAULT_K)
            st.experimental_rerun()

# Main content: Tabs with simple headings
tab1, tab2, tab3 = st.tabs(["📈 See Your Data", "🔍 Quantum Results ", "📊 Data Visualization"])

with tab1:
    st.subheader("What Your Stock Data Looks Like")
    if run_btn or st.session_state.get('data_fetched', False):
        with st.spinner("Getting latest stock prices..."):
            try:
                tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
                if not tickers:
                    raise ValueError("Please enter at least one valid stock ticker.")
                prices = fetch_prices(tickers, period=period, interval=interval)
                st.session_state['prices'] = prices
                st.session_state['data_fetched'] = True
            except Exception as e:
                st.error(f"Oops! Couldn't get prices: {e}. Try different stocks or check your internet.")
                st.stop()

        if prices.empty:
            st.warning("No price data found. Try different stocks or time period.")
            st.stop()

        rets, mu, Sigma = returns_and_cov(prices)
        st.session_state['mu'] = mu
        st.session_state['Sigma'] = Sigma

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Recent Stock Prices (Last 200 Days)")
            st.dataframe(prices.tail(200).style.format("{:.2f}"))

        with col2:
            st.markdown("#### Expected Yearly Growth Rates")
            df_stats = pd.DataFrame({"Expected Growth (%)": [f"{r*100:.2f}%" for r in mu]}, index=prices.columns)
            st.dataframe(df_stats)

        st.markdown("#### How Prices Have Changed Over Time")
        st.line_chart(prices, use_container_width=True)
    else:
        st.info("Click 'Start Optimization' to see your data.")

with tab2:
    st.subheader("What the Optimization Found")
    if run_btn:
        start_time = time.time()
        n = len(st.session_state['mu'])
        mu = st.session_state['mu']
        Sigma = st.session_state['Sigma']
        prices = st.session_state['prices']
        tickers = list(prices.columns)

        # Compute baseline (normal/unoptimized: equal weights on all assets)
        w_baseline = np.ones(n) / n
        stats_baseline = portfolio_stats(w_baseline, mu, Sigma, rf=float(rf_annual))
        stats_baseline['weights'] = w_baseline
        proj_ret_baseline, low_baseline, high_baseline = projected_return(stats_baseline['return'], projection_years)

        # Simple explanation box
        st.markdown("""
        <div class="explanation">
        <strong>Quick Guide to Results:</strong><br>
        - <strong>Normal Choice (Before Any Smart Changes):</strong> Just splitting money equally among all stocks. Growth guess: {baseline_proj}% over {years} years (could be between {low_baseline:.2f}% and {high_baseline:.2f}% in simulations).<br>
        - <strong>After Smart Changes:</strong> Using math (including quantum ideas) to pick better mixes for more growth or less risk. Check the numbers below!
        </div>
        """.format(
            baseline_proj=f"{proj_ret_baseline:.2f}",
            years=projection_years,
            low_baseline=low_baseline,
            high_baseline=high_baseline
        ), unsafe_allow_html=True)

        if selected_mode == "Mean-Variance":
            st.info("Style: Markowitz Mean - Variance")

            with st.status("Preparing Model...", expanded=True):
                Q, const = build_qubo(mu, Sigma, k=int(k), risk_aversion=float(risk_aversion), A=float(penalty_A))

            with st.status("Running Quantum-Inspired Optimizer...", expanded=True) as status:
                progress_bar = st.progress(0)
                def update_progress(step, total):
                    progress_bar.progress(step / total)
                solver = get_qaoa_solver(Q, int(qaoa_layers), float(qaoa_lr), int(qaoa_steps))
                params, best_val = solver.optimize(progress_callback=update_progress)
                status.update(label="Quantum step done!", state="complete")

            with st.status("Picking Best Choices...", expanded=True):
                x_qaoa, q_cost = solver.sample_best_bitstring(params, n_shots=2048)
                if x_qaoa is None:
                    st.warning("Quantum step had a hiccup. Try more steps or different settings.")
                    x_qaoa = np.zeros(len(mu), dtype=int)

            w_qaoa = bitstring_to_weights(x_qaoa)
            stats_q = portfolio_stats(w_qaoa, mu, Sigma, rf=float(rf_annual))
            stats_q['weights'] = w_qaoa

            with st.status("Running Standard Math Optimizer...", expanded=True):
                w_cont = markowitz_continuous(mu, Sigma, risk_aversion=float(risk_aversion), allow_short=False)
                x_topk = discretize_top_k(w_cont, int(k))
                w_classic = bitstring_to_weights(x_topk)
                stats_c = portfolio_stats(w_classic, mu, Sigma, rf=float(rf_annual))
                stats_c['weights'] = w_classic

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Quantum-Suggested Mix (After Smart Changes)")
                df_q = pd.DataFrame({
                    "Stock": prices.columns,
                    "Picked?": list(map(int, x_qaoa)),
                    "Share (%)": [f"{w*100:.2f}%" for w in w_qaoa],
                })
                st.dataframe(df_q)
                st.metric("Growth per Risk Score", f"{stats_q['sharpe']:.4f}", help="Higher means better balance of growth and safety")
                st.caption(f"Yearly Growth Guess: {stats_q['return']*100:.3f}%  ·  Risk Level: {stats_q['volatility']*100:.3f}%")
                proj_ret_q, low_q, high_q = projected_return(stats_q['return'], projection_years)
                st.metric(f"Guessed Growth Over {projection_years} Years", f"{proj_ret_q:.2f}%", help=f"Could range from {low_q:.2f}% to {high_q:.2f}% based on simulations")
                st.caption(f"After quantum changes, growth guess is '{num2words(round(proj_ret_q))}' percent over {projection_years} years, vs normal {proj_ret_baseline:.2f}%.")
                risk_red_q = calculate_risk_reduction(stats_baseline['volatility'], stats_q['volatility'])
                delta_label_q = "lowered" if risk_red_q > 0 else "raised"
                st.metric("How Much Risk Lowered", f"{risk_red_q:.2f}% {delta_label_q}", help=f"From normal risk of {stats_baseline['volatility']*100:.3f}%")

            with col2:
                st.markdown("##### Standard Math-Suggested Mix (After Smart Changes)")
                df_c = pd.DataFrame({
                    "Stock": prices.columns,
                    "Picked?": list(map(int, x_topk)),
                    "Share (%)": [f"{w*100:.2f}%" for w in w_classic],
                })
                st.dataframe(df_c)
                st.metric("Growth per Risk Score", f"{stats_c['sharpe']:.4f}", help="Higher means better balance of growth and safety")
                st.caption(f"Yearly Growth Guess: {stats_c['return']*100:.3f}%  ·  Risk Level: {stats_c['volatility']*100:.3f}%")
                proj_ret_c, low_c, high_c = projected_return(stats_c['return'], projection_years)
                st.metric(f"Guessed Growth Over {projection_years} Years", f"{proj_ret_c:.2f}%", help=f"Could range from {low_c:.2f}% to {high_c:.2f}% based on simulations")
                st.caption(f"After standard changes, growth guess is '{num2words(round(proj_ret_c))}' percent over {projection_years} years, vs normal {proj_ret_baseline:.2f}%.")
                risk_red_c = calculate_risk_reduction(stats_baseline['volatility'], stats_c['volatility'])
                delta_label_c = "lowered" if risk_red_c > 0 else "raised"
                st.metric("How Much Risk Lowered", f"{risk_red_c:.2f}% {delta_label_c}", help=f"From normal risk of {stats_baseline['volatility']*100:.3f}%")

            elapsed = time.time() - start_time
            st.caption(f"Time taken: {elapsed:.1f} seconds")

            st.download_button("Download Quantum Mix CSV", df_q.to_csv(index=False), "quantum_mix.csv")
            st.download_button("Download Standard Mix CSV", df_c.to_csv(index=False), "standard_mix.csv")

            # Store for charts
            st.session_state['stats_dict'] = {
                "Normal Equal Split": stats_baseline,
                "Quantum Mix": stats_q,
                "Standard Mix": stats_c
            }
            st.session_state['tickers'] = tickers

            # PDF Export
            if st.button("Save Results as PDF"):
                pdf_buffer = generate_pdf_report(st.session_state['stats_dict'], [], selected_mode, projection_years)
                st.download_button("Download PDF Report", pdf_buffer, "portfolio_report.pdf", "application/pdf")

            # User Feedback
            feedback = st.text_area("What do you think? Any suggestions?")
            if st.button("Send Feedback"):
                # Placeholder: Could log or email feedback
                st.success("Thanks for your feedback!")

        elif selected_mode == "Sharpe Ratio":
            st.info("Mode: Sharpe Ratio (classical exact + QAOA proxy)")

            with st.status("Running classical Sharpe optimization...", expanded=True):
                def sharpe_objective(w, mu, cov, rf=0.0):
                    ret = float(np.dot(w, mu))
                    vol = math.sqrt(max(float(w @ cov @ w), 1e-12))
                    sharpe = (ret - rf) / vol
                    return -sharpe


                n = len(mu)
                x0 = np.ones(n) / n
                bounds = [(0.0, 1.0)] * n
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
                res = minimize(sharpe_objective, x0, args=(mu, Sigma, float(rf_annual)), method="SLSQP",
                               bounds=bounds, constraints=cons, options={'ftol': 1e-9, 'maxiter': 1000})
                w_sharpe = np.clip(res.x, 0.0, 1.0)
                if w_sharpe.sum() > 0:
                    w_sharpe /= w_sharpe.sum()
                stats_sh = portfolio_stats(w_sharpe, mu, Sigma, rf=float(rf_annual))
                stats_sh['weights'] = w_sharpe

                x_sh_topk = discretize_top_k(w_sharpe, int(k))
                w_sh_topk = bitstring_to_weights(x_sh_topk)
                stats_sh_topk = portfolio_stats(w_sh_topk, mu, Sigma, rf=float(rf_annual))
                stats_sh_topk['weights'] = w_sh_topk

            with st.status("Building QAOA Sharpe-proxy QUBO...", expanded=True):
                mu_shift = -float(rf_annual)
                Qp, constp = build_qubo(mu, Sigma, k=int(k), risk_aversion=float(sharpe_proxy_gamma),
                                        A=float(penalty_A), mu_shift=mu_shift)

            with st.status("Running QAOA (Sharpe-proxy)...", expanded=True) as status:
                progress_bar = st.progress(0)


                def update_progress(step, total):
                    progress_bar.progress(step / total)


                solver_p = QAOAPortfolioSolver(Qp, p=int(qaoa_layers), lr=float(qaoa_lr), steps=int(qaoa_steps))
                params_p, best_val_p = solver_p.optimize(progress_callback=update_progress)
                status.update(label="QAOA optimization complete!", state="complete")

            with st.status("Sampling bitstrings (Sharpe-proxy)...", expanded=True):
                x_qaoa_sh, q_cost_sh = solver_p.sample_best_bitstring(params_p, n_shots=2048)
                if x_qaoa_sh is None:
                    x_qaoa_sh = np.zeros(len(mu), dtype=int)
                w_qaoa_sh = bitstring_to_weights(x_qaoa_sh)
                stats_q_sh = portfolio_stats(w_qaoa_sh, mu, Sigma, rf=float(rf_annual))
                stats_q_sh['weights'] = w_qaoa_sh

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Classical Continuous (After Optimization)")
                df_sh = pd.DataFrame({"Ticker": prices.columns, "Weight": [f"{w * 100:.2f}%" for w in w_sharpe]})
                st.dataframe(df_sh)
                st.metric("Sharpe Ratio", f"{stats_sh['sharpe']:.4f}")
                st.caption(
                    f"Annual Return: {stats_sh['return'] * 100:.3f}%  ·  Volatility: {stats_sh['volatility'] * 100:.3f}%")
                proj_ret_sh = projected_return(stats_sh['return'], projection_years)
                st.metric(
                    label=f"Projected {projection_years}-Year Cumulative Return (Mean)",
                    value=f"{proj_ret_sh[0]:.2f}%",
                    delta=f"+{proj_ret_sh[0] - some_baseline:.2f}% vs Baseline" if 'some_baseline' in locals() else None,  # Optional: Add comparison
                    delta_color="normal"  # Colors delta green for positive changes
                )

                # Additional metrics for the range (enhancement)
                st.metric(
                    label=f"Low Estimate (5th Percentile)",
                    value=f"{proj_ret_sh[1]:.2f}%",
                    delta_color="inverse"
                )
                st.metric(
                    label=f"High Estimate (95th Percentile)",
                    value=f"{proj_ret_sh[2]:.2f}%",
                    delta_color="normal"
                )
                st.caption(
                    f"After classical continuous optimization, the projected return is '{num2words(round(proj_ret_sh))}' percent over {projection_years} years, compared to the normal {proj_ret_baseline:.2f}%.")
                risk_red_sh = calculate_risk_reduction(stats_baseline['volatility'], stats_sh['volatility'])
                delta_label_sh = "decreased" if risk_red_sh > 0 else "increased"
                st.metric("Risk Reduction (Volatility)", f"{risk_red_sh:.2f}% {delta_label_sh}",
                          help=f"Compared to normal portfolio volatility of {stats_baseline['volatility'] * 100:.3f}%")

                st.markdown("##### Classical Top-k (After Optimization)")
                df_sh_topk = pd.DataFrame({
                    "Ticker": prices.columns,
                    "Selected": list(map(int, x_sh_topk)),
                    "Weight": [f"{w * 100:.2f}%" for w in w_sh_topk],
                })
                st.dataframe(df_sh_topk)
                st.metric("Sharpe Ratio", f"{stats_sh_topk['sharpe']:.4f}")
                st.caption(
                    f"Annual Return: {stats_sh_topk['return'] * 100:.3f}%  ·  Volatility: {stats_sh_topk['volatility'] * 100:.3f}%")
                proj_ret_sh_topk = projected_return(stats_sh_topk['return'], projection_years)
                st.metric(f"Projected {projection_years}-Year Cumulative Return", f"{proj_ret_sh_topk:.2f}%",
                          help="Simple compounding based on annualized return (assumes constant growth)")
                st.caption(
                    f"After classical top-k optimization, the projected return is '{num2words(round(proj_ret_sh_topk))}' percent over {projection_years} years, compared to the normal {proj_ret_baseline:.2f}%.")
                risk_red_sh_topk = calculate_risk_reduction(stats_baseline['volatility'], stats_sh_topk['volatility'])
                delta_label_sh_topk = "decreased" if risk_red_sh_topk > 0 else "increased"
                st.metric("Risk Reduction (Volatility)", f"{risk_red_sh_topk:.2f}% {delta_label_sh_topk}",
                          help=f"Compared to normal portfolio volatility of {stats_baseline['volatility'] * 100:.3f}%")

            with col2:
                st.markdown("##### QAOA Proxy (After Optimization)")
                df_qs = pd.DataFrame({
                    "Ticker": prices.columns,
                    "Selected": list(map(int, x_qaoa_sh)),
                    "Weight": [f"{w * 100:.2f}%" for w in w_qaoa_sh],
                })
                st.dataframe(df_qs)
                st.metric("Sharpe Ratio", f"{stats_q_sh['sharpe']:.4f}")
                st.caption(
                    f"Annual Return: {stats_q_sh['return'] * 100:.3f}%  ·  Volatility: {stats_q_sh['volatility'] * 100:.3f}%  ·  QUBO Cost: {q_cost_sh:.4f}")
                proj_ret_q_sh = projected_return(stats_q_sh['return'], projection_years)
                st.metric(f"Projected {projection_years}-Year Cumulative Return", f"{proj_ret_q_sh:.2f}%",
                          help="Simple compounding based on annualized return (assumes constant growth)")
                st.caption(
                    f"After QAOA proxy optimization, the projected return is '{num2words(round(proj_ret_q_sh))}' percent over {projection_years} years, compared to the normal {proj_ret_baseline:.2f}%.")
                risk_red_q_sh = calculate_risk_reduction(stats_baseline['volatility'], stats_q_sh['volatility'])
                delta_label_q_sh = "decreased" if risk_red_q_sh > 0 else "increased"
                st.metric("Risk Reduction (Volatility)", f"{risk_red_q_sh:.2f}% {delta_label_q_sh}",
                          help=f"Compared to normal portfolio volatility of {stats_baseline['volatility'] * 100:.3f}%")

            # Store for charts
            st.session_state['stats_dict'] = {
                "Normal (Equal Weights)": stats_baseline,
                "Classical Continuous": stats_sh,
                "Classical Top-k": stats_sh_topk,
                "QAOA Proxy": stats_q_sh
            }
            st.session_state['tickers'] = tickers


        elif selected_mode == "Minimum Variance":
            st.info("Style: Focusing on Lowest Risk")
            # Implement min variance: set risk_aversion high, ignore returns in objective
            w_min_var = markowitz_continuous(np.zeros_like(mu), Sigma, risk_aversion=100.0, allow_short=False)
            stats_min_var = portfolio_stats(w_min_var, mu, Sigma, rf=float(rf_annual))
            stats_min_var['weights'] = w_min_var
            # Display similar to above

            st.session_state['stats_dict'] = {
                "Normal Equal Split": stats_baseline,
                "Low Risk Mix": stats_min_var
            }

        # Real-time refresh button
        if st.button("Update with Latest Prices"):
            st.experimental_rerun()

    else:
        st.info("Click 'Start Optimization' to see suggestions.")

with tab3:
    st.subheader("Pictures to Explain the Results")
    if run_btn and 'stats_dict' in st.session_state:
        stats_dict = st.session_state['stats_dict']
        tickers = st.session_state['tickers']

        # Use Plotly for interactive charts
        st.markdown("#### Guessed Future Growth Over Time")
        fig_growth = plot_projected_growth_interactive(stats_dict, projection_years, selected_mode, tickers)
        st.plotly_chart(fig_growth, use_container_width=True)
        st.caption("Move your mouse over lines to see details. This shows how money might grow year by year.")

        st.markdown("#### How Risky Each Choice Is")
        fig_risk = plot_risk_comparison_interactive(stats_dict, selected_mode, tickers)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Bars show risk levels; shorter is safer. Hover for exact numbers.")

        st.markdown("#### How Money is Split Among Stocks")
        fig_weights = plot_weights_comparison_interactive(stats_dict, tickers, selected_mode)
        st.plotly_chart(fig_weights, use_container_width=True)
        st.caption("Bars show percentage of money in each stock. Hover to compare.")

        st.markdown("#### Risk vs Growth Dots")
        fig_scatter = plot_risk_return_scatter_interactive(stats_dict, selected_mode)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Dots show growth vs risk; bigger dots mean better balance. Hover for details.")

        st.markdown("#### Best Possible Growth-Safety Line")
        fig_frontier = plot_efficient_frontier_interactive(mu, Sigma, stats_dict, selected_mode)
        st.plotly_chart(fig_frontier, use_container_width=True)
        st.caption("The dashed line is the 'best possible' mixes. Dots show where your choices land.")

        st.markdown("#### Risk vs Growth in Lines")
        fig_risk_reward = plot_risk_vs_reward_line_interactive(stats_dict, selected_mode)
        st.plotly_chart(fig_risk_reward, use_container_width=True)
        st.caption("Lines connect risk and growth points for easy comparison.")

    else:
        st.info("Run the optimization to see pictures.")

# Footer with simple help
st.markdown("---")
st.caption("This tool helps suggest how to split money among stocks for better growth or safety. It's for learning only—not real advice! If something's unclear, try the explanations or send feedback.")
