💠 Crazy Returns – Quantum Portfolio Optimization

Crazy Returns is an interactive Streamlit web app that combines finance, optimization, and quantum computing to help investors analyze and optimize stock portfolios. It merges classical portfolio theory with quantum-inspired methods like QUBO and QAOA (via PennyLane) to create smarter investment strategies.

🚀 Features

📊 Stock Data Fetching – Live prices via yfinance

⚡ Portfolio Optimization – Mean-Variance, Sharpe Ratio, Minimum Variance

🧠 Quantum Optimization – QUBO formulation + QAOA solver

🎯 Classical Baseline – Markowitz & CVXPY optimization

📈 Visual Dashboards – Risk-return scatter, efficient frontier, weights, growth projections

🔮 Monte Carlo Simulations – Portfolio growth forecasting

📝 PDF Export – Simple report generation with ReportLab

🎛️ Interactive UI – Sidebar filters, sliders, and expanders

🛠️ Tech Stack

Frontend/UI: Streamlit

Data & Finance: yfinance, numpy, pandas

Visualization: plotly, matplotlib, seaborn

Optimization: scipy, cvxpy, joblib

Quantum: PennyLane
 (QAOA)

Reports: reportlab

📌 How It Works

Select Stocks & Timeframe → App fetches data with yfinance

Calculate Metrics → Returns, volatility, covariance

Optimize Portfolios → Classical & Quantum-inspired solvers

Compare Results → Equal-weight vs optimized portfolios

Visualize & Export → Charts + downloadable PDF

⚡ Installation

Clone the repo and install dependencies:

git clone https://github.com/your-username/crazy-returns.git
cd crazy-returns
pip install -r requirements.txt

▶️ Usage

Run the Streamlit app:

streamlit run app.py

📸 Screenshots

(Add your app screenshots here for better showcase)

📚 Roadmap

 Multi-period optimization

 Real-time live portfolio tracker

 Advanced PDF reporting

 Deploy on Streamlit Cloud

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License

MIT License – feel free to use and modify.

🔥 This project is perfect for finance enthusiasts, data scientists, and quantum curious minds exploring the future of portfolio optimization.
