#  ML Stock Signal & Ranking System

A machine-learning–based stock **signal and ranking system** that predicts short-term stock direction, estimates expected returns, and ranks stocks by confidence and risk-adjusted strength.

Built as a practical project to demonstrate **machine learning, feature engineering, and quantitative-style decision making** using real market data.

>  Disclaimer: This project is for educational and research purposes only.  
> It is **not financial advice**.

---

##  What It Does

For any stock ticker (e.g. AAPL, NVDA, SPY), the system:
- Predicts **next-day direction** (UP / DOWN)
- Outputs **confidence** (model probability)
- Estimates **expected return** on UP days
- Ranks stocks by signal strength
- Filters trades using overall market trend (SPY)

This mirrors how real quantitative systems generate **trade signals**, not price targets.

---

##  How It Works (High Level)

- Historical price data downloaded via Yahoo Finance
- Features engineered from price action (returns, trends, volatility)
- Supervised ML classification using **Random Forest**
- One trained model per ticker
- Models saved and reused for fast daily predictions

---

##  Quick Start (Terminal Commands)

```bash
# Activate virtual environment (Windows)
cd C:\Projects\ml-stock-predictor
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train models (downloads data automatically)
python src/train.py

# Predict a single stock
python src/predict.py AAPL

# Predict any new ticker (auto-downloads data)
python src/predict.py NVDA
python src/predict.py SPY

# Run daily ranked signals
python src/daily_signal.py
