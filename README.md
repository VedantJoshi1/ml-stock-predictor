# 📈 Machine Learning Stock Predictor

An end-to-end machine learning application that predicts the **next-day direction** (up or down) of a stock using historical market data and technical indicators.

Built with **Python**, **scikit-learn**, and **real financial data**.  
Designed as a practical ML project for learning, experimentation, and portfolio demonstration.

> ⚠️ Disclaimer: This project is for educational purposes only and is **not financial advice**.

---

## ⚡ Quick Start (Terminal Commands)

```powershell
# Activate virtual environment (Windows)
cd C:\Projects\ml-stock-predictor
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train a model for a stock
python src/train.py AAPL

# Predict tomorrow's direction for a stock
python src/predict.py AAPL

# Predict multiple stocks at once
python src/predict_multiple.py

# Add a new ticker (example)
python src/predict.py NVDA
