# 📈 Machine Learning Stock Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end machine learning project that predicts **next-day stock direction** and ranks stocks using **confidence-weighted expected returns**.

Built with Python and scikit-learn using real historical market data.  
Designed as a **research tool**, **signal input**, and **ML portfolio project** — not financial advice.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is **not financial advice** and should not be used as the sole basis for trading decisions.

---

## 🎯 Why This Project Exists

This project demonstrates:
- Practical machine learning applied to financial data
- Feature engineering with technical indicators
- Model evaluation beyond raw accuracy
- Real-world ML workflow (train → save → predict → rank)

Use cases:
- Research assistance
- Signal confirmation
- ML experimentation
- Portfolio demonstration

---

## 🔍 How It Works

1. **Data Collection**: Uses `yfinance` to download historical stock prices.
2. **Feature Engineering**: Calculates returns, moving averages,
