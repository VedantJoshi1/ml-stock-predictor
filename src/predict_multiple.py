# src/predict_multiple.py
from predict import predict

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY", "QQQ"]

for ticker in tickers:
    predict(ticker)
