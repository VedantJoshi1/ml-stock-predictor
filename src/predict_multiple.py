# src/predict_multiple.py
import sys
from predict import predict

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "^GSPC", "SPY"]

for ticker in tickers:
    predict(ticker)
