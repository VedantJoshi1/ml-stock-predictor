# src/features.py
import numpy as np

def create_features(df):
    df = df.copy()

    close = df["Close"]

    df["Return"] = close.pct_change()
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["Volatility"] = df["Return"].rolling(5).std()

    df["Tomorrow"] = close.shift(-1)
    df["Target"] = (df["Tomorrow"] > close).astype(int)

    df = df.dropna()

    features = ["MA5", "MA20", "Volatility"]

    return df, features
