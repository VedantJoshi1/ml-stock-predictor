import pandas as pd

def create_features(df):
    df = df.copy()

    # Next-day price
    df["Tomorrow"] = df["Close"].shift(-1)

    # Daily return
    df["Return"] = df["Close"].pct_change()

    # Technical features
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Return"].rolling(5).std()

    # Target: 1 if price goes up tomorrow
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df = df.dropna()
    return df
