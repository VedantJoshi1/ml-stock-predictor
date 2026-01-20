import pandas as pd

def create_features(df):
    df = df.copy()

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Return"].rolling(5).std()

    df.dropna(inplace=True)
    return df
