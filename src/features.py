import pandas as pd
import numpy as np

def create_features(df):
    df["Return"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Volatility"] = df["Return"].rolling(10).std()

    # Market regime (bull = 1, bear = 0)
    df["Bull_Regime"] = (df["MA_20"] > df["MA_50"]).astype(int)

    # Targets
    df["Target_1d"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Target_30d"] = (df["Close"].shift(-30) > df["Close"]).astype(int)
    df["Target_90d"] = (df["Close"].shift(-90) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df
