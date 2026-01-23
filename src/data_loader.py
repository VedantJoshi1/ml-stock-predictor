import yfinance as yf
import pandas as pd

def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01", progress=False, group_by="column")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Close"] = df["Close"].astype(float)
    return df
