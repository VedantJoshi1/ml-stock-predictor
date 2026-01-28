# src/data_loader.py
import yfinance as yf
import pandas as pd

def load_data(ticker, period="5y"):
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # Flatten MultiIndex columns (CRITICAL FIX)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df
