import yfinance as yf
import pandas as pd

def load_data(ticker):
    df = yf.download(
        ticker,
        start="2010-01-01",
        progress=False,
        group_by="column"
    )

    # 🔥 FORCE FLAT COLUMNS (DESTROYS MULTIINDEX)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure Close is a proper Series
    df["Close"] = df["Close"].astype(float)

    return df
