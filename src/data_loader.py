import yfinance as yf
import pandas as pd

def load_data(ticker, period="10y"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.dropna(inplace=True)
    return df
