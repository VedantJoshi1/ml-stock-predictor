import yfinance as yf

def load_data(ticker, start="2010-01-01"):
    df = yf.download(ticker, start=start)
    df["Ticker"] = ticker
    return df
