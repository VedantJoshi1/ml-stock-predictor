import yfinance as yf

def load_data(ticker, start="2010-01-01"):
    df = yf.download(ticker, start=start)
    return df

if __name__ == "__main__":
    df = load_data("AAPL")
    print(df.head())
