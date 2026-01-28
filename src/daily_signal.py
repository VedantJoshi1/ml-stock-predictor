# src/daily_signal.py
import joblib
from train import train_model
from predict import predict_signal

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
MARKET = "SPY"

def ensure_model(ticker):
    try:
        joblib.load(f"models/{ticker}.joblib")
    except:
        train_model(ticker)

def main():
    print("\n📊 DAILY MARKET SIGNAL\n")

    ensure_model(MARKET)
    spy = predict_signal(MARKET)

    print(
        f"SPY: {spy['direction']} | Confidence: {spy['confidence']:.2%}"
    )
    print("-" * 50)

    if spy["direction"] == "DOWN":
        print("Market bearish — NO LONG TRADES today.")
        return

    signals = []

    for ticker in TICKERS:
        try:
            ensure_model(ticker)
            sig = predict_signal(ticker)
            signals.append(sig)
        except Exception as e:
            print(f"{ticker}: ERROR ({e})")

    signals.sort(key=lambda x: x["score"], reverse=True)

    for s in signals:
        print(
            f"{s['ticker']} | {s['direction']} | "
            f"Conf: {s['confidence']:.2%} | "
            f"ExpRet: {s['expected_return']:.2%} | "
            f"Score: {s['score']:.4f}"
        )

if __name__ == "__main__":
    main()
