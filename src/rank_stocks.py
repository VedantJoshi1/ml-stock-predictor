import joblib
import os

def rank_tickers(tickers):
    scores = []

    for ticker in tickers:
        path = f"models/{ticker}.joblib"
        if not os.path.exists(path):
            continue

        data = joblib.load(path)["models"]["1d"]

        score = (
            data["precision"] * 0.6 +
            data["expected_return"] * 10 * 0.4
        )

        scores.append((ticker, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 Ranked Stock Signals (Best → Worst)\n")
    for t, s in scores:
        print(f"{t:<6} Score: {s:.3f}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY", "QQQ"]
    rank_tickers(tickers)
