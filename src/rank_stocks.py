import os
import joblib
import numpy as np

from data_loader import load_data
from features import create_features

TICKERS = ["AAPL", "NVDA", "SPY", "QQQ"]

def rank_stocks():
    results = []

    for ticker in TICKERS:
        model_path = f"models/{ticker}.joblib"

        if not os.path.exists(model_path):
            print(f"Skipping {ticker} (no trained model)")
            continue

        model_data = joblib.load(model_path)
        model = model_data["model"]
        features = model_data["features"]
        expected_return = model_data["expected_return"]

        df = load_data(ticker)
        df = create_features(df)

        latest = df.iloc[-1:][features]

        proba = model.predict_proba(latest)[0]
        prediction = np.argmax(proba)
        confidence = proba[prediction]

        if prediction == 1:  # Only rank UP signals
            score = confidence * expected_return
            results.append({
                "ticker": ticker,
                "confidence": confidence,
                "expected_return": expected_return,
                "score": score
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\n📊 Ranked Stock Signals (Best → Worst)\n")

    if not results:
        print("No bullish signals today.")
        return

    for r in results:
        print(
            f"{r['ticker']:6} | "
            f"Confidence: {r['confidence']*100:5.2f}% | "
            f"Exp Return: {r['expected_return']*100:5.2f}% | "
            f"Score: {r['score']*100:5.2f}"
        )

if __name__ == "__main__":
    rank_stocks()
