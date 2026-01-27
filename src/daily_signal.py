import joblib
import numpy as np
import os

from data_loader import load_data
from features import create_features

# -------------------------
# CONFIG (adjust later)
# -------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
CONF_THRESHOLD = 0.55
MIN_EXPECTED_RETURN = 0.0

# -------------------------
def load_model(ticker):
    path = f"models/{ticker}.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# -------------------------
def get_signal(ticker):
    model_data = load_model(ticker)
    if model_data is None:
        return None

    model = model_data["model"]
    features = model_data["features"]
    expected_return = model_data["expected_return"]

    df = load_data(ticker)
    df = create_features(df)
    latest = df.iloc[-1:][features]

    proba = model.predict_proba(latest)[0]
    confidence = proba[1]  # probability of UP

    direction = "UP" if confidence >= 0.5 else "DOWN"
    score = confidence * expected_return

    return {
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "expected_return": expected_return,
        "score": score
    }

# -------------------------
def main():
    print("\n📊 DAILY MARKET SIGNAL\n")

    # ---- SPY FIRST (MARKET REGIME)
    spy = get_signal("SPY")
    if spy is None:
        print("⚠️ SPY model missing. Train SPY first.")
        return

    print(f"SPY Market Direction: {spy['direction']} | Confidence: {spy['confidence']*100:.2f}%")
    print("-" * 50)

    if spy["direction"] == "DOWN":
        print("🚫 Market is bearish. Avoid long trades.\n")
        return

    signals = []

    for ticker in TICKERS:
        s = get_signal(ticker)
        if s is None:
            continue

        if s["confidence"] < CONF_THRESHOLD:
            continue
        if s["expected_return"] <= MIN_EXPECTED_RETURN:
            continue

        signals.append(s)

    signals.sort(key=lambda x: x["score"], reverse=True)

    if not signals:
        print("📉 No high-quality trades today.")
        return

    print("✅ TRADE CANDIDATES (Best → Worst)\n")

    for s in signals:
        print(
            f"{s['ticker']} | {s['direction']} | "
            f"Conf: {s['confidence']*100:.2f}% | "
            f"Exp Ret: {s['expected_return']*100:.2f}% | "
            f"Score: {s['score']:.4f}"
        )

    print("\n⚠️ Educational use only. Manage risk.")

# -------------------------
if __name__ == "__main__":
    main()
