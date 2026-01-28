# src/predict.py
import sys
import joblib
from data_loader import load_data
from features import create_features

def predict_signal(ticker):
    model_data = joblib.load(f"models/{ticker}.joblib")

    df = load_data(ticker, period="6mo")
    df, _ = create_features(df)

    latest = df.iloc[-1:][model_data["features"]]

    prob = model_data["model"].predict_proba(latest)[0][1]
    direction = "UP" if prob >= 0.5 else "DOWN"

    return {
        "ticker": ticker,
        "direction": direction,
        "confidence": prob,
        "expected_return": model_data["expected_return"],
        "score": prob * model_data["expected_return"]
    }

if __name__ == "__main__":
    ticker = sys.argv[1]
    res = predict_signal(ticker)

    print(f"\n{ticker} Prediction")
    print(f"Direction: {res['direction']}")
    print(f"Confidence: {res['confidence']:.2%}")
    print(f"Expected Return: {res['expected_return']:.2%}")
