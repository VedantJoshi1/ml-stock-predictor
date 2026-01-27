import sys
import os
import joblib
import numpy as np

from data_loader import load_data
from features import create_features
from train import train_ticker

def predict(ticker):
    model_path = f"models/{ticker}.joblib"

    if not os.path.exists(model_path):
        print(f"No model for {ticker}. Training now...")
        train_ticker(ticker)

    model_data = joblib.load(model_path)
    features = model_data["features"]
    models = model_data["models"]

    df = load_data(ticker)
    df = create_features(df)

    latest = df.iloc[-1:][features]

    print(f"\n📊 {ticker} Predictions")

    for horizon, data in models.items():
        model = data["model"]
        proba = model.predict_proba(latest)[0]
        direction = np.argmax(proba)
        confidence = proba[direction]

        arrow = "📈 UP" if direction == 1 else "📉 DOWN"

        print(
            f"{horizon.upper():>4} | {arrow} | "
            f"Confidence: {confidence*100:.2f}% | "
            f"Expected Return (UP): {data['expected_return']*100:.2f}%"
        )

if __name__ == "__main__":
    ticker = sys.argv[1]
    predict(ticker)
