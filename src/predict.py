import sys
import os
import joblib
import numpy as np

from data_loader import load_data
from features import create_features
from train import train_model


def predict(ticker):
    model_path = f"models/{ticker}.joblib"

    # Auto-train if model doesn't exist
    if not os.path.exists(model_path):
        print(f"\n⚠️ No model found for {ticker}. Training now...")
        train_model(ticker)
        print("✅ Training complete.\n")

    # Load trained model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    features = model_data["features"]
    expected_return = model_data["expected_return"]

    # Load latest data
    df = load_data(ticker)
    df = create_features(df)

    latest = df.iloc[-1:][features]

    proba = model.predict_proba(latest)[0]
    prediction = np.argmax(proba)
    confidence = proba[prediction]

    direction = "UP 📈" if prediction == 1 else "DOWN 📉"

    print(f"{ticker} Prediction for Tomorrow")
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Expected Return (if UP): {expected_return * 100:.2f}%\n")


if __name__ == "__main__":
    ticker = sys.argv[1].upper()
    predict(ticker)
