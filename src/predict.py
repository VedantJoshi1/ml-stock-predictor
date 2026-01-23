import sys
import joblib
from data_loader import load_data
from features import create_features
from train import train_model
import os

def predict(ticker):
    model_path = f"models/{ticker}_rf_model.joblib"

    # Train model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"No existing model found for {ticker}. Training now...")
        model = train_model(ticker)
        if model is None:
            print("Prediction aborted due to insufficient data.")
            return
    else:
        model = joblib.load(model_path)

    # Load latest data
    df = load_data(ticker)
    df = create_features(df)

    features = ["Return", "MA5", "MA20", "Volatility", "RSI", "Price_vs_MA20"]
    X_latest = df[features].iloc[-1:]  # keep DataFrame to avoid sklearn warnings

    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0][prediction]

    direction = "UP 📈" if prediction == 1 else "DOWN 📉"

    print(f"\n{ticker} Prediction for Tomorrow")
    print(f"Direction: {direction}")
    print(f"Confidence: {probability:.2%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    predict(ticker)
