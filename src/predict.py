import sys
import joblib
from data_loader import load_data
from features import create_features

def predict(ticker):
    # Load trained model
    model = joblib.load(f"models/{ticker}_rf_model.joblib")

    # Load latest data
    df = load_data(ticker)
    df = create_features(df)

    features = ["Return", "MA_5", "MA_20", "Volatility"]
    X_latest = df[features].iloc[-1:]


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

    predict(sys.argv[1].upper())
