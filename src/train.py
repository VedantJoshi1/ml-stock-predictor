from data_loader import load_data
from features import create_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import joblib
import os
import sys

def train_model(ticker):
    print(f"\nTraining model for {ticker}...")
    df = load_data(ticker)
    df = create_features(df)

    features = ["Return", "MA5", "MA20", "Volatility", "RSI", "Price_vs_MA20"]
    X = df[features]
    y = df["Target"]

    if len(X) < 200:
        print(f"Not enough data to train {ticker} model.")
        return None

    X_train = X[:-100]
    X_test = X[-100:]
    y_train = y[:-100]
    y_test = y[-100:]

    # Tuned Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Logistic Regression baseline
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print(f"{ticker} | RF Precision: {precision_score(y_test, rf_preds):.2f}")
    print(f"{ticker} | LR Precision: {precision_score(y_test, lr_preds):.2f}")

    # Save RF model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_rf_model.joblib"
    joblib.dump(rf_model, model_path)
    print(f"{ticker} model saved at {model_path}")
    return rf_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    train_model(ticker)
