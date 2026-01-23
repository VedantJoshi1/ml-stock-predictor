import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from data_loader import load_data
from features import create_features

def train_model(ticker):
    df = load_data(ticker)
    df = create_features(df)

    features = ["Return", "MA_5", "MA_20", "Volatility"]
    X = df[features]
    y = df["Target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)

    # Expected return calculation
    test_returns = df.iloc[split:]["Return"].values
    expected_return = test_returns[preds == 1].mean()

    model_data = {
        "model": model,
        "features": features,
        "expected_return": expected_return
    }

    joblib.dump(model_data, f"models/{ticker}.joblib")

    print(f"{ticker} | Precision: {precision:.2f}")
    print(f"{ticker} | Expected Return (UP days): {expected_return*100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"]


    for ticker in tickers:
        train_model(ticker)
