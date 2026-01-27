import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

from data_loader import load_data
from features import create_features

FEATURES = ["Return", "MA_5", "MA_20", "MA_50", "Volatility", "Bull_Regime"]

def train_ticker(ticker):
    print(f"Training model for {ticker}...")

    df = load_data(ticker)
    df = create_features(df)

    models = {}

    for horizon, target in {
        "1d": "Target_1d",
        "30d": "Target_30d",
        "90d": "Target_90d",
    }.items():

        X = df[FEATURES]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False, test_size=0.2
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        precision = precision_score(y_test, preds)

        expected_return = df.loc[y == 1, "Return"].mean()

        models[horizon] = {
            "model": model,
            "precision": precision,
            "expected_return": expected_return
        }

        print(f"{ticker} | {horizon} Precision: {precision:.2f}")

    joblib.dump(
        {
            "models": models,
            "features": FEATURES
        },
        f"models/{ticker}.joblib"
    )

    print("-" * 40)

if __name__ == "__main__":
    tickers = sys.argv[1:] or ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
    for t in tickers:
        train_ticker(t)
