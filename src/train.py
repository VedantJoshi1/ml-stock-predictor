# src/train.py
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_data
from features import create_features

def train_model(ticker):
    print(f"Training model for {ticker}...")

    df = load_data(ticker)
    df, features = create_features(df)

    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42
    )
    model.fit(X, y)

    expected_return = df.loc[df["Target"] == 1, "Return"].mean()

    joblib.dump(
        {
            "model": model,
            "features": features,
            "expected_return": expected_return
        },
        f"models/{ticker}.joblib"
    )

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    train_model(ticker)
