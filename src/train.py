from data_loader import load_data
from features import create_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression

def train_model(ticker):
    df = load_data(ticker)
    df = create_features(df)

    features = ["Return", "MA_5", "MA_20", "Volatility"]
    X = df[features]
    y = df["Target"]

    X_train = X[:-100]
    X_test = X[-100:]
    y_train = y[:-100]
    y_test = y[-100:]

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=100,
        random_state=1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    precision = precision_score(y_test, preds)
    print(f"{ticker} RF Precision:", precision)

    return model, X_test, y_test, preds


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    for ticker in tickers:
        train_model(ticker)
