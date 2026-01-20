from data_loader import load_data
from features import create_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def train_model():
    df = load_data("AAPL")
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
    print("Precision:", precision)

if __name__ == "__main__":
    train_model()
