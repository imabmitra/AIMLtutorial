from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_data
from preprocessing import build_preprocessor
from model import build_pipeline

def train():
    df = load_data()

    X = df[["age", "fare", "sex"]]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
