import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_data_model_pipeline():
    # Load data & model
    df = pd.read_csv("data.csv")
    model = joblib.load("model.joblib")

    # Data sanity: check 5 columns
    expected_cols = ["sepal_length","sepal_width","petal_length","petal_width","species"]
    assert all(col in df.columns for col in expected_cols), "Missing columns in data"
    print("Data sanity check passed")

    # Predict using first 4 features
    X = df.drop("species", axis=1)
    y = df["species"]
    preds = model.predict(X)

    # Accuracy check
    acc = accuracy_score(y, preds)
    assert acc > 0.7, f"Model accuracy too low: {acc:.2f}"
    print(f"Model prediction accuracy: {acc:.2f}")
