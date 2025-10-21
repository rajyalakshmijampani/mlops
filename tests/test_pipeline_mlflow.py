import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os

def load_best_mlflow_model(uri,model_name="IRIS_Rand_Forest"):
    mlflow_tracking_uri = uri

    client = MlflowClient()
    all_versions = client.get_latest_versions(name=model_name)
    best_version = None
    best_accuracy = -1

    for v in all_versions:
        run_id = v.run_id
        metrics = client.get_run(run_id).data.metrics
        acc = metrics.get("accuracy", -1)
        if acc > best_accuracy:
            best_accuracy = acc
            best_version = v.version

    print(f"Best MLflow model version: {best_version} with accuracy: {best_accuracy:.4f}")
    model_uri = f"models:/{model_name}/{best_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def test_data_model_pipeline(uri):
    df = pd.read_csv("data.csv")

    # Data sanity check
    expected_cols = ["sepal_length","sepal_width","petal_length","petal_width","species"]
    assert all(col in df.columns for col in expected_cols), "Missing columns in data"
    print("Data sanity check passed")

    # Load the best MLflow model
    model = load_best_mlflow_model(uri)

    # Predict using first 4 features
    X = df.drop("species", axis=1)
    y = df["species"]
    preds = model.predict(X)

    # Accuracy check
    acc = accuracy_score(y, preds)
    assert acc > 0.7, f"Model accuracy too low: {acc:.2f}"
    print(f"Model prediction accuracy: {acc:.2f}")
