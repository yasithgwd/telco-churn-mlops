import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from churn.data import load_data, clean_data
from churn.features import split_features_traget, preprocessing_pipeline
from churn.split import make_train_test_split
from churn.train import train_logreg_model
from churn.evaluate import evaluate_model
from churn.io import save_model, save_json

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
TARGET_COL = "Churn Value"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "logreg.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "reports" / "metrics.json"

if __name__ == "__main__":
    print("testing the code in train.py")
    df = load_data(str(DATA_PATH))

    df = load_data(str(DATA_PATH))
    df = clean_data(df)

    X, y = split_features_traget(df, TARGET_COL)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor = preprocessing_pipeline(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = train_logreg_model(X_train_t, y_train)
    metrics = evaluate_model(model, X_test_t, y_test)

    print(metrics)

    save_model(model, str(MODEL_PATH))
    save_json(metrics, str(METRICS_PATH))
    print("METRICS OBJECT:", metrics)
    print("METRICS TYPE:", type(metrics))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


    print("Training data shape:", X_train_t.shape)
    print("Test data shape:", X_test_t.shape)