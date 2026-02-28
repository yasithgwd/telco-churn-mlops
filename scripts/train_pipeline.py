import sys
from pathlib import Path
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from churn.data import load_data, clean_data
from churn.features import split_features_target, preprocessing_pipeline
from churn.split import make_train_test_split
from churn.train import train_logreg_model
from churn.evaluate import evaluate_model
from churn.io import save_model, save_json, utc_now_compact, sha256_file, git_commit_hash

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
TARGET_COL = "Churn Value"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

if __name__ == "__main__":
    run_id = utc_now_compact()

    model_dir = MODELS_DIR / run_id
    report_dir = REPORTS_DIR / run_id

    model_path = model_dir / "model.joblib"
    metrics_path = report_dir / "metrics.json"
    run_path = report_dir / "run.json"

    df = load_data(str(DATA_PATH))
    X, y = split_features_target(df, TARGET_COL)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor = preprocessing_pipeline(clean_data)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = train_logreg_model(X_train_t, y_train)
    metrics = evaluate_model(model, X_test_t, y_test)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    save_model(full_pipeline, str(model_path))
    save_json(metrics, str(metrics_path))

    run_meta = {
        "run_id": run_id,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "data_sha256": sha256_file(DATA_PATH),
        "target_col": TARGET_COL,
        "git_commit": git_commit_hash(PROJECT_ROOT),
    }
    save_json(run_meta, str(run_path))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved run metadata to {run_path}")