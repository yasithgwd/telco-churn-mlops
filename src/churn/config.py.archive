import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data import clean_data


# -------------------------------------------------
# Paths (same pattern as train.py)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"

MODEL_PATH = MODELS_DIR / "logreg.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURES_PATH = REPORTS_DIR / "feature_columns.json"


# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
def run_prediction(input_path: Path, output_path: Path) -> None:
    # Load artifacts
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load feature columns (used during training)
    with open(FEATURES_PATH, "r") as f:
        feature_columns = json.load(f)

    # Load and clean input data
    df = pd.read_csv(input_path)
    df = clean_data(df)

    # Drop target if present
    if "Churn Value" in df.columns:
        df = df.drop(columns=["Churn Value"])

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Transform + predict
    X = preprocessor.transform(df)
    predictions = model.predict(X)

    # Probability (for churn class = 1)
    probabilities = model.predict_proba(X)[:, 1]

    # Save results
    output_df = df.copy()
    output_df["prediction"] = predictions
    output_df["churn_probability"] = probabilities

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch predictions")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output",
        default=PREDICTIONS_DIR / "predictions.csv",
        help="Path to output predictions CSV"
    )

    args = parser.parse_args()

    run_prediction(
        input_path=Path(args.input),
        output_path=Path(args.output)
    )
