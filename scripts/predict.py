import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PRODUCTION_FILE = ARTIFACTS_DIR / "models" / "production.json"

def load_production_model_path() -> Path:
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(f"Production file not found: {PRODUCTION_FILE}")

    with open(PRODUCTION_FILE, "r", encoding="utf-8") as f:
        prod = json.load(f)

    model_rel = prod["model_path"]
    return PROJECT_ROOT / model_rel

def main():
    parser = argparse.ArgumentParser(description="Run churn prediction")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--model",
        required=False,
        help="Optional override model path (for testing)",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="predictions.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    # Decide which model to load
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = load_production_model_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)

    df = pd.read_csv(args.input)

    # Pipeline already includes cleaning and preprocessing
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    output_df = df.copy()
    output_df["churn_probability"] = probs
    output_df["churn_prediction"] = preds

    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
