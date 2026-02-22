import argparse
import sys
import pandas as pd
import joblib
from pathlib import Path

# Add src to path so sklearn can unpickle custom functions
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn.data import clean_data  

def main():
    parser = argparse.ArgumentParser(description="Telco churn batch prediction")
    parser.add_argument("--model", default="artifacts/models/logreg.joblib", help="Path to saved pipeline .joblib")
    parser.add_argument("--input", default="input.csv", help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()

    # Load model (pipeline)
    try:
        pipeline = joblib.load(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model from {args.model}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Load input
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR: Failed to read input CSV from {args.input}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Predict
    try:
        proba = pipeline.predict_proba(df)[:, 1]
        pred = pipeline.predict(df)
    except Exception as e:
        print(f"ERROR: Prediction failed\n{e}", file=sys.stderr)
        sys.exit(1)

    # Build output
    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = pred

    # Save
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
