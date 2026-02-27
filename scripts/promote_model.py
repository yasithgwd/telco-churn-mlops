import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
PRODUCTION_FILE = MODELS_DIR / "production.json"

def main():
    parser = argparse.ArgumentParser(description="Promote model to production")
    parser.add_argument("--run_id", required=True, help="Run ID to to promote")
    args = parser.parse_args()

    run_id = args.run_id
    model_path = MODELS_DIR / run_id / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model for run_id {run_id} not found")
    
    production_data = {
        "run_id": run_id,
        "model_path": str(model_path.relative_to(PROJECT_ROOT))
    }

    with open(PRODUCTION_FILE, "w", encoding="utf-8") as f:
        json.dump(production_data, f, indent=2)

    print(f"Promoted run {run_id} to production")

if __name__ == "__main__":
    main() 