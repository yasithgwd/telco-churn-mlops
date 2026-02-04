import json
from pathlib import Path
import joblib

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_model(modl, path:str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modl, path)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)