from __future__ import annotations
import json
from pathlib import Path
import joblib
import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit_hash(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_model(obj: Any, path: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, str(p))


def save_json(obj: Any, path: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)