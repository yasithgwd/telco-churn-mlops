# Codebase Documentation

## Overview

This project is organized into:

- `scripts/`: runnable entry points for training, promotion, and prediction
- `src/churn/`: reusable data, feature, split, training, evaluation, and I/O logic
- `data/`: raw and optional processed datasets
- `artifacts/`: versioned models, reports, and predictions

## Directory Map

```text
scripts/
  train_pipeline.py   # Trains and saves versioned model + reports
  promote_model.py    # Updates production model pointer
  predict.py          # Batch prediction using production model or override path

src/churn/
  data.py             # Load + clean data
  features.py         # Feature/target split + preprocessing pipeline builder
  split.py            # Train/test split helper
  train.py            # Logistic regression trainer
  evaluate.py         # Classification metrics
  io.py               # Save/load helpers + run metadata utilities
```

## End-to-End Flow (Step-by-Step + Why)

1. Run `scripts/train_pipeline.py`.
Reason: this is the canonical training entry point for the current version.

2. Load raw dataset from `data/raw/Telco_customer_churn.csv`.
Reason: keeps training on a fixed source path for consistent runs.

3. Split features/target using target column `Churn Value`.
Reason: preserves one clear binary classification label contract.

4. Create train/test split with stratification.
Reason: keeps class distribution stable across train/test, improving evaluation reliability.

5. Build preprocessing pipeline (`cleaner` + `ColumnTransformer`) and fit on training data.
Reason: avoids leakage and ensures prediction-time transformations match training-time logic.

6. Train logistic regression model on transformed training data.
Reason: establishes a strong and interpretable baseline classifier.

7. Evaluate model on transformed test data.
Reason: captures baseline performance with accuracy, precision, recall, F1, and confusion matrix.

8. Save versioned artifacts using a UTC timestamp run ID (`YYYYMMDD_HHMMSS`).
Reason: enables reproducibility, comparison, and controlled promotion.

## Artifacts Produced by Training

For run ID `<RUN_ID>`:

- Model: `artifacts/models/<RUN_ID>/model.joblib`
- Metrics: `artifacts/reports/<RUN_ID>/metrics.json`
- Run metadata: `artifacts/reports/<RUN_ID>/run.json`

`run.json` includes:

- `run_id`
- `model_path`
- `metrics_path`
- `data_path`
- `data_sha256`
- `target_col`
- `git_commit` (if available)

## Production Promotion Flow

Script: `scripts/promote_model.py --run_id <RUN_ID>`

Step-by-step:

1. Validate `artifacts/models/<RUN_ID>/model.joblib` exists.
Reason: prevents promoting missing or invalid runs.

2. Write `artifacts/models/production.json` with `run_id` and `model_path`.
Reason: gives prediction a stable production pointer decoupled from training.

## Prediction Flow

Script: `scripts/predict.py`

Path selection order:

1. If `--model` is provided, use it.
Reason: supports testing or backfills against a specific artifact.

2. Otherwise read `artifacts/models/production.json`.
Reason: defaults to deployed/approved model.

Prediction behavior:

- Reads CSV input into DataFrame
- Uses `pipeline.predict_proba(df)[:, 1]` for churn probability
- Applies threshold `0.5` for binary class
- Writes output CSV with:
  - `churn_probability`
  - `churn_prediction`

## Core Module Contracts

- `src/churn/data.py`
  - `load_data(path) -> DataFrame`
  - `clean_data(df) -> DataFrame`
- `src/churn/features.py`
  - `split_features_target(df, target_col) -> (X, y)`
  - `preprocessing_pipeline(X, clean_data_func) -> Pipeline`
- `src/churn/split.py`
  - `make_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)`
- `src/churn/train.py`
  - `train_logreg_model(X_train, y_train, random_state=42)`
- `src/churn/evaluate.py`
  - `evaluate_model(model, X_test, y_test) -> metrics dict`
- `src/churn/io.py`
  - `save_model(model, path)`
  - `save_json(obj, path)`
  - `utc_now_compact()`
  - `sha256_file(path)`
  - `git_commit_hash(project_root)`
