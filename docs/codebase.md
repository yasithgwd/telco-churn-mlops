# Codebase Documentation

## Overview

This project follows a simple split between:

- `scripts/`: runnable entry points
- `src/churn/`: reusable modules for data, features, train, evaluate, and I/O
- `data/`: raw and processed datasets
- `artifacts/`: saved models, reports, and predictions

## Directory Map

```text
scripts/
  train.py         # Training pipeline runner
  predict.py       # Batch prediction runner (pipeline artifact path)

src/churn/
  data.py          # Load + clean raw data
  features.py      # Feature/target split + preprocessing pipeline
  split.py         # Train/test split helper
  train.py         # Logistic regression training function
  evaluate.py      # Classification metrics
  io.py            # Model/JSON persistence helpers
  config.py        # Alternate prediction CLI-style implementation
```

## Training Flow

Entry point: `scripts/train.py`

Sequence:

1. Adds `src` to `sys.path` for imports.
2. Loads `data/raw/Telco_customer_churn.csv`.
3. Cleans data using `src/churn/data.py`.
4. Splits into features/target (`Churn Value`).
5. Performs train/test split (stratified).
6. Builds preprocessing pipeline and transforms train/test sets.
7. Trains logistic regression model from `src/churn/train.py`.
8. Evaluates with accuracy, precision, recall, F1, confusion matrix.
9. Saves:
   - model to `artifacts/models/logreg.joblib`
   - metrics to `artifacts/reports/metrics.json`

## Prediction Paths

There are currently two prediction implementations:

- `scripts/predict.py`: Expects a single serialized pipeline artifact (`telco_churn_pipeline.joblib`) and predicts directly on raw-style input columns.
- `src/churn/config.py`: Loads separate model/preprocessor/feature-columns artifacts and runs a batch prediction routine.

Because these paths use different artifact assumptions, they should be treated as separate workflows unless unified.

## Core Module Contracts

- `src/churn/data.py`
  - `load_data(path) -> DataFrame`
  - `clean_data(df) -> DataFrame`
- `src/churn/features.py`
  - `split_features_traget(df, target_col) -> (X, y)`
  - `preprocessing_pipeline(X) -> ColumnTransformer`
- `src/churn/split.py`
  - `make_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)`
- `src/churn/train.py`
  - `train_logreg_model(X_train, y_train, random_state=42)`
- `src/churn/evaluate.py`
  - `evaluate_model(model, X_test, y_test) -> metrics dict`
- `src/churn/io.py`
  - `save_model(model, path)`
  - `save_json(obj, path)`

## Current Caveats

- `scripts/train.py` loads data twice before cleaning; one load is redundant.
- The function name `split_features_traget` contains a typo and is used consistently with that name.
- `scripts/predict.py` artifact expectations do not match `scripts/train.py` outputs by default.
