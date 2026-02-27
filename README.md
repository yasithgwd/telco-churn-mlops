# telco-churn-mlops

Telco customer churn prediction project using a logistic regression baseline and a modular training pipeline.

## Documentation

- Dataset guide: `docs/dataset.md`
- Codebase guide: `docs/codebase.md`

## Current Version Workflow

The project currently uses three scripts:

- `scripts/train_pipeline.py`: trains a model and saves versioned artifacts
- `scripts/promote_model.py`: marks one trained run as production
- `scripts/predict.py`: runs batch predictions using the production model (or an explicit model path)

## Quickstart (Step-by-Step + Why)

1. Install dependencies.
Reason: ensures `pandas`, `numpy`, and `scikit-learn` are available.

```bash
pip install -r requirements.txt
```

2. Train a new model pipeline.
Reason: creates a timestamped model artifact and matching reports for traceability.

```bash
python3 scripts/train_pipeline.py
```

3. Find the new run ID.
Reason: promotion requires selecting a specific trained run.

```bash
ls artifacts/models
```

4. Promote that run to production.
Reason: prediction without `--model` reads `artifacts/models/production.json`.

```bash
python3 scripts/promote_model.py --run_id <RUN_ID>
```

5. Run predictions on a CSV file.
Reason: appends `churn_probability` and `churn_prediction` columns to output.

```bash
python3 scripts/predict.py --input input.csv --output predictions.csv
```

Optional: use a specific model artifact directly (without production pointer).

```bash
python3 scripts/predict.py --model artifacts/models/<RUN_ID>/model.joblib --input input.csv --output predictions.csv
```

## Repository Layout

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── models/
│   ├── reports/
│   └── predictions/
├── scripts/
│   ├── train_pipeline.py
│   ├── promote_model.py
│   └── predict.py
└── src/churn/
    ├── data.py
    ├── features.py
    ├── split.py
    ├── train.py
    ├── evaluate.py
    └── io.py
```
