# telco-churn-mlops

Telco customer churn prediction project using a logistic regression baseline and a modular training pipeline.

## Documentation

- Dataset guide: `docs/dataset.md`
- Codebase guide: `docs/codebase.md`

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python3 scripts/train.py
```

Run predictions (script-based pipeline artifact):

```bash
python3 scripts/predict.py --model telco_churn_pipeline.joblib --input input.csv --output predictions.csv
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
│   ├── train.py
│   └── predict.py
└── src/churn/
    ├── data.py
    ├── features.py
    ├── split.py
    ├── train.py
    ├── evaluate.py
    ├── io.py
    └── config.py
```
