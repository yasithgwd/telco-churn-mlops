# Dataset Documentation

## Source

- Primary file: `data/raw/Telco_customer_churn.csv`
- Optional spreadsheet source: `data/raw/Telco_customer_churn.xlsx`

## Shape

- Rows: 7043
- Columns: 33
- Target column: `Churn Value` (binary label)

Target distribution (`Churn Value`):

- `0`: 5174
- `1`: 1869

## Raw Columns

`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Tenure Months`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`

## Data Cleaning Rules (Current Implementation)

Defined in `src/churn/data.py`:

- Drops leakage-like columns if present:
  - `Churn Label`, `Churn Score`, `CLTV`
- Drops location/identifier columns:
  - `Customer ID`, `Count`, `Country`, `State`, `City`, `Zip Code`
- Fills missing `Total Charges` with `0` if `tenure` exists

Notes:

- In the raw data, the customer ID column is `CustomerID` (no space), while cleaning drops `Customer ID` (with space). This means `CustomerID` is currently not dropped by that rule.
- The `tenure` check may not match the current raw schema, which uses `Tenure Months`.

## Feature and Target Split

Defined in `src/churn/features.py`:

- Features: all columns except `Churn Value`
- Target: `Churn Value`

## Training/Test Split

Defined in `src/churn/split.py`:

- Test size: `0.2`
- Random state: `42`
- Stratified split: enabled by default

## Preprocessing

Defined in `src/churn/features.py`:

- Numeric features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Most-frequent imputation
  - One-hot encoding (`handle_unknown="ignore"`)

## Outputs

- Model artifact: `artifacts/models/logreg.joblib`
- Metrics report: `artifacts/reports/metrics.json`
- Optional processed file present in repo: `data/processed/telco_churn_processed.csv`
