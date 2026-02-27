# Dataset Documentation

## Source

- Primary training file: `data/raw/Telco_customer_churn.csv`
- Optional spreadsheet source: `data/raw/Telco_customer_churn.xlsx`

## Shape and Target

- Rows: `7043`
- Columns: `33`
- Target column: `Churn Value` (binary)

Target distribution:

- `0`: `5174`
- `1`: `1869`

Reason this matters: the dataset is imbalanced, so preserving class ratio in train/test split is important.

## Raw Columns

`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Tenure Months`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`

## Cleaning Rules in Current Code

Defined in `src/churn/data.py` (`clean_data`):

1. Copy input DataFrame.
Reason: avoids in-place mutation side effects.

2. Drop leakage columns when present:
`Churn Label`, `Churn Score`, `CLTV`, `Churn Reason`
Reason: these columns directly encode or strongly leak churn outcomes.

3. Drop location/identifier columns when present:
`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Lat Long`
Reason: reduces identifiers and high-cardinality fields not required for baseline model behavior.

4. Fill missing `Total Charges` with `0` only if column `tenure` exists.
Reason: intended missing-value handling rule in code.

## Important Current Behavior Notes

- The raw schema uses `Tenure Months`, not `tenure`.
- Because of that mismatch, the current conditional fill for `Total Charges` usually does not run.
- The preprocessing stage still applies numeric median imputation, so missing numeric values are handled later in the pipeline.

## Feature/Target Split

Defined in `src/churn/features.py`:

- Features (`X`): all columns except `Churn Value`
- Target (`y`): `Churn Value`

Reason this matters: all modeling and preprocessing are built around this fixed target contract.

## Train/Test Split

Defined in `src/churn/split.py`:

- Test size: `0.2`
- Random state: `42`
- Stratified split: enabled by default

Reason this matters: reproducible and class-balanced evaluation.

## Preprocessing Pipeline

Defined in `src/churn/features.py`:

1. Cleaner step (`FunctionTransformer(clean_data)`).
Reason: applies the same cleaning during both training and inference.

2. Numeric branch:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
Reason: robust missing-value handling and scale normalization for linear models.

3. Categorical branch:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`
Reason: converts categories to numeric model input and prevents failures on unseen categories.

## Training Outputs

For each training run `<RUN_ID>`:

- Model artifact: `artifacts/models/<RUN_ID>/model.joblib`
- Metrics report: `artifacts/reports/<RUN_ID>/metrics.json`
- Run metadata: `artifacts/reports/<RUN_ID>/run.json`

Production pointer:

- `artifacts/models/production.json`

Reason this matters: versioned artifacts support reproducibility; production pointer supports controlled deployment selection.
