import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def split_features_target(
        df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target variable"""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def preprocessing_pipeline(clean_data_func) -> Pipeline:
    """
    Build end-to-end preprocessing pipeline
    - Step 1: Clean (drop leakage/unwanted columns)
    - Step 2: Numeric (impute + scale)
    - Step 3: Categorical (impute + one-hot)
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # make_column_selector defers column detection to fit time, after the
    # cleaner step has already run — avoids calling clean_data_func twice.
    full_pipeline = Pipeline([
        ("cleaner", FunctionTransformer(func=clean_data_func, validate=False)),
        ("preprocessor", ColumnTransformer([
            ("num", numeric_transformer, make_column_selector(dtype_include="number")),
            ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "category", "bool"]))
        ]))
    ])

    return full_pipeline
