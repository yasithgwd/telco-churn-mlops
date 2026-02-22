import pandas as pd
from typing import Tuple 
from sklearn.compose import ColumnTransformer
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


def preprocessing_pipeline(X: pd.DataFrame, clean_data_func) -> Pipeline:
    """
    Build end-to-end preprocessing pipeline
    - Step 1: Clean (drop leakage/unwanted columns)
    - Step 2: Numeric (impute + scale)
    - Step 3: Categorical (impute + one-hot)
    """
    # First, figure out which columns to keep after cleaning
    X_clean = clean_data_func(X)
    
    numeric_cols = X_clean.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full pipeline: cleaning + preprocessing
    full_pipeline = Pipeline([
        ("cleaner", FunctionTransformer(func=clean_data_func, validate=False)),
        ("preprocessor", ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]))
    ])

    return full_pipeline
