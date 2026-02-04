import pandas as pd
from typing import Tuple 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def split_features_traget(
        df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target variable"""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y

def preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing transformer
    - Numeric: impute + scale 
    - categorical : impute + one-hot encode
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor
