import pandas as pd
from typing import Tuple 

def split_features_traget(
        df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target variable"""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


