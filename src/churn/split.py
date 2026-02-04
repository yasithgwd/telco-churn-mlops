from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split X and y into train/test. Uses stratification by default for classification.
    """
    stratify_y = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )