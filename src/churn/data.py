import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load raw data from a CSV file"""
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning ops"""
    df = df.copy()

    # Drop Leakage Columns
    drop_cols = ["Churn Label", "Churn Score", "CLTV"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fix totlal charges isuue
    if "Total Charges" in df.columns and "tenure" in df.columns:
        df["Total Charges"] = df["Total Charges"].fillna(0)

    return df
