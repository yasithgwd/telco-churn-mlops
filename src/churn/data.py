import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load raw data from a CSV file"""
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning ops"""
    df = df.copy()

    # Drop Leakage Columns
    drop_cols = ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    drop_unwanted_cols = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Latitude", "Longitude", "Lat Long"]
    df = df.drop(columns=[c for c in drop_unwanted_cols if c in df.columns])

    # Fix Total Charges: empty strings are not NaN, so coerce then fill
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)

    return df
