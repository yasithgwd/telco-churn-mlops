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
    # df = df.drop(columns=[c for c in drop_unwanted_cols if c in df.columns])

    # Create an empty list to store columns that actually exist in the DataFrame
    columns_to_drop = []

    # Loop through the unwanted columns list
    for col in drop_unwanted_cols:
        # Check if the column exists in the DataFrame
        if col in df.columns:
            columns_to_drop.append(col)
        else:
            print(col, " not found in the dataset")

    # Drop only the existing unwanted columns
    df = df.drop(columns=columns_to_drop)

    # Fix totlal charges isuue
    if "Total Charges" in df.columns and "tenure" in df.columns:
        df["Total Charges"] = df["Total Charges"].fillna(0)

    return df
