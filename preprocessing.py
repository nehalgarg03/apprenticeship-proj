import pandas as pd

def load_and_clean_geographic_data(csv_path, drop_cols=None):
    """
    Load and clean census-style geographic data.
    """
    df = pd.read_csv(csv_path)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Convert numeric columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def aggregate_by_region(df, region_col):
    """
    Aggregate numeric features by a geographic region.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    return df.groupby(region_col)[numeric_cols].mean().reset_index()
