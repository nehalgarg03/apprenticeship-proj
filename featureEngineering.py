import pandas as pd
import numpy as np


def clean_location_column(df, location_col="NAME"):
    """
    Extract county and state from 'County; State' formatted strings.
    """
    df = df.copy()
    df["county"] = df[location_col].str.split(";").str[0].str.strip()
    df["state"] = df[location_col].str.split(";").str[1].str.strip()
    return df.drop(columns=[location_col])


def normalize_population_features(df, population_cols):
    """
    Convert raw population counts into proportions to avoid scale bias.
    """
    df = df.copy()
    total_pop = df[population_cols].sum(axis=1)

    for col in population_cols:
        df[f"{col}_ratio"] = df[col] / total_pop.replace(0, np.nan)

    return df



def compute_education_rates(
    df,
    total_pop_col,
    hs_grad_col,
    college_grad_col
):
    """
    Compute standardized education outcome metrics.
    """
    df = df.copy()

    df["hs_grad_rate"] = df[hs_grad_col] / df[total_pop_col].replace(0, np.nan)
    df["college_grad_rate"] = df[college_grad_col] / df[total_pop_col].replace(0, np.nan)

    return df


def aggregate_by_geography(df, group_cols):
    """
    Aggregate numeric features by geography (county, city, state).
    """
    numeric_cols = df.select_dtypes(include="number").columns
    return (
        df.groupby(group_cols)[numeric_cols]
        .mean()
        .reset_index()
    )



def build_feature_matrix(
    df,
    location_col,
    population_cols,
    total_pop_col,
    hs_grad_col,
    college_grad_col,
    group_cols=["county", "state"]
):
    """
    End-to-end feature engineering pipeline.
    """
    df = clean_location_column(df, location_col)
    df = normalize_population_features(df, population_cols)
    df = compute_education_rates(
        df,
        total_pop_col,
        hs_grad_col,
        college_grad_col
    )
    df = aggregate_by_geography(df, group_cols)

    # Drop original raw count columns
    df = df.drop(columns=population_cols, errors="ignore")

    return df
