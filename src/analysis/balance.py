"""
Covariate balance diagnostics for propensity score matching.

This module provides functions for assessing pre- and post-matching balance
between treatment and control groups via standardized mean differences (SMD)
and summary balance tables.
"""

import pandas as pd
import numpy as np


def compute_smd(df: pd.DataFrame, treatment_col: str, covariates: list) -> pd.Series:
    """Compute standardized mean differences (SMD) for covariates between treatment groups.

    SMD is defined as (mean_treated - mean_control) / pooled_std, where
    pooled_std = sqrt((var_treated + var_control) / 2). Values near zero
    indicate good balance; |SMD| > 0.1 is commonly flagged as imbalanced.

    Args:
        df: DataFrame containing the treatment column and all covariate columns.
        treatment_col: Name of the binary treatment column (0/1).
        covariates: List of covariate column names to evaluate.

    Returns:
        Series indexed by covariate name containing the SMD for each variable.
        Covariates with zero pooled standard deviation are assigned SMD of 0.0.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    smd = {}
    for v in covariates:
        t_mean = treated[v].astype(float).mean()
        c_mean = control[v].astype(float).mean()
        t_var = treated[v].astype(float).var()
        c_var = control[v].astype(float).var()
        pooled_sd = np.sqrt((t_var + c_var) / 2)
        # avoid division by zero
        if pooled_sd == 0:
            smd[v] = 0.0
        else:
            smd[v] = (t_mean - c_mean) / pooled_sd
    return pd.Series(smd)


def balance_table(df: pd.DataFrame, treatment_col: str, covariates: list) -> pd.DataFrame:
    """Build a summary balance table comparing treated and control group means.

    For each covariate, reports the mean in the treated group, the mean in the
    control group, and the standardized mean difference (SMD) as a single
    diagnostic row.

    Args:
        df: DataFrame containing the treatment column and all covariate columns.
        treatment_col: Name of the binary treatment column (0/1).
        covariates: List of covariate column names to include in the table.

    Returns:
        DataFrame with one row per covariate and columns:
        ``covariate``, ``mean_treated``, ``mean_control``, ``smd``.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    rows = []
    for v in covariates:
        rows.append(
            {
                "covariate": v,
                "mean_treated": treated[v].astype(float).mean(),
                "mean_control": control[v].astype(float).mean(),
                "smd": compute_smd(df, treatment_col, [v]).iloc[0],
            }
        )
    return pd.DataFrame(rows)
