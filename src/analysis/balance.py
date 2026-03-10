import pandas as pd
import numpy as np


def compute_smd(df: pd.DataFrame, treatment_col: str, covariates: list) -> pd.Series:
    """Compute standardized mean differences (SMD) for covariates between treatment groups.

    SMD = (mean_treated - mean_control) / pooled_std
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
    """Return a table with means and SMD for covariates."""
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
