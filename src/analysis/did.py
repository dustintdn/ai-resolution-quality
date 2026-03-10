"""
Difference-in-Differences (DiD) estimators.

This module implements two approaches to DiD estimation: a simple group-means
estimator and a regression-based estimator with an interaction term, both
supporting an arbitrary pre/post period cutoff.
"""

import pandas as pd
import statsmodels.formula.api as smf


def difference_in_differences(df: pd.DataFrame, time_col: str, period_cutoff, treatment_col: str, outcome_col: str):
    """Estimate the average treatment effect using a simple DiD on group means.

    Splits observations into pre/post periods based on ``period_cutoff``, then
    computes the DiD estimate as:
    (mean_treated_post - mean_treated_pre) - (mean_control_post - mean_control_pre)

    Args:
        df: DataFrame containing time, treatment, and outcome columns.
        time_col: Name of the column representing the time dimension.
        period_cutoff: Threshold value in ``time_col`` that divides pre (< cutoff)
            from post (>= cutoff) periods. Can be a datetime, int, or float.
        treatment_col: Name of the binary treatment column (0/1).
        outcome_col: Name of the outcome column to analyze.

    Returns:
        did: The scalar DiD estimate of the average treatment effect.
        cell_means: Dict with keys ``treated_pre``, ``treated_post``,
            ``control_pre``, ``control_post`` containing the group means used
            to compute the estimate.
    """
    df = df.copy()
    df["period"] = (df[time_col] >= period_cutoff).astype(int)

    grp = df.groupby([treatment_col, "period"]).agg(mean_outcome=(outcome_col, "mean")).reset_index()
    # pivot
    pivot = grp.pivot(index=treatment_col, columns="period", values="mean_outcome")
    # treated: index 1, control: index 0
    treated_pre = pivot.loc[1, 0]
    treated_post = pivot.loc[1, 1]
    control_pre = pivot.loc[0, 0]
    control_post = pivot.loc[0, 1]

    did = (treated_post - treated_pre) - (control_post - control_pre)
    return did, {
        "treated_pre": treated_pre,
        "treated_post": treated_post,
        "control_pre": control_pre,
        "control_post": control_post,
    }


def regression_did(df: pd.DataFrame, time_col: str, period_cutoff, treatment_col: str, outcome_col: str, covariates: list | None = None):
    """Estimate a regression-based DiD using an OLS model with an interaction term.

    Fits the model:
    ``outcome ~ treatment + period + treatment:period [+ covariates]``
    using HC1 heteroskedasticity-robust standard errors. The coefficient on
    ``treatment:period`` is the DiD estimate of the average treatment effect.

    Args:
        df: DataFrame containing time, treatment, outcome, and optional covariate columns.
        time_col: Name of the column representing the time dimension.
        period_cutoff: Threshold value in ``time_col`` that divides pre (< cutoff)
            from post (>= cutoff) periods. Can be a datetime, int, or float.
        treatment_col: Name of the binary treatment column (0/1).
        outcome_col: Name of the outcome column to model.
        covariates: Optional list of additional covariate column names to include
            as controls in the regression. Defaults to None.

    Returns:
        model: The fitted statsmodels OLS results object.
        coef: The DiD coefficient for the ``treatment:period`` interaction term,
            or None if the term is not present in the model.
    """
    df = df.copy()
    df["period"] = (df[time_col] >= period_cutoff).astype(int)
    # build formula: outcome ~ ai_assisted + period + ai_assisted:period + covariates
    terms = [treatment_col, "period", f"{treatment_col}:period"]
    if covariates:
        terms += covariates
    formula = f"{outcome_col} ~ " + " + ".join(terms)
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    coef_name = f"{treatment_col}:period"
    coef = model.params.get(coef_name, None)
    return model, coef

