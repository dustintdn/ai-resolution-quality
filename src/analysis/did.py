import pandas as pd
import statsmodels.formula.api as smf


def difference_in_differences(df: pd.DataFrame, time_col: str, period_cutoff, treatment_col: str, outcome_col: str):
    """Simple DiD estimator using group means.

    period_cutoff: a value in `time_col` that splits pre/post (e.g., a datetime)
    Returns the DiD estimate (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
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
    """Run a regression-based DiD with an interaction term.

    Returns the fitted model and the DiD coefficient on `treatment:period`.
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

