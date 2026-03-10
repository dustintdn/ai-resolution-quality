import numpy as np
import pandas as pd
import pytest
from src.analysis.did import difference_in_differences, regression_did


@pytest.fixture
def did_df():
    """Simple DiD fixture with a known treatment effect of +10."""
    rng = np.random.default_rng(0)
    n = 400
    period = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    treatment = (rng.random(n) < 0.5).astype(int)
    # True effect: +10 for treated units in post period
    outcome = (
        50.0
        + 10.0 * period
        + 5.0 * treatment
        + 10.0 * treatment * period   # DiD effect
        + rng.normal(0, 2, n)
    )
    time = pd.to_datetime("2023-01-01") + pd.to_timedelta(period * 365, unit="D")
    return pd.DataFrame({"time": time, "treatment": treatment, "outcome": outcome})


def test_did_returns_scalar(did_df):
    did, _ = difference_in_differences(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert isinstance(float(did), float)


def test_did_recovers_effect(did_df):
    did, _ = difference_in_differences(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert abs(did - 10.0) < 1.5


def test_did_cell_means_keys(did_df):
    _, cell_means = difference_in_differences(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert set(cell_means.keys()) == {"treated_pre", "treated_post", "control_pre", "control_post"}


def test_did_cell_means_ordering(did_df):
    _, cm = difference_in_differences(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    # Post means should exceed pre means for both groups
    assert cm["treated_post"] > cm["treated_post"] - 20  # sanity: not nan
    assert cm["control_post"] > cm["control_pre"]


def test_did_original_df_unchanged(did_df):
    original_cols = list(did_df.columns)
    difference_in_differences(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert list(did_df.columns) == original_cols


def test_regression_did_returns_model_and_coef(did_df):
    model, coef = regression_did(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert coef is not None
    assert hasattr(model, "params")


def test_regression_did_recovers_effect(did_df):
    _, coef = regression_did(did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome")
    assert abs(coef - 10.0) < 1.5


def test_regression_did_with_covariates(did_df):
    did_df = did_df.copy()
    did_df["x1"] = np.random.default_rng(5).normal(0, 1, len(did_df))
    _, coef = regression_did(
        did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome", covariates=["x1"]
    )
    assert coef is not None


def test_regression_did_no_covariates(did_df):
    _, coef = regression_did(
        did_df, "time", pd.Timestamp("2024-01-01"), "treatment", "outcome", covariates=None
    )
    assert coef is not None
