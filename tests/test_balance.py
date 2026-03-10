import numpy as np
import pandas as pd
import pytest
from src.analysis.balance import compute_smd, balance_table


@pytest.fixture
def balanced_df():
    """Two groups with identical distributions — SMD should be ~0."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "treatment": np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
    })


@pytest.fixture
def imbalanced_df():
    """Treated group has much higher x1 mean — SMD should be large."""
    rng = np.random.default_rng(1)
    treated = pd.DataFrame({"treatment": 1, "x1": rng.normal(10, 1, 100)})
    control = pd.DataFrame({"treatment": 0, "x1": rng.normal(0, 1, 100)})
    return pd.concat([treated, control], ignore_index=True)


def test_smd_balanced_near_zero(balanced_df):
    smd = compute_smd(balanced_df, "treatment", ["x1", "x2"])
    assert abs(smd["x1"]) < 0.3
    assert abs(smd["x2"]) < 0.3


def test_smd_imbalanced_large(imbalanced_df):
    smd = compute_smd(imbalanced_df, "treatment", ["x1"])
    assert abs(smd["x1"]) > 5.0


def test_smd_zero_variance():
    df = pd.DataFrame({"treatment": [1, 1, 0, 0], "x": [3.0, 3.0, 3.0, 3.0]})
    smd = compute_smd(df, "treatment", ["x"])
    assert smd["x"] == 0.0


def test_smd_returns_series(balanced_df):
    smd = compute_smd(balanced_df, "treatment", ["x1", "x2"])
    assert isinstance(smd, pd.Series)
    assert set(smd.index) == {"x1", "x2"}


def test_balance_table_shape(balanced_df):
    tbl = balance_table(balanced_df, "treatment", ["x1", "x2"])
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.shape == (2, 4)
    assert list(tbl.columns) == ["covariate", "mean_treated", "mean_control", "smd"]


def test_balance_table_covariates(balanced_df):
    tbl = balance_table(balanced_df, "treatment", ["x1", "x2"])
    assert set(tbl["covariate"]) == {"x1", "x2"}


def test_balance_table_means_correct():
    df = pd.DataFrame({
        "treatment": [1, 1, 0, 0],
        "x": [4.0, 6.0, 1.0, 3.0],
    })
    tbl = balance_table(df, "treatment", ["x"])
    assert tbl.loc[0, "mean_treated"] == pytest.approx(5.0)
    assert tbl.loc[0, "mean_control"] == pytest.approx(2.0)
