import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.analysis.psm import estimate_propensity_score, match_nearest_neighbor


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    treatment = (rng.random(n) < 0.4).astype(int)
    return pd.DataFrame({"treatment": treatment, "x1": x1, "x2": x2})


def test_propensity_score_length(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    assert len(ps) == len(sample_df)


def test_propensity_score_range(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    assert (ps >= 0).all() and (ps <= 1).all()


def test_propensity_score_returns_model(sample_df):
    _, model = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    assert isinstance(model, LogisticRegression)


def test_propensity_score_single_covariate(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1"])
    assert len(ps) == len(sample_df)


def test_match_returns_dataframe(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    matched = match_nearest_neighbor(sample_df, ps, "treatment")
    assert isinstance(matched, pd.DataFrame)


def test_match_adds_propensity_score_column(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    matched = match_nearest_neighbor(sample_df, ps, "treatment")
    assert "propensity_score" in matched.columns


def test_match_caliper_reduces_sample(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    matched_wide = match_nearest_neighbor(sample_df, ps, "treatment", caliper=1.0)
    matched_tight = match_nearest_neighbor(sample_df, ps, "treatment", caliper=0.001)
    assert len(matched_tight) <= len(matched_wide)


def test_match_original_df_unchanged(sample_df):
    original_cols = list(sample_df.columns)
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    match_nearest_neighbor(sample_df, ps, "treatment")
    assert list(sample_df.columns) == original_cols


def test_match_contains_both_groups(sample_df):
    ps, _ = estimate_propensity_score(sample_df, "treatment", ["x1", "x2"])
    matched = match_nearest_neighbor(sample_df, ps, "treatment", caliper=1.0)
    assert 0 in matched["treatment"].values
    assert 1 in matched["treatment"].values
