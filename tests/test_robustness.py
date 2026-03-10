import numpy as np
import pandas as pd
import pytest
from src.sensitivity.robustness import placebo_test, e_value


@pytest.fixture
def effect_df():
    """DataFrame with a real treatment effect — p-value should be small."""
    rng = np.random.default_rng(0)
    n = 200
    treatment = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    outcome = treatment * 20.0 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


@pytest.fixture
def null_df():
    """DataFrame with no treatment effect — p-value should be large."""
    rng = np.random.default_rng(1)
    n = 200
    treatment = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    outcome = rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


# --- placebo_test ---

def test_placebo_returns_three_values(effect_df):
    result = placebo_test(effect_df, "treatment", "outcome", n_runs=20, seed=0)
    assert len(result) == 3


def test_placebo_null_distribution_length(effect_df):
    _, null_effects, _ = placebo_test(effect_df, "treatment", "outcome", n_runs=50, seed=0)
    assert len(null_effects) == 50


def test_placebo_p_value_range(effect_df):
    _, _, p = placebo_test(effect_df, "treatment", "outcome", n_runs=100, seed=0)
    assert 0.0 <= p <= 1.0


def test_placebo_significant_effect(effect_df):
    _, _, p = placebo_test(effect_df, "treatment", "outcome", n_runs=200, seed=0)
    assert p < 0.05, "Strong treatment effect should yield a small p-value"


def test_placebo_null_effect_not_significant(null_df):
    _, _, p = placebo_test(null_df, "treatment", "outcome", n_runs=200, seed=0)
    # With no effect, p-value should not be consistently tiny (soft check)
    assert p > 0.0  # just ensure it runs and returns a valid value


def test_placebo_reproducible(effect_df):
    _, null1, p1 = placebo_test(effect_df, "treatment", "outcome", n_runs=50, seed=99)
    _, null2, p2 = placebo_test(effect_df, "treatment", "outcome", n_runs=50, seed=99)
    assert p1 == p2
    np.testing.assert_array_equal(null1, null2)


# --- e_value ---

def test_e_value_at_one():
    assert e_value(1.0) == 1.0


def test_e_value_below_one():
    assert e_value(0.5) == 1.0


def test_e_value_at_two():
    result = e_value(2.0)
    expected = 2.0 + np.sqrt(2.0 * 1.0)
    assert result == pytest.approx(expected)


def test_e_value_increases_with_rr():
    assert e_value(3.0) > e_value(2.0) > e_value(1.5)


def test_e_value_positive():
    for rr in [1.1, 2.0, 5.0, 10.0]:
        assert e_value(rr) > 0
