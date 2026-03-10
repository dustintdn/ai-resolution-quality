import pandas as pd
import pytest
from src.data.generate import generate_synthetic_conversations, COVARIATES


def test_generate_length():
    df = generate_synthetic_conversations(500, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_generate_columns():
    df = generate_synthetic_conversations(100, seed=0)
    expected = {
        "conversation_id", "created_at", "period", "ai_assisted",
        "issue_severity", "customer_tenure", "time_of_day",
        "agent_experience", "resolution_time", "satisfaction_score", "escalated",
    }
    assert expected.issubset(set(df.columns))


def test_generate_reproducible():
    df1 = generate_synthetic_conversations(200, seed=7)
    df2 = generate_synthetic_conversations(200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds():
    df1 = generate_synthetic_conversations(200, seed=1)
    df2 = generate_synthetic_conversations(200, seed=2)
    assert not df1["resolution_time"].equals(df2["resolution_time"])


def test_pre_period_no_ai():
    df = generate_synthetic_conversations(1000, seed=42)
    pre = df[df["period"] == 0]
    assert (pre["ai_assisted"] == 0).all(), "Pre-period rows must not be AI-assisted"


def test_outcome_ranges():
    df = generate_synthetic_conversations(1000, seed=42)
    assert (df["resolution_time"] >= 5).all()
    assert (df["satisfaction_score"] >= 1).all()
    assert (df["satisfaction_score"] <= 5).all()
    assert df["escalated"].isin([0, 1]).all()


def test_period_binary():
    df = generate_synthetic_conversations(200, seed=42)
    assert df["period"].isin([0, 1]).all()


def test_covariates_constant():
    assert COVARIATES == ["issue_severity", "customer_tenure", "time_of_day", "agent_experience"]
