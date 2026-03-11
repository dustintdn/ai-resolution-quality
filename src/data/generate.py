"""Synthetic conversation data generator.

Simulates a customer-support dataset with an AI-assistance rollout:
- Pre-period: all agents are human-only
- Post-period: ~50 % of interactions are AI-assisted (controllable via ai_rate)

Confounders: issue_severity, customer_tenure, time_of_day, agent_experience
Outcomes:    resolution_time, satisfaction_score, escalated
"""
from __future__ import annotations

import numpy as np
import pandas as pd


COVARIATES = ["issue_severity", "customer_tenure", "time_of_day", "agent_experience"]


def generate_synthetic_conversations(
    n: int = 2000,
    ai_rate: float = 0.5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic support conversations.

    Parameters
    ----------
    n:
        Number of rows to generate.
    ai_rate:
        Fraction of post-period conversations that receive AI assistance.
        Ignored in the pre-period (all human-only).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        conversation_id, created_at, period (0=pre/1=post),
        ai_assisted, issue_severity, customer_tenure, time_of_day,
        agent_experience, resolution_time, satisfaction_score, escalated.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Timeline: pre-period is 2023, post-period is 2024
    # ------------------------------------------------------------------
    pre_n = n // 2
    post_n = n - pre_n

    pre_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=pre_n), unit="D"
    )
    post_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=post_n), unit="D"
    )
    created_at = pd.concat(
        [pd.Series(pre_dates), pd.Series(post_dates)], ignore_index=True
    )
    period = np.concatenate([np.zeros(pre_n, dtype=int), np.ones(post_n, dtype=int)])

    # ------------------------------------------------------------------
    # Treatment assignment (confounded by covariates)
    # ------------------------------------------------------------------
    issue_severity = rng.integers(1, 6, size=n)          # 1–5 scale
    customer_tenure = rng.integers(0, 121, size=n)        # months (0–120)
    time_of_day = rng.integers(0, 24, size=n)             # hour
    agent_experience = rng.integers(1, 11, size=n)        # years (1–10)

    # AI assistance only available post-period; higher severity ↑ chance
    logit = (
        -1.0
        + 0.3 * (issue_severity - 3)
        - 0.01 * (customer_tenure - 60)
        + 0.05 * (agent_experience - 5)
    )
    prob_ai = 1 / (1 + np.exp(-logit))
    # Only post-period rows can be treated; apply ai_rate as a scale
    ai_assisted = np.where(
        period == 1,
        (rng.random(n) < prob_ai * (ai_rate / 0.5)).astype(int),
        0,
    )

    # ------------------------------------------------------------------
    # Outcomes (structural equations)
    # ------------------------------------------------------------------
    noise_rt = rng.normal(0, 10, size=n)
    resolution_time = (
        60.0
        + 8.0 * issue_severity
        - 15.0 * ai_assisted         # AI reduces resolution time
        - 0.1 * customer_tenure
        - 2.0 * agent_experience
        + noise_rt
    ).clip(5)

    noise_sat = rng.normal(0, 0.5, size=n)
    satisfaction_score = (
        3.0
        - 0.3 * issue_severity
        + 0.8 * ai_assisted          # AI improves satisfaction
        + 0.01 * customer_tenure
        + 0.1 * agent_experience
        + noise_sat
    ).clip(1, 5)

    esc_logit = (
        -2.0
        + 0.5 * issue_severity
        - 1.0 * ai_assisted
        - 0.2 * agent_experience
    )
    escalated = (rng.random(n) < 1 / (1 + np.exp(-esc_logit))).astype(int)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "conversation_id": np.arange(1, n + 1),
            "created_at": created_at,
            "period": period,
            "ai_assisted": ai_assisted,
            "issue_severity": issue_severity,
            "customer_tenure": customer_tenure,
            "time_of_day": time_of_day,
            "agent_experience": agent_experience,
            "resolution_time": resolution_time.round(1),
            "satisfaction_score": satisfaction_score.round(2),
            "escalated": escalated,
        }
    )
    return df.sort_values("created_at").reset_index(drop=True)


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_conversations(n=2000)
    out = "data/synthetic_conversations.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    print(df.head())
