import numpy as np
import pandas as pd


def placebo_test(df: pd.DataFrame, treatment_col: str, outcome_col: str, n_runs: int = 100, seed: int | None = None):
    """Randomly permute treatment assignment to perform a placebo (permutation) test."""
    rng = np.random.default_rng(seed)
    orig_effect = df.groupby(treatment_col)[outcome_col].mean().diff().iloc[-1]
    null_effects = []
    for _ in range(n_runs):
        perm = df[treatment_col].sample(frac=1.0, replace=False, random_state=rng.integers(0, 1_000_000)).values
        df_perm = df.copy()
        df_perm["perm_treat"] = perm
        null_effects.append(df_perm.groupby("perm_treat")[outcome_col].mean().diff().iloc[-1])
    null_effects = np.array(null_effects)
    p_value = (np.abs(null_effects) >= abs(orig_effect)).mean()
    return orig_effect, null_effects, p_value


def e_value(rr: float) -> float:
    """Compute the E-value for a risk ratio `rr`.

    E-value formula: RR + sqrt(RR * (RR - 1)) for RR > 1.
    """
    if rr <= 1:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1))
