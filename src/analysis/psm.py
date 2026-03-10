"""
Propensity Score Matching (PSM) utilities.

This module provides functions for estimating propensity scores via logistic
regression and performing nearest-neighbor matching with a caliper, enabling
causal inference on observational data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_propensity_score(df: pd.DataFrame, treatment_col: str, covariates: list):
    """Estimate propensity scores using logistic regression.

    Fits a logistic regression model predicting treatment assignment from the
    given covariates and returns the predicted probability of treatment for
    each observation.

    Args:
        df: Input DataFrame containing treatment and covariate columns.
        treatment_col: Name of the binary treatment column (0/1).
        covariates: List of column names to use as model features.

    Returns:
        ps: Array of propensity scores (probability of treatment) for each row.
        model: The fitted LogisticRegression model.
    """
    X = df[covariates].astype(float)
    y = df[treatment_col].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return ps, model


def match_nearest_neighbor(df: pd.DataFrame, ps: np.ndarray, treatment_col: str, caliper: float = 0.05):
    """Match treated units to controls using nearest-neighbor propensity score matching.

    For each treated unit, finds the nearest control unit by propensity score.
    Matches are only accepted if the distance is within the caliper threshold.
    Unmatched treated units (outside caliper) are excluded from the result.

    Args:
        df: Input DataFrame containing the treatment column and all covariates.
        ps: Array of propensity scores aligned with df's row order.
        treatment_col: Name of the binary treatment column (0/1).
        caliper: Maximum allowable propensity score distance for a valid match.
            Defaults to 0.05.

    Returns:
        DataFrame of matched treated and control units concatenated together,
        with a ``propensity_score`` column added. Row index is reset.
    """
    df = df.copy()
    df["propensity_score"] = ps

    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])

    matched_idx = []
    for i, d in enumerate(distances[:, 0]):
        if d <= caliper:
            matched_idx.append(control.index[indices[i, 0]])

    matched_control = control.loc[matched_idx]
    matched = pd.concat([treated, matched_control], ignore_index=True)
    return matched
