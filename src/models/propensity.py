"""Propensity score models.

Wraps sklearn estimators with a consistent interface used by the PSM pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    max_iter: int = 1000,
) -> tuple[np.ndarray, Pipeline]:
    """Fit a logistic regression propensity model.

    Returns
    -------
    ps : np.ndarray
        Estimated propensity scores (probability of treatment).
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (scaler + logistic regression).
    """
    X = df[covariates].astype(float)
    y = df[treatment_col].astype(int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter)),
        ]
    )
    pipe.fit(X, y)
    ps = pipe.predict_proba(X)[:, 1]
    return ps, pipe


def gradient_boosting_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    n_estimators: int = 100,
    calibrate: bool = True,
) -> tuple[np.ndarray, object]:
    """Fit a gradient-boosted propensity model with optional isotonic calibration.

    Returns
    -------
    ps : np.ndarray
        Calibrated propensity scores.
    model : estimator
        Fitted (possibly calibrated) model.
    """
    X = df[covariates].astype(float)
    y = df[treatment_col].astype(int)
    base = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    if calibrate:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return ps, model
