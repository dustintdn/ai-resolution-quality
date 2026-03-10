import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_propensity_score(df: pd.DataFrame, treatment_col: str, covariates: list):
    X = df[covariates].astype(float)
    y = df[treatment_col].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return ps, model


def match_nearest_neighbor(df: pd.DataFrame, ps: np.ndarray, treatment_col: str, caliper: float = 0.05):
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
