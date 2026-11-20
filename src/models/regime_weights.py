"""Sample weights that up-weight rare weather regimes during training.

The flags use the FORECAST weather the sample sees (the same inputs the model gets), so they
are available at issue time. Thresholds are quantiles of the training rows only.

    hot           tmax_ai or heat_index_ai in the top 10 %
    cold/fog      North-India tmin in the bottom 10 %, or North fog proxy >= 0.3
    storm/heavy   all-India rain, 3-day rain or wind gusts in the top 5 %
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {  # name -> list of (column, quantile, direction)
    "hot": [("tmax_ai", 0.90, ">="), ("heat_index_ai", 0.90, ">=")],
    "cold_fog": [("tmin_nr", 0.10, "<="), ("fog_nr", None, ">=0.3")],
    "storm": [("precip_ai", 0.95, ">="), ("rain3_ai", 0.95, ">="), ("gust_max_ai", 0.95, ">=")],
}


def fit_thresholds(X: pd.DataFrame) -> dict:
    th = {}
    for name, rules in RULES.items():
        for col, q, how in rules:
            if col not in X:
                continue
            if q is None:
                th[(name, col)] = (how[:2], float(how[2:]))
            else:
                v = X[col].to_numpy(float)
                v = v[~np.isnan(v)]
                if len(v):
                    th[(name, col)] = (how, float(np.quantile(v, q)))
    return th


def flags(X: pd.DataFrame, th: dict) -> pd.DataFrame:
    out = pd.DataFrame(False, index=X.index, columns=list(RULES))
    for (name, col), (how, v) in th.items():
        x = X[col].to_numpy(float)
        hit = (x >= v) if how == ">=" else (x <= v)
        out[name] |= np.nan_to_num(hit, nan=False).astype(bool)
    return out


def regime_weights(X: pd.DataFrame, th: dict, boost: float) -> np.ndarray:
    """1 + boost for rows in any rare regime, rescaled to mean 1 (keeps the loss scale)."""
    if not boost:
        return np.ones(len(X))
    w = 1.0 + boost * flags(X, th).any(axis=1).to_numpy(float)
    return w / w.mean()
