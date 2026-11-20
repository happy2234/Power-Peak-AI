"""Shared helpers: prediction format, quantile utilities, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import QUANTILES, TARGETS

QCOLS = ["p10", "p50", "p90"]
PRED_COLS = ["model", "origin", "date", "h", "target", *QCOLS, "actual"]


def sort_quantiles(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove quantile crossing by sorting along the quantile axis (P10 <= P50 <= P90)."""
    return np.sort(arr, axis=axis)


def tab_to_long(tab: pd.DataFrame, preds: dict[str, np.ndarray], model: str) -> pd.DataFrame:
    """tab: rows (origin, h, date, actual_<t>); preds: target -> [n, 3] levels."""
    frames = []
    for t in TARGETS:
        if t not in preds:
            continue
        q = sort_quantiles(preds[t])
        frames.append(pd.DataFrame({
            "model": model, "origin": tab["origin"].to_numpy(), "date": tab["date"].to_numpy(),
            "h": tab["h"].to_numpy(), "target": t, "p10": q[:, 0], "p50": q[:, 1], "p90": q[:, 2],
            "actual": tab[f"actual_{t}"].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def mean_pinball(df: pd.DataFrame) -> float:
    """Average pinball loss over P10/P50/P90 (same units as the target)."""
    y = df["actual"].to_numpy()
    return float(np.mean([pinball(y, df[c].to_numpy(), t).mean() for c, t in zip(QCOLS, QUANTILES)]))


def metrics(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["actual", "p50"])
    if d.empty:
        return {}
    e = d["p50"] - d["actual"]
    out = {"n": len(d), "MAPE": float((e.abs() / d["actual"]).mean() * 100),
           "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
           "bias": float(e.mean())}
    if d[["p10", "p90"]].notna().all().all() and (d["p90"] > d["p10"]).any():
        out["pinball"] = mean_pinball(d)
        out["coverage80"] = float(((d["actual"] >= d["p10"]) & (d["actual"] <= d["p90"])).mean() * 100)
    return out
