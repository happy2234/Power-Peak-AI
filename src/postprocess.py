"""Post-processing learned from PAST backtest outputs only: ensemble weights and conformal
calibration of the P10–P90 interval per target and horizon.

Rolling protocol (no look-ahead). For the forecasts issued in month M, the weights / corrections
are fitted on backtest rows whose *target date* is before M starts, i.e. whose actuals were
published before any forecast in M was issued. The live forecast uses the same fit on all
backtest rows (`fit_live`), saved to data/processed/postprocess.json.

Ensemble: P(q) = w * LSTM(q) + (1 - w) * LightGBM(q) for q in P10/P50/P90 (quantile averaging),
w per target x horizon on a 0..1 grid (step 0.05), chosen to minimise MAPE of the P50 over the
last `ENS_WINDOW_DAYS` days of past rows; 0.5 until enough rows exist.

Conformal (split-conformal CQR, per model x target x horizon): score s = max(P10 - y, y - P90) / P50
over the last `CAL_WINDOW_DAYS` days of past rows; q = the ceil((1 - alpha)(n + 1)) / n empirical
quantile; P10 -= q * P50, P90 += q * P50. q < 0 narrows an interval that was too wide.

Adaptive level (adaptive conformal inference, Gibbs & Candès 2021, applied monthly): alpha starts
at 0.2 and after each month moves by ACI_GAMMA * (0.2 - miss rate of the last 31 days of
already-calibrated forecasts whose actuals are published). Errors shrink as the training set
grows and seasons change, so residuals from the past window alone over-cover; the feedback
keeps the realised coverage near 80 %. ACI_GAMMA = 0 is plain split-conformal.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED, TARGETS

ENS_MEMBERS = ("lstm", "lgbm")          # (w applies to the first)
ENS_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
ENS_WINDOW_DAYS = 180
ENS_MIN_ROWS = 40
CAL_WINDOW_DAYS = 90
CAL_MIN_ROWS = 30
COVERAGE = 0.8
ACI_GAMMA = 0.7
ALPHA_BOUNDS = (0.02, 0.5)


# ============================================================================ ensemble
def ensemble_weight(past: pd.DataFrame, a: str, b: str) -> float:
    """past: rows of members a and b for ONE target x horizon, with actuals."""
    j = past[past["model"] == a].merge(past[past["model"] == b], on=["origin", "date"], suffixes=("_a", "_b"))
    j = j.dropna(subset=["actual_a", "p50_a", "p50_b"])
    if len(j) < ENS_MIN_ROWS:
        return 0.5
    y = j["actual_a"].to_numpy()
    pa, pb = j["p50_a"].to_numpy(), j["p50_b"].to_numpy()
    mape = [np.mean(np.abs(w * pa + (1 - w) * pb - y) / y) for w in ENS_GRID]
    return float(ENS_GRID[int(np.argmin(mape))])


def fit_ensemble(past: pd.DataFrame, members=ENS_MEMBERS) -> dict:
    """-> {target: {h: w}} from past rows (any dates; caller restricts to the past)."""
    a, b = members
    lo = past["date"].max() - pd.Timedelta(days=ENS_WINDOW_DAYS) if len(past) else None
    p = past[past["model"].isin(members)]
    if lo is not None:
        p = p[p["date"] > lo]
    return {t: {h: ensemble_weight(p[(p["target"] == t) & (p["h"] == h)], a, b)
                for h in range(1, HORIZON + 1)} for t in TARGETS}


def apply_ensemble(rows: pd.DataFrame, weights: dict, members=ENS_MEMBERS, name: str = "ensemble") -> pd.DataFrame:
    a, b = members
    keys = ["origin", "date", "h", "target"]
    j = rows[rows["model"] == a].merge(rows[rows["model"] == b], on=keys, suffixes=("_a", "_b"))
    if j.empty:
        return pd.DataFrame(columns=rows.columns)
    w = np.array([weights[t][h] for t, h in zip(j["target"], j["h"])])
    out = j[keys].copy()
    out["model"] = name
    for q in ("p10", "p50", "p90"):
        out[q] = w * j[f"{q}_a"] + (1 - w) * j[f"{q}_b"]
    out["actual"] = j["actual_a"]
    out["ens_w"] = w
    if "refit_month_a" in j:
        out["refit_month"] = j["refit_month_a"]
    return out


# ============================================================================ conformal
def conformal_q(past: pd.DataFrame, alpha: float = 1 - COVERAGE) -> float:
    """past: rows for one model x target x horizon. Relative CQR quantile (0 if too few)."""
    d = past.dropna(subset=["actual", "p10", "p50", "p90"])
    if len(d) < CAL_MIN_ROWS:
        return 0.0
    s = np.maximum(d["p10"] - d["actual"], d["actual"] - d["p90"]) / d["p50"]
    n = len(s)
    level = min(1.0, max(0.0, np.ceil((1 - alpha) * (n + 1)) / n))
    return float(np.quantile(s, level, method="higher"))


def fit_conformal(past: pd.DataFrame, models, alphas: dict | None = None) -> dict:
    """-> {model: {target: {h: q}}} on the last CAL_WINDOW_DAYS of `past`.
    alphas: {(model, target, h): alpha} (default 0.2 everywhere)."""
    out = {}
    for m in models:
        p = past[past["model"] == m]
        if p.empty:
            continue
        p = p[p["date"] > p["date"].max() - pd.Timedelta(days=CAL_WINDOW_DAYS)]
        out[m] = {t: {h: conformal_q(p[(p["target"] == t) & (p["h"] == h)],
                                     (alphas or {}).get((m, t, h), 1 - COVERAGE))
                      for h in range(1, HORIZON + 1)} for t in TARGETS}
    return out


def update_alphas(alphas: dict, calibrated: pd.DataFrame, ms) -> dict:
    """ACI step: realised miss rate over the 31 days before `ms` of rows already calibrated."""
    if not ACI_GAMMA or calibrated.empty:
        return alphas
    rec = calibrated[(calibrated["date"] < ms) & (calibrated["date"] >= ms - pd.Timedelta(days=31))]
    rec = rec.dropna(subset=["actual", "p10", "p90"])
    out = dict(alphas)
    for (m, t, h), g in rec.groupby(["model", "target", "h"]):
        if len(g) < 10:
            continue
        miss = 1 - ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()
        a = out.get((m, t, h), 1 - COVERAGE) + ACI_GAMMA * ((1 - COVERAGE) - miss)
        out[(m, t, h)] = float(np.clip(a, *ALPHA_BOUNDS))
    return out


def apply_conformal(rows: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Adjust P10/P90 of the models present in `cal`; keeps the raw interval in p10_raw/p90_raw."""
    out = rows.copy()
    out["p10_raw"], out["p90_raw"] = out["p10"], out["p90"]
    q = np.array([cal.get(m, {}).get(t, {}).get(h, 0.0) for m, t, h in zip(out["model"], out["target"], out["h"])])
    q = np.nan_to_num(q)
    out["p10"] = out["p10"] - q * out["p50"]
    out["p90"] = out["p90"] + q * out["p50"]
    # a negative q may not cross the median
    out["p10"] = np.minimum(out["p10"], out["p50"])
    out["p90"] = np.maximum(out["p90"], out["p50"])
    out["conf_q"] = q
    return out


# ============================================================================ rolling application
def rolling_postprocess(bt: pd.DataFrame, members=ENS_MEMBERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bt: raw backtest (all models incl. warm-up months). Returns (rows of every quantile model
    plus "ensemble", with conformal P10/P90 and the raw interval kept in p10_raw/p90_raw; log of
    the parameters fitted for each month).

    Pass 1 builds the ensemble month by month with the weights available in that month, so the
    ensemble's own past rows (used to calibrate its interval) never use later weights."""
    ours = bt[bt["model"] != "grid_india"].copy()
    ours["month"] = ours["origin"].dt.to_period("M").dt.to_timestamp()
    have_ens = set(members) <= set(ours["model"])
    months = sorted(ours["month"].unique())
    weights, ens_rows = {}, []
    for ms in months:                         # pass 1: ensemble weights + ensemble rows per month
        past = ours[ours["date"] < ms].drop(columns="month")
        if have_ens:
            weights[ms] = fit_ensemble(past, members)
            ens_rows.append(apply_ensemble(ours[ours["month"] == ms].drop(columns="month"), weights[ms], members))
    allrows = ours.drop(columns="month")
    if ens_rows:
        allrows = pd.concat([allrows] + ens_rows, ignore_index=True)
    allrows["month"] = allrows["origin"].dt.to_period("M").dt.to_timestamp()
    qmodels = sorted(allrows["model"].unique())
    out, log, alphas = [], [], {}
    for ms in months:                         # pass 2: conformal on past rows of every model
        past = allrows[allrows["date"] < ms]
        done = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        alphas = update_alphas(alphas, done, ms)
        cal = fit_conformal(past, qmodels, alphas)
        out.append(apply_conformal(allrows[allrows["month"] == ms].drop(columns="month"), cal))
        log.append({"month": str(pd.Timestamp(ms).date()), "past_days": int(past["date"].nunique()),
                    "ensemble": weights.get(ms), "conformal": cal,
                    "alpha": {f"{m}|{t}|{h}": a for (m, t, h), a in alphas.items()}})
    res = pd.concat(out, ignore_index=True)
    # the state the live forecast continues from: one more ACI step with the latest outcomes
    nxt = pd.Timestamp(months[-1]) + pd.offsets.MonthBegin(1) if months else None
    res.attrs["alphas"] = update_alphas(alphas, res, max(nxt, res["date"].max() + pd.Timedelta(days=1))) if nxt else alphas
    return res, pd.DataFrame(log)


def fit_live(bt: pd.DataFrame, processed: pd.DataFrame, members=ENS_MEMBERS, extra: dict | None = None,
             alphas: dict | None = None) -> dict:
    """Parameters for the live forecast from ALL backtest rows (all actuals are in the past)."""
    ours = bt[bt["model"] != "grid_india"]
    w = fit_ensemble(ours, members) if set(members) <= set(ours["model"]) else None
    raw = processed.copy()
    raw["p10"], raw["p90"] = raw["p10_raw"], raw["p90_raw"]
    alphas = alphas if alphas is not None else processed.attrs.get("alphas", {})
    cal = fit_conformal(raw, sorted(raw["model"].unique()), alphas)
    out = {"ensemble_members": list(members), "ensemble": w, "conformal": cal,
           "aci_alpha": {f"{m}|{t}|{h}": a for (m, t, h), a in alphas.items()},
           "calibration": {"window_days": CAL_WINDOW_DAYS, "aci_gamma": ACI_GAMMA},
           "fitted_on_dates": [str(ours["date"].min().date()), str(ours["date"].max().date())],
           **(extra or {})}
    (PROCESSED / "postprocess.json").write_text(json.dumps(_jsonable(out), indent=1))
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def load_live() -> dict | None:
    p = PROCESSED / "postprocess.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())

    def ints(m):
        return {t: {int(h): v for h, v in hs.items()} for t, hs in m.items()}
    if d.get("ensemble"):
        d["ensemble"] = ints(d["ensemble"])
    d["conformal"] = {m: ints(v) for m, v in d.get("conformal", {}).items()}
    return d
