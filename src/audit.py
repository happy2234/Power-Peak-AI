"""Fairness audit of the comparison with Grid-India's week-ahead forecast -> reports/fairness_audit.md.

The headline comparison averaged our 1..7-day-ahead forecasts for each day. Grid-India prepares
its week-ahead forecast several days before the week starts, so for most days it is forecasting
much further ahead. This module re-does the comparison step by step:

  0. original basis   — our mean of h=1..7 vs Grid-India, same days
  1. like with like   — one forecast of ours per day (our P50 at a single horizon) vs Grid-India
  2. matched issue    — our forecast issued with the information Grid-India had: origin = the
                        day before Grid-India's 'prepared' date (Field2 in the listing; the PSP
                        report of that day is the latest one published), same lead per day.
                        Needs leads up to 13 days: a long-lead LightGBM refit monthly exactly like
                        the main backtest; days beyond the 7-day weather archive use climatology
                        (normals from data before the backtest), never observed weather.
  3. by source        — matched, split by how the Grid-India value was read (text labels vs the
                        digitised curve)
  4. bias-corrected   — Grid-India minus a constant bias per target estimated on PAST weeks only
                        (days whose actuals were published before that week's prepared date)
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, TARGETS

log = logging.getLogger(__name__)
LONG_H = 14
TGT = list(TARGETS)


# ============================================================================ issue dates
def issue_dates() -> pd.DataFrame:
    """One row per Grid-India forecast file: source_file, forecast_start, prepared (Field2),
    uploaded (CreatedOn)."""
    import re

    from src.fetch.grid_india import safe_filename

    rows = []
    for f in sorted((RAW / "_index").glob("DEMAND_FORECAST_*.json")):
        for e in json.loads(f.read_text()):
            m = re.search(r"(\d{2}-\w{3}-\d{4}) to", e.get("Title_", ""))
            rows.append({"source_file": safe_filename(e["FilePath"]),
                         "title_start": pd.to_datetime(m.group(1), format="%d-%b-%Y") if m else pd.NaT,
                         "prepared": pd.to_datetime(e.get("Field2"), errors="coerce"),
                         "uploaded": pd.to_datetime(e.get("CreatedOn"), format="%d-%m-%Y %H:%M", errors="coerce")})
    return pd.DataFrame(rows).drop_duplicates("source_file")


def grid_india_with_issue(actuals: pd.DataFrame) -> pd.DataFrame:
    """Grid-India daily forecasts (one per date/target, latest issue) + issue information."""
    from src.models.baselines import grid_india_forecasts

    gi = grid_india_forecasts()
    fd = pd.read_parquet(PROCESSED / "forecast_daily.parquet")
    files = fd.groupby("forecast_start").agg(source_file=("source_file", "first"),
                                             label_kind=("label_kind", "first")).reset_index()
    files = files.merge(issue_dates(), on="source_file", how="left")
    gi = gi.merge(files, on="forecast_start", how="left").merge(actuals, on=["date", "target"], how="left")
    # an issue without a listing date: assume prepared the day before the week starts (generous to GI)
    gi["prepared"] = gi["prepared"].fillna(gi["forecast_start"] - pd.Timedelta(days=1))
    gi["prepared"] = gi[["prepared", "forecast_start"]].min(axis=1)
    gi["issue_origin"] = gi["prepared"] - pd.Timedelta(days=1)       # last day with published actuals
    gi["lead"] = (gi["date"] - gi["issue_origin"]).dt.days
    return gi


# ============================================================================ long-lead model
def climatology_fn(daily: pd.DataFrame):
    """dates -> WX_FEATS from day-of-year normals of OBSERVED weather up to NORMALS_END
    (anomalies = 0). Used for leads beyond the 7-day weather forecast archive."""
    from src.features.build import NORMALS_END, WX_FEATS, fit_normals, normal_at

    raw = [c for c in WX_FEATS if not c.endswith("_anom")]
    nrm = fit_normals(daily, raw, NORMALS_END)

    def f(dates):
        dates = pd.DatetimeIndex(dates)
        out = {c: (normal_at(nrm, c, dates) if not c.endswith("_anom") else np.zeros(len(dates)))
               for c in WX_FEATS}
        return pd.DataFrame(out, index=dates)
    return f


def long_lead_backtest(start: str, save: bool = True) -> pd.DataFrame:
    """LightGBM with horizons 1..LONG_H, refit monthly (same protocol as the main backtest)."""
    from src.features.build import load_daily, make_tabular, valid_origins
    from src.models.baselines import LGBMQuantile
    from src.models.tune import best_params

    daily, fc = load_daily()
    clim = climatology_fn(daily)
    end = daily.index.max() - pd.Timedelta(days=1)
    hz = range(1, LONG_H + 1)
    preds = []
    t0 = time.time()
    for ms in pd.date_range(pd.Timestamp(start), end, freq="MS"):
        me = min(ms + pd.offsets.MonthEnd(0), end)
        test_o = valid_origins(daily, start=ms, end=me)
        # labels of training samples must be known before the month: origin + LONG_H < ms
        train_o = valid_origins(daily, start="2023-06-01", end=ms - pd.Timedelta(days=LONG_H + 1))
        if len(test_o) == 0:
            continue
        tr = make_tabular(daily, fc, train_o, horizons=hz, climatology=clim)
        te = make_tabular(daily, fc, test_o, horizons=hz, climatology=clim)
        m = LGBMQuantile(best_params("lgbm")).fit(tr)
        p = m.predict_long(te)
        p["model"] = "lgbm_long"
        p = p.merge(te[["origin", "h", "wx_source"]], on=["origin", "h"], how="left")
        preds.append(p)
        log.info("long-lead month %s done (%.1f min)", ms.strftime("%Y-%m"), (time.time() - t0) / 60)
    out = pd.concat(preds, ignore_index=True)
    out = out[out["date"] <= daily.index.max()]
    if save:
        out.to_parquet(PROCESSED / "audit_long_lead.parquet", index=False)
    return out


# ============================================================================ comparisons
def _gap(ae_ours, ae_gi) -> dict:
    ae_ours, ae_gi = np.asarray(ae_ours, float), np.asarray(ae_gi, float)
    return {"days": len(ae_ours), "our_MAE": ae_ours.mean(), "gi_MAE": ae_gi.mean(),
            "improvement": 1 - ae_ours.mean() / ae_gi.mean(), "win_rate": float((ae_ours < ae_gi).mean())}


def _mape(e, y):
    return float(np.mean(np.abs(e) / y) * 100)


def compare(j: pd.DataFrame, ours_col: str = "p50_ours", gi_col: str = "p50_gi") -> pd.DataFrame:
    rows = []
    for t, g in j.dropna(subset=[ours_col, gi_col, "actual"]).groupby("target"):
        r = _gap((g[ours_col] - g["actual"]).abs(), (g[gi_col] - g["actual"]).abs())
        r.update(target=t, our_MAPE=_mape(g[ours_col] - g["actual"], g["actual"]),
                 gi_MAPE=_mape(g[gi_col] - g["actual"], g["actual"]),
                 our_bias=float((g[ours_col] - g["actual"]).mean()),
                 gi_bias=float((g[gi_col] - g["actual"]).mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def past_bias(gi: pd.DataFrame) -> pd.Series:
    """For each Grid-India row, the mean error of Grid-India over all earlier days of the same
    target whose actuals were published before this row's prepared date (expanding, past only)."""
    out = pd.Series(np.nan, index=gi.index)
    g = gi.dropna(subset=["actual"])
    for t, rows in gi.groupby("target"):
        hist = g[g["target"] == t][["date", "p50", "actual"]].sort_values("date")
        err = (hist["p50"] - hist["actual"]).to_numpy()
        dates = hist["date"].to_numpy()
        for i, r in rows.iterrows():
            known = dates <= np.datetime64(r["issue_origin"])        # actual published by prepared date
            if known.sum() >= 14:
                out[i] = err[known].mean()
    return out


def run_audit(final_model: str, processed: pd.DataFrame, daily: pd.DataFrame,
              long_lead: pd.DataFrame, eval_start: str):
    from src.evaluate import actuals_long

    gi = grid_india_with_issue(actuals_long(daily))
    gi = gi[(gi["date"] >= pd.Timestamp(eval_start)) & (gi["date"] <= daily.index.max())].dropna(subset=["actual"])
    ours = processed[(processed["model"] == final_model) & (processed["origin"] >= pd.Timestamp(eval_start))]
    res, steps = {}, []

    # 0. original basis: mean of our h=1..7 forecasts per day
    m0 = ours.groupby(["date", "target"], as_index=False)["p50"].mean().rename(columns={"p50": "p50_ours"})
    j0 = gi.rename(columns={"p50": "p50_gi"}).merge(m0, on=["date", "target"])
    res["0 original"] = compare(j0)
    steps.append(("0. Original basis: mean of our 1–7-day-ahead forecasts for each day", "0 original"))

    # 1. one forecast per day: our h=7 (the furthest ahead we forecast)
    o7 = ours[ours["h"] == 7][["date", "target", "p50"]].rename(columns={"p50": "p50_ours"})
    res["1 single h=7"] = compare(gi.rename(columns={"p50": "p50_gi"}).merge(o7, on=["date", "target"]))
    steps.append(("1. Like with like: one P50 per day (our 7-day-ahead forecast) vs Grid-India's value", "1 single h=7"))

    # 2. matched issue time with the long-lead model (same origin, same lead per day)
    ll = long_lead.rename(columns={"p50": "p50_ours", "origin": "issue_origin"})[
        ["issue_origin", "date", "target", "p50_ours", "h", "wx_source"]]
    jm = gi.rename(columns={"p50": "p50_gi"}).merge(ll, on=["issue_origin", "date", "target"], how="left")
    unmatched = int(jm["p50_ours"].isna().sum())
    jm = jm.dropna(subset=["p50_ours"])
    res["2 matched issue (lgbm_long)"] = compare(jm)
    steps.append(("2. Matched issue time: our forecast made with the data Grid-India had (same issue day, "
                  "same lead per day; long-lead LightGBM)", "2 matched issue (lgbm_long)"))

    # 2b. matched issue time with the final model where its 7-day horizon reaches
    fm = ours.rename(columns={"p50": "p50_ours", "origin": "issue_origin"})[["issue_origin", "date", "target", "p50_ours"]]
    jf = gi.rename(columns={"p50": "p50_gi"}).merge(fm, on=["issue_origin", "date", "target"])
    res[f"2b matched issue ({final_model}, leads <= 7 only)"] = compare(jf)
    steps.append((f"2b. Matched issue time, final model ({final_model}) — only the days within its 7-day "
                  "horizon (the first days of each Grid-India week)", f"2b matched issue ({final_model}, leads <= 7 only)"))

    # 3. by source of the Grid-India value
    for kind, lab in (("label", "text labels"), ("curve", "digitised curve")):
        sub = jm[jm["source"] == kind]
        if len(sub):
            res[f"3 matched, GI from {lab}"] = compare(sub)
            steps.append((f"3. Matched, Grid-India value read from {lab}", f"3 matched, GI from {lab}"))

    # 4. bias-corrected Grid-India (constant bias from past weeks only)
    gi_b = gi.copy()
    gi_b["bias_past"] = past_bias(gi_b)
    jb = gi_b.rename(columns={"p50": "p50_gi"}).merge(ll, on=["issue_origin", "date", "target"])
    jb["p50_gi_corr"] = jb["p50_gi"] - jb["bias_past"]
    jb = jb.dropna(subset=["p50_gi_corr", "p50_ours"])
    res["4 matched + GI bias-corrected"] = compare(jb, gi_col="p50_gi_corr")
    steps.append(("4. Matched + Grid-India's constant bias removed (bias estimated on past weeks only)",
                  "4 matched + GI bias-corrected"))

    info = {"unmatched_gi_rows": unmatched, "gi_rows": len(gi),
            "lead_distribution": jm.groupby("lead").size().to_dict(),
            "prepared_lag": (gi.drop_duplicates("forecast_start")["forecast_start"]
                             - gi.drop_duplicates("forecast_start")["prepared"]).dt.days.value_counts().sort_index().to_dict(),
            "long_lead_wx_source": jm["wx_source"].value_counts().sort_index().to_dict(),
            "bias_rows": len(jb)}
    return steps, res, info, jm


def leakage_check(daily: pd.DataFrame, eval_start: str) -> dict:
    """Weather inputs of every backtest row: source codes and a direct comparison with observed."""
    from src.features.build import load_daily

    p = PROCESSED / "backtest_wx_source.parquet"
    src = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["origin", "h", "date", "wx_source"])
    src_e = src[src["origin"] >= pd.Timestamp(eval_start)]
    counts = src_e["wx_source"].value_counts().sort_index().to_dict()
    _, fc = load_daily()
    # issue date of every archived forecast row = date - lead; used rows: lead h for origin+h
    used = src_e[src_e["wx_source"] == 0][["origin", "h", "date"]].merge(
        fc[["date", "lead", "tmax_ai", "precip_ai"]], left_on=["date", "h"], right_on=["date", "lead"], how="left")
    issued = used["date"] - pd.to_timedelta(used["lead"], unit="D")
    obs = daily[["tmax_ai", "precip_ai"]].reindex(used["date"]).to_numpy()
    same = np.isclose(used[["tmax_ai", "precip_ai"]].to_numpy(), obs, atol=1e-9).all(1)
    return {"rows": len(src_e), "source_counts": {int(k): int(v) for k, v in counts.items()},
            "issued_on_origin": int((issued == used["origin"]).sum()), "checked": len(used),
            "issued_after_origin": int((issued > used["origin"]).sum()),
            "identical_to_observed": int(same.sum()),
            "tmax_mae_vs_observed": float(np.nanmean(np.abs(used["tmax_ai"].to_numpy() - obs[:, 0])))}
