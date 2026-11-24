#!/usr/bin/env python
"""Read-only export of everything needed to explain the AI part of the project (class presentation).

    python scripts/export_ai_data.py

Writes ONLY into ai_data/ (never data/ or models/). Never retrains or overwrites a production model;
the ablation and the LightGBM learning curve use temporary in-memory fits. Network access is blocked
at the socket level. Reuses src/ (metrics, evaluate, models, features). Seed 42.
"""
from __future__ import annotations

import inspect
import json
import socket
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _no_net(*a, **k):
    raise RuntimeError("network access is disabled in export_ai_data.py")


socket.socket.connect = _no_net            # type: ignore[assignment]
socket.create_connection = _no_net         # type: ignore[assignment]
socket.getaddrinfo = _no_net               # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

from src import evaluate as ev  # noqa: E402
from src import fixed_split as fx  # noqa: E402
from src.config import ENCODER_DAYS, HORIZON, MODELS_DIR, PROCESSED, TARGET_UNITS, TARGETS  # noqa: E402
from src.features import build as fb  # noqa: E402
from src.features.calendar import calendar_features  # noqa: E402
from src.models.common import metrics  # noqa: E402
from src.models.registry import latest_dir  # noqa: E402
from src.utils.device import set_seed  # noqa: E402
from src.walkthrough import _WEATHER_RX  # noqa: E402

SEED = 42
OUT = ROOT / "ai_data"
TGT = list(TARGETS)
EXAMPLE_DAY = pd.Timestamp("2026-11-14")
ALT = [ROOT.parent / "Power-Peak-AI-export", ROOT.parent / "india-power-forecast"]   # read-only fallbacks

HEAD = ["seasonal_naive", "lgbm@compact", "lstm_w@compact", "tft@v1", "ensemble", "grid_india"]
LABEL = {"seasonal_naive": "Seasonal naive", "lgbm@compact": "LightGBM", "lstm_w@compact": "LSTM",
         "tft@v1": "TFT", "ensemble": "Ensemble (LSTM+LightGBM)", "grid_india": "Grid-India"}
SHORT = {k: v.split(" (")[0] for k, v in LABEL.items()}
QMODELS = [m for m in HEAD if m != "grid_india"]

MISSING: list[str] = []            # lines for MISSING.md
SOURCES: dict[str, str] = {}       # file -> source description (for README.md)
README_COLS: dict[str, str] = {}


def note(msg: str) -> None:
    print("  NOTE:", msg)
    MISSING.append(msg)


def save(df: pd.DataFrame, name: str, source: str, index: bool = False) -> pd.DataFrame:
    df.to_csv(OUT / name, index=index, float_format="%.6g")
    SOURCES[name] = source
    README_COLS[name] = ", ".join(map(str, df.columns))
    print(f"  wrote {name}: {len(df)} rows")
    return df


def find_file(rel: str) -> tuple[Path | None, str]:
    """Local project first, then the two sibling projects (read-only)."""
    for base in [ROOT] + ALT:
        p = base / rel
        if p.exists():
            return p, ("this project" if base == ROOT else str(base))
    return None, "not found"


# ============================================================================ feature groups
def ai_group(n: str) -> str:
    """Weather / Calendar / Lag / Rolling / Other. Reproduces the 35/26/12/9/13 split of the 95 distinct
    features (src.walkthrough.feature_group gives 31/25/9/3/27; see MISSING.md)."""
    if "lag" in n or n.endswith("_last"):
        return "Lag"
    if "roll" in n or n.endswith(("_std7", "_std30")):
        return "Rolling"
    if _WEATHER_RX.match(n) or (n.startswith("d_") and _WEATHER_RX.match(n[2:])) or n == "wx_source":
        return "Weather"
    if (n in ("dow", "month", "is_weekend", "days_to_festival", "days_from_festival", "is_bridge", "diwali_window",
              "is_festival", "is_holiday", "is_national_holiday", "is_saturday", "is_sunday", "ref_holidays")
            or n.startswith(("dow_", "doy_", "fest_"))):
        return "Calendar"
    return "Other"


def unit_of(t: str) -> str:
    return TARGET_UNITS[t]


def ape(d: pd.DataFrame) -> pd.Series:
    return (d["p50"] - d["actual"]).abs() / d["actual"] * 100


def mrow(g: pd.DataFrame) -> dict:
    m = metrics(g)
    d = g.dropna(subset=["actual", "p50"])
    return {"mape_pct": m.get("MAPE"), "mae": m.get("MAE"), "rmse": m.get("RMSE"), "bias": m.get("bias"),
            "bias_pct": float(((d["p50"] - d["actual"]) / d["actual"]).mean() * 100) if len(d) else np.nan,
            "pinball_mean": m.get("pinball", np.nan), "coverage_p10_p90_pct": m.get("coverage80", np.nan),
            "n_days": int(d["date"].nunique()), "n_forecasts": int(m.get("n", 0))}


# ============================================================================ 1. model cards
def model_cards(cfgs: dict, daily) -> pd.DataFrame:
    import torch

    from src.models.deep import DeepForecaster

    log = {}
    for line in (PROCESSED / "training_log.jsonl").read_text().splitlines():
        d = json.loads(line)
        log[d["saved_to"].split("/")[-1]] = d
    sig = inspect.signature(DeepForecaster.__init__).parameters
    rows = []
    # ---- LightGBM
    d = latest_dir("lgbm")
    import pickle
    with open(d / "model.pkl", "rb") as fh:
        lg = pickle.load(fh)
    c = cfgs["lgbm"]
    p = c["params"]
    trees = {f"{k}|{q}": int(m.booster_.num_trees()) for (k, q), m in lg.models.items()}
    rows.append({
        "model": "LightGBM", "key": "lgbm@compact", "type": "Gradient-boosted trees (quantile regression)",
        "architecture": f"9 LightGBM quantile models (3 targets x P10/P50/P90), horizons 1-7 pooled with 'h' as a feature, "
                        f"num_leaves={p['num_leaves']}, predicts ratio to the 7-day reference level, CQR-widened",
        "parameters_or_trees": sum(trees.values()), "parameter_unit": "trees (sum of 9 models)",
        "input_window": "7-day reference level; lags 7/14/364 d, 30-d rolling mean, 7/30-d std; weather for target day",
        "encoder_features": np.nan, "decoder_features": np.nan, "distinct_features": len(c["features"]),
        "feature_set": c["feature_set"], "hidden": np.nan, "layers": np.nan, "d_model": np.nan, "heads": np.nan,
        "dropout": np.nan, "learning_rate": p["learning_rate"], "batch_size": np.nan, "weight_decay": np.nan,
        "optimiser": "gradient boosting (quantile / pinball objective)", "scheduler": "none",
        "early_stopping_patience": 100, "num_leaves": p["num_leaves"], "n_estimators_max": p["n_estimators"],
        "min_child_samples": p["min_child_samples"], "colsample_bytree": p["colsample_bytree"],
        "subsample": p["subsample"], "reg_lambda": p["reg_lambda"], "max_bin": p["max_bin"],
        "epochs_run": np.nan, "best_epoch": np.nan, "best_iterations_P50": json.dumps(
            {k: v for k, v in c["best_iters"].items() if k.endswith("|0.5")}),
        "best_validation_loss": np.nan, "best_validation_loss_unit": "n/a (early-stopped per tree model)",
        "training_seconds": log.get(d.name, {}).get("train_seconds"), "device": "CPU",
        "trained_through": c["train_end"], "model_dir": str(d.relative_to(ROOT)),
        "extra": json.dumps({"best_iters": c["best_iters"], "trees_per_model": trees, "cqr": c["cqr"]})})
    cards = {"LightGBM": c["features"]}
    # ---- deep models
    for kind, name, key in (("lstm", "LSTM", "lstm_w@compact"), ("tft", "TFT", "tft@v1")):
        d = latest_dir(kind)
        c = cfgs[kind]
        p = c["params"]
        hist = pd.read_csv(d / "history.csv")
        net = DeepForecaster.load(d).model
        n_par = sum(x.numel() for x in net.parameters())
        best_ep = int(hist["val_loss"].idxmin()) + 1
        arch = (f"Encoder-decoder LSTM: Linear({len(c['enc_features'])}->{p['hidden']}) + {p['layers']}-layer LSTM encoder over "
                f"{c['enc_days']} days, LSTM decoder over {c['horizon']} future days, quantile head (P10/P50/P90 x 3 targets)"
                if kind == "lstm" else
                f"Compact Temporal Fusion Transformer: variable-selection networks (enc/dec), LSTM encoder-decoder, "
                f"{p['heads']}-head attention, gated residual networks, d_model={p['d_model']}, quantile head")
        tl = log.get(d.name, {})
        rows.append({
            "model": name, "key": key, "type": "Deep neural network (probabilistic seq2seq)", "architecture": arch,
            "parameters_or_trees": n_par, "parameter_unit": "trainable parameters",
            "input_window": f"{c['enc_days']}-day encoder history + {c['horizon']}-day decoder (known-future inputs)",
            "encoder_features": len(c["enc_features"]), "decoder_features": len(c["dec_features"]),
            "distinct_features": len(set(c["enc_features"]) | set(c["dec_features"])), "feature_set": c["feature_set"],
            "hidden": p.get("hidden", np.nan), "layers": p.get("layers", np.nan), "d_model": p.get("d_model", np.nan),
            "heads": p.get("heads", np.nan), "dropout": p["dropout"], "learning_rate": p["lr"],
            "batch_size": p.get("batch_size_used", p["batch_size"]), "weight_decay": p["weight_decay"],
            "optimiser": "AdamW (grad-clip 1.0, fp16 autocast on CUDA)",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=6)",
            "early_stopping_patience": sig["patience"].default, "num_leaves": np.nan, "n_estimators_max": np.nan,
            "min_child_samples": np.nan, "colsample_bytree": np.nan, "subsample": np.nan, "reg_lambda": np.nan,
            "max_bin": np.nan, "epochs_run": len(hist), "best_epoch": best_ep, "best_iterations_P50": "",
            "best_validation_loss": c["best_val_pinball_std"], "best_validation_loss_unit": "mean pinball, standardised ratio",
            "training_seconds": float(hist["seconds"].sum()), "device": tl.get("device", c.get("device")),
            "trained_through": c["train_end"], "model_dir": str(d.relative_to(ROOT)),
            "extra": json.dumps({"loss": "pinball (P10/P50/P90), rare-weather weight boost=%s" % c["regime_boost"],
                                 "max_epochs": sig["max_epochs"].default, "seed": c.get("seed"), "cqr": c["cqr"],
                                 "training_log_seconds_incl_refit": tl.get("train_seconds"),
                                 "enc_features": c["enc_features"], "dec_features": c["dec_features"]})})
        cards[name] = list(dict.fromkeys(c["enc_features"] + c["dec_features"]))
    # ---- ensemble and baselines
    post = json.loads((PROCESSED / "postprocess.json").read_text())
    w = pd.DataFrame(post["ensemble"]).astype(float)
    base = {k: np.nan for k in rows[0]}
    rows.append({**base, "model": "Ensemble (LSTM+LightGBM)", "key": "ensemble", "type": "Weighted quantile average",
                 "architecture": "P(q) = w*LSTM(q) + (1-w)*LightGBM(q), w per target x day-ahead (0..1 grid, 0.05 step), "
                                 "then adaptive conformal calibration of P10-P90", "parameters_or_trees": 0,
                 "parameter_unit": "no trained parameters (63 weights/conformal terms)",
                 "input_window": "outputs of LSTM and LightGBM", "distinct_features": np.nan,
                 "optimiser": "grid search of w on last 180 days of past backtest rows (min MAPE)",
                 "model_dir": "data/processed/postprocess.json",
                 "extra": json.dumps({"w_lstm_range": [float(w.values.min()), float(w.values.max())],
                                      "conformal_window_days": post["calibration"]["window_days"],
                                      "aci_gamma": post["calibration"]["aci_gamma"]})})
    rows.append({**base, "model": "Seasonal naive", "key": "seasonal_naive", "type": "Baseline",
                 "architecture": "P50 = same weekday one week earlier; P10/P90 = P50 x empirical quantiles of y(d)/y(d-7)",
                 "parameters_or_trees": 0, "parameter_unit": "none (2 quantile ratios per target)",
                 "input_window": "1 value (7 days earlier)", "distinct_features": 3, "feature_set": "target lag7 only",
                 "model_dir": "src/models/baselines.py::SeasonalNaive"})
    rows.append({**base, "model": "Grid-India", "key": "grid_india", "type": "External benchmark (official forecast)",
                 "architecture": "Grid-India NLDC week-ahead forecast (point only): range midpoint for peaks, integral of "
                                 "15-min curve for energy", "parameters_or_trees": np.nan,
                 "parameter_unit": "n/a (not our model)", "input_window": "prepared ~5 days before its week starts",
                 "model_dir": "data/raw/forecast (parsed by src/parse/forecast_parser.py)"})
    df = pd.DataFrame(rows)
    save(df.drop(columns=[]), "model_cards.csv", "models/*/<latest>/config.yaml, history.csv, model.pt/model.pkl; "
         "data/processed/training_log.jsonl, postprocess.json; src/models/*")
    full = df.to_dict("records")
    for r in full:
        r["feature_list"] = cards.get(r["model"], [])
        r["extra"] = json.loads(r["extra"]) if isinstance(r["extra"], str) else None
    (OUT / "model_cards.json").write_text(json.dumps(full, indent=1, default=lambda o: None if o != o else str(o)))
    SOURCES["model_cards.json"] = "same as model_cards.csv + full feature lists"
    README_COLS["model_cards.json"] = "one object per model; same fields as the CSV plus feature_list and extra"
    return df


# ============================================================================ 2-3. metrics
def model_metrics(bt_win, bt_all, fixed) -> pd.DataFrame:
    rows = []
    for window, d in (("rolling_12m_backtest", bt_win), ("fixed_test_split", fixed)):
        for m in HEAD:
            sub = d[d["model"] == m]
            if sub.empty:
                continue
            per = {}
            for t in TGT:
                g = sub[sub["target"] == t]
                if g.empty:
                    continue
                per[t] = mrow(g)
                rows.append({"window": window, "model": LABEL[m], "model_key": m, "target": t, "unit": unit_of(t), **per[t]})
            if len(per) == 3:
                rows.append({"window": window, "model": LABEL[m], "model_key": m, "target": "overall_mean_of_3_targets",
                             "unit": "%", "mape_pct": np.mean([v["mape_pct"] for v in per.values()]),
                             "bias_pct": np.mean([v["bias_pct"] for v in per.values()]),
                             "coverage_p10_p90_pct": np.nanmean([v["coverage_p10_p90_pct"] for v in per.values()])
                             if m != "grid_india" else np.nan,
                             "n_days": min(v["n_days"] for v in per.values())})
    df = pd.DataFrame(rows)
    return save(df, "model_metrics.csv",
                "data/processed/backtest_processed.parquet (rolling; origins >= 2025-11-01, Grid-India days >= 2025-11-01) and "
                "fixed_split_predictions.parquet (fixed test, from india-power-forecast); Grid-India fixed rows = its "
                "backtest_processed rows with date >= 2025-04-01; metrics via src.models.common.metrics")


def by_horizon(bt_win) -> pd.DataFrame:
    rows = []
    for (m, t, h), g in bt_win[bt_win["model"].isin(QMODELS)].groupby(["model", "target", "h"]):
        r = mrow(g)
        rows.append({"model": LABEL[m], "model_key": m, "target": t, "unit": unit_of(t), "days_ahead": int(h),
                     "mape_pct": r["mape_pct"], "mae": r["mae"], "bias": r["bias"], "bias_pct": r["bias_pct"],
                     "coverage_p10_p90_pct": r["coverage_p10_p90_pct"], "n_days": r["n_days"]})
    return save(pd.DataFrame(rows), "metrics_by_horizon.csv",
                "data/processed/backtest_processed.parquet (12-month window; Grid-India omitted: its 'h' counts from its "
                "forecast-week start, not from the issue date)")


# ============================================================================ 4-6. predictions and errors
def predictions_long(bt_all) -> None:
    d = bt_all[bt_all["model"].isin(HEAD)].copy()
    s = pd.Timestamp(ev.BACKTEST_START)
    d["in_12m_window"] = np.where(d["model"] == "grid_india", d["date"] >= s, d["origin"] >= s)
    out = pd.DataFrame({"date": d["date"].dt.date, "target": d["target"], "unit": d["target"].map(TARGET_UNITS),
                        "model": d["model"].map(LABEL), "model_key": d["model"], "days_ahead": d["h"],
                        "origin": d["origin"].dt.date, "actual": d["actual"], "p10": d["p10"], "p50": d["p50"],
                        "p90": d["p90"], "in_12m_window": d["in_12m_window"]})
    out = out.sort_values(["model_key", "target", "date", "days_ahead"])
    save(out, "predictions_long.csv", "data/processed/backtest_processed.parquet (calibrated P10/P90; includes the "
         "2025-05..2025-10 warm-up months, flagged by in_12m_window=False). Only the 6 headline models; variants omitted")


def error_tables(bt_win, daily) -> None:
    b = bt_win[bt_win["model"].isin(HEAD)].dropna(subset=["actual", "p50"]).copy()
    b["ape"] = ape(b)
    # ---- error distribution
    rows = []
    for (m, t), g in b.groupby(["model", "target"]):
        q = g["ape"].quantile([.25, .5, .75, .9, .95])
        rows.append({"model": LABEL[m], "model_key": m, "target": t, "n_forecasts": len(g), "abs_pct_error_p25": q[.25],
                     "abs_pct_error_p50": q[.5], "abs_pct_error_p75": q[.75], "abs_pct_error_p90": q[.9],
                     "abs_pct_error_p95": q[.95], "abs_pct_error_max": g["ape"].max(), "mape_pct": g["ape"].mean()})
    save(pd.DataFrame(rows), "error_distribution.csv", "data/processed/backtest_processed.parquet (12-month window; one "
         "row per forecast = origin x day ahead)")

    labels = ev.weather_regimes(daily)
    # ---- worst days (per target day: mean over the day-ahead forecasts, as src.evaluate.worst_days)
    rows = []
    for (m, t), g in b.groupby(["model", "target"]):
        day = g.groupby("date").agg(actual=("actual", "first"), forecast=("p50", "mean"), pct_error=("ape", "mean"),
                                    n_forecasts=("p50", "size"))
        day["signed_pct_error"] = (day["forecast"] - day["actual"]) / day["actual"] * 100
        w = day.sort_values("pct_error", ascending=False).head(10).reset_index()
        tg = ev.tag_days(w["date"], daily, labels)
        w["weather_condition"] = tg["weather"].to_numpy()
        w["weekday"] = tg["day"].to_numpy()
        flag = [", ".join(x for x in (h, f) if x) or "none" for h, f in zip(tg["holiday"], tg["festival"])]
        w["holiday_festival"] = flag
        w.insert(0, "rank", range(1, len(w) + 1))
        w.insert(0, "target", t)
        w.insert(0, "model_key", m)
        w.insert(0, "model", LABEL[m])
        w["unit"] = unit_of(t)
        rows.append(w)
    save(pd.concat(rows, ignore_index=True), "worst_days.csv",
         "backtest_processed.parquet + features.parquet (weather regime via src.evaluate.weather_regimes) + "
         "src.features.calendar; forecast = mean P50 over the days-ahead available for that date")

    # ---- errors by group
    dates = pd.DatetimeIndex(sorted(b["date"].unique()))
    cal = calendar_features(dates)
    flag = pd.Series(np.where(cal["is_national_holiday"] == 1, "national holiday",
                              np.where(cal["is_festival"] == 1, "festival",
                                       np.where(cal["is_holiday"] == 1, "other holiday", "regular day"))), index=dates)
    tmax = daily["tmax_ai"].reindex(dates)
    qs = tmax.quantile([0, .25, .5, .75, 1]).to_numpy()
    bucket = pd.Series(pd.cut(tmax, qs, include_lowest=True,
                              labels=[f"Q{i + 1} ({qs[i]:.1f}-{qs[i + 1]:.1f} C)" for i in range(4)]).astype(str), index=dates)
    groups = {
        "month": pd.Series([f"{d.month:02d} {d:%b}" for d in dates], index=dates),
        "weekday": pd.Series([f"{d.dayofweek + 1} {d:%a}" for d in dates], index=dates),
        "season": pd.Series(dates.month.map(ev.SEASONS), index=dates),
        "holiday_festival_flag": flag, "weather_condition": labels.reindex(dates),
        "tmax_ai_quartile": bucket}
    rows = []
    for gname, lab in groups.items():
        b["_g"] = lab.reindex(b["date"]).to_numpy()
        for (m, t, v), g in b.groupby(["model", "target", "_g"]):
            rows.append({"model": LABEL[m], "model_key": m, "target": t, "group_type": gname, "group": v,
                         "mape_pct": g["ape"].mean(), "bias_pct": ((g["p50"] - g["actual"]) / g["actual"]).mean() * 100,
                         "n_days": g["date"].nunique(), "n_forecasts": len(g)})
    save(pd.DataFrame(rows), "errors_by_group.csv", "backtest_processed.parquet (12-month window) + features.parquet "
         "(tmax_ai quartiles over the window's days) + src.evaluate (SEASONS, weather_regimes) + src.features.calendar")


# ============================================================================ 7. ablation + LightGBM curve
def ablation(cfgs) -> pd.DataFrame:
    import lightgbm as lgb

    from src.models.baselines import LGBMQuantile

    t0 = time.time()
    daily, fc = fx.features()
    fit_o, val_o, test_o = fx.split_origins(daily)
    # the data has grown since the stored fixed split was run: stop at its test end so numbers are comparable
    info = json.loads((PROCESSED / "fixed_split_info.json").read_text())
    test_o = test_o[test_o <= pd.Timestamp(info["test"][1])]
    tab = fb.make_tabular(daily, fc, fit_o.append(val_o))
    tab_val = tab[tab["origin"].isin(val_o)]
    tab_test = fb.make_tabular(daily, fc, test_o)
    params = fx.best_params("lgbm")
    cols = fb.tabular_feature_cols(tab, "compact")
    assert set(cols) == set(cfgs["lgbm"]["features"]), "compact feature list differs from the stored model"
    grp = {c: ai_group(c) for c in cols}
    refw = [c for c in cols if c.startswith("ref_") and _WEATHER_RX.match(c[4:])]
    variants = {
        "all_features": [], "no_weather": [c for c in cols if grp[c] == "Weather"],
        "no_calendar": [c for c in cols if grp[c] == "Calendar"],
        "no_lag_and_rolling": [c for c in cols if grp[c] in ("Lag", "Rolling")],
        "weather_only": [c for c in cols if grp[c] != "Weather"],
        "calendar_only": [c for c in cols if grp[c] != "Calendar"],
        "no_weather_incl_ref_levels (extra)": [c for c in cols if grp[c] == "Weather"] + refw}
    rows, curves = [], []
    for name, drop in variants.items():
        set_seed(SEED)
        tv = tab.drop(columns=drop)
        m = LGBMQuantile(params, seed=SEED, regime_boost=0.0, feature_set="compact").fit(
            tv, valid_start=val_o[0], refit=False)
        for part, tb in (("validation", tab_val), ("test", tab_test)):
            p = m.predict_long(tb)
            p = p[p["date"] >= pd.Timestamp(fx.TEST_START)] if part == "test" else p
            for t in TGT:
                g = p[p["target"] == t].dropna(subset=["actual"])
                rows.append({"variant": name, "split": part, "target": t, "n_features": len(m.features),
                             "mape_pct": metrics(g)["MAPE"], "n_forecasts": len(g)})
        if name == "all_features":
            for k in TGT:      # learning curve (validation / training pinball per boosting iteration)
                d = tv.dropna(subset=[f"y_{k}"])
                cut = val_o[0]
                tr, va = d[d["origin"] < cut - pd.Timedelta(days=7)], d[d["origin"] >= cut]
                for q in (0.1, 0.5, 0.9):
                    res = {}
                    r = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=SEED, **m.params)
                    r.fit(tr[m.features], tr[f"y_{k}"], eval_set=[(tr[m.features], tr[f"y_{k}"]), (va[m.features], va[f"y_{k}"])],
                          eval_names=["train", "valid"], callbacks=[lgb.record_evaluation(res)])
                    best = m.best_iters[(k, q)]
                    n = len(res["valid"]["quantile"])
                    curves.append(pd.DataFrame({
                        "model": "LightGBM (fixed-split refit, in memory)", "target": k, "quantile": q,
                        "iteration": np.arange(1, n + 1), "train_loss": res["train"]["quantile"],
                        "val_loss": res["valid"]["quantile"], "is_best_iter": np.arange(1, n + 1) == best,
                        "loss_unit": "pinball loss of y/reference-level ratio (single quantile)",
                        "fit_window": f"train origins {tr['origin'].min().date()}..{tr['origin'].max().date()}, "
                                      f"val origins {va['origin'].min().date()}..{va['origin'].max().date()}"}))
        print(f"    ablation {name}: {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["variant", "target", "n_features"], columns="split", values="mape_pct").reset_index()
    piv = piv.rename(columns={"validation": "val_mape_pct", "test": "test_mape_pct"})
    base = piv[piv["variant"] == "all_features"].set_index("target")
    piv["val_change_vs_all_pp"] = piv["val_mape_pct"] - piv["target"].map(base["val_mape_pct"])
    piv["test_change_vs_all_pp"] = piv["test_mape_pct"] - piv["target"].map(base["test_mape_pct"])
    piv["test_change_vs_all_pct"] = piv["test_change_vs_all_pp"] / piv["target"].map(base["test_mape_pct"]) * 100
    order = list(variants)
    piv["_o"] = piv["variant"].map({v: i for i, v in enumerate(order)})
    piv["_t"] = piv["target"].map({t: i for i, t in enumerate(TGT)})
    piv = piv.sort_values(["_o", "_t"]).drop(columns=["_o", "_t"])
    piv.insert(0, "split_info", f"fit {fit_o.min().date()}..{fit_o.max().date()} | val {val_o.min().date()}.."
                                f"{val_o.max().date()} | test {test_o.min().date()}..{test_o.max().date()} (origins); seed {SEED}")
    save(piv, "ablation.csv", "fresh in-memory LightGBM fits (src.models.baselines.LGBMQuantile, compact feature set, "
         "hyper-parameters models/lgbm/best_params_fixed_split.yaml), fixed split from src.fixed_split; nothing saved to models/")
    cv = pd.concat(curves, ignore_index=True)
    return piv, cv


# ============================================================================ 8. importance
def importance(cfgs, daily, fc) -> dict:
    import pickle

    out = {}
    d = latest_dir("lgbm")
    with open(d / "model.pkl", "rb") as fh:
        lg = pickle.load(fh)
    rows = []
    for (k, q), m in lg.models.items():
        g = m.booster_.feature_importance("gain")
        tot = g.sum()
        for f, v in zip(m.booster_.feature_name(), g):
            rows.append({"target": k, "quantile": q, "feature": f, "group": ai_group(f), "gain": v,
                         "pct_of_total": v / tot * 100 if tot else 0.0})
    fi = pd.DataFrame(rows).sort_values(["target", "quantile", "gain"], ascending=[True, True, False])
    save(fi, "feature_importance.csv", "models/lgbm/<latest>/model.pkl (LightGBM gain, 9 models: 3 targets x P10/P50/P90)")
    g = fi.groupby(["target", "quantile", "group"], as_index=False).agg(gain=("gain", "sum"), pct_of_total=("pct_of_total", "sum"),
                                                                        n_features=("feature", "size"))
    allq = g.groupby(["target", "group"], as_index=False).agg(gain=("gain", "mean"), pct_of_total=("pct_of_total", "mean"),
                                                              n_features=("n_features", "first"))
    allq["quantile"] = "mean of P10/P50/P90"
    save(pd.concat([g.assign(quantile=g["quantile"].astype(str)), allq], ignore_index=True)[
        ["target", "quantile", "group", "n_features", "gain", "pct_of_total"]],
        "feature_importance_by_group.csv", "sum of feature_importance.csv per group (and mean over the 3 quantiles)")
    out["fi_group"] = allq

    # ---- LSTM permutation importance (production LSTM, its own validation slice = last 15 % of origins)
    import torch

    from src.models.deep import DeepForecaster
    t0 = time.time()
    m = DeepForecaster.load(latest_dir("lstm"))
    origins = fb.valid_origins(daily, start="2023-06-01")
    val_o = origins[-max(int(len(origins) * 0.15), 20):]
    a = m._arrays(daily, fc, val_o)
    e0, d0, _ = m._scale(a)
    ref = a["ref"]
    act = a["y"] * ref[:, None, :]                                  # [N, H, T] levels (NaN if unknown)

    def score(e, d):
        with torch.no_grad():
            p = m.model(torch.from_numpy(e), torch.from_numpy(d)).float().numpy()
        p[..., 0] -= m.cqr
        p[..., 2] += m.cqr
        p = np.sort(p, -1)
        lvl = (1 + p[..., 1] * m.y_sd[None, None, :]) * ref[:, None, :]
        return np.nanmean(np.abs(lvl - act) / act * 100, axis=(0, 1))   # MAPE per target

    base = score(e0, d0)
    rng = np.random.default_rng(SEED)
    names = list(dict.fromkeys(m.enc_feats + m.dec_feats))
    ei = {n: i for i, n in enumerate(m.enc_feats)}
    di = {n: i for i, n in enumerate(m.dec_feats)}

    def permuted(feats, reps):
        res = []
        for _ in range(reps):
            e, d = e0.copy(), d0.copy()
            perm = rng.permutation(len(e))
            for n in feats:
                if n in ei:
                    e[:, :, ei[n]] = e0[perm][:, :, ei[n]]
                if n in di:
                    d[:, :, di[n]] = d0[perm][:, :, di[n]]
            res.append(score(e, d))
        return np.mean(res, 0)

    rows = []
    for n in names:
        s = permuted([n], 5)
        for k, t in enumerate(TGT):
            rows.append({"level": "feature", "name": n, "group": ai_group(n), "target": t,
                         "baseline_mape_pct": base[k], "permuted_mape_pct": s[k], "increase_pp": s[k] - base[k],
                         "in_encoder": n in ei, "in_decoder": n in di})
    for gname in ("Weather", "Calendar", "Lag", "Rolling", "Other"):
        fs = [n for n in names if ai_group(n) == gname]
        s = permuted(fs, 10)
        for k, t in enumerate(TGT):
            rows.append({"level": "group (all features of the group permuted together)", "name": gname, "group": gname,
                         "target": t, "baseline_mape_pct": base[k], "permuted_mape_pct": s[k], "increase_pp": s[k] - base[k],
                         "in_encoder": np.nan, "in_decoder": np.nan})
    pi = pd.DataFrame(rows)
    pi["n_validation_origins"] = len(val_o)
    pi["validation_origins"] = f"{val_o.min().date()}..{val_o.max().date()}"
    save(pi, "lstm_permutation_importance.csv", "models/lstm/<latest> (production LSTM) on its validation slice (last 15 % of "
         "origins); input columns shuffled across samples (seed 42, 5 repeats per feature, 10 per group); metric = MAPE of P50")
    # roll per-feature increases up to groups too
    gs = pi[pi["level"] == "feature"].groupby(["group", "target"], as_index=False)["increase_pp"].sum().rename(
        columns={"increase_pp": "sum_of_single_feature_increase_pp"})
    gj = pi[pi["level"] != "feature"][["group", "target", "increase_pp"]].rename(columns={"increase_pp": "joint_permutation_increase_pp"})
    save(gs.merge(gj, on=["group", "target"]), "lstm_permutation_importance_by_group.csv",
         "lstm_permutation_importance.csv (sum of single-feature increases vs permuting the whole group at once)")
    print(f"    LSTM permutation importance: {time.time() - t0:.0f}s")

    # ---- TFT variable selection and attention
    mt = DeepForecaster.load(latest_dir("tft"))
    a = mt._arrays(daily, fc, val_o)
    _, ar = mt.predict_quantiles(daily, fc, val_o, return_weights=True)
    w = ar["weights"]
    rows = []
    for side, key, feats in (("encoder (history)", "enc_vars", mt.enc_feats), ("decoder (known future)", "dec_vars", mt.dec_feats)):
        mw = w[key].mean(axis=(0, 1))
        for f, v in zip(feats, mw):
            rows.append({"side": side, "feature": f, "group": ai_group(f), "mean_selection_weight": v,
                         "pct_of_side_total": v / mw.sum() * 100})
    tv = pd.DataFrame(rows).sort_values(["side", "mean_selection_weight"], ascending=[True, False])
    tv["validation_origins"] = f"{val_o.min().date()}..{val_o.max().date()}"
    save(tv, "tft_variable_selection.csv", "models/tft/<latest> variable-selection softmax weights (mean over validation "
         "origins and time steps)")
    att = w.get("attention")
    if att is not None:
        am = att.mean(0)                                            # [H, T+H]
        T = mt.enc_days
        rows = []
        for h in range(am.shape[0]):
            for j in range(am.shape[1]):
                rows.append({"days_ahead": h + 1, "attended_position": f"encoder day t-{T - 1 - j}" if j < T else
                             f"decoder day +{j - T + 1}", "days_back_from_origin": (T - 1 - j) if j < T else np.nan,
                             "mean_attention_weight": am[h, j]})
        save(pd.DataFrame(rows), "tft_attention.csv", "models/tft/<latest> attention weights (head-averaged, mean over "
             "validation origins)")
    else:
        note("TFT attention weights were not returned by the model")
    return out


# ============================================================================ 9-10. ensemble / calibration
def ensemble_tables(bt_win) -> None:
    b = bt_win[bt_win["model"].isin(["lstm_w@compact", "lgbm@compact", "ensemble"])].dropna(subset=["actual", "p50"]).copy()
    b["ape"] = ape(b)
    rows = []
    for t in TGT:
        for h in list(range(1, 8)) + ["all"]:
            g = b[(b["target"] == t) & ((b["h"] == h) if h != "all" else True)]
            r = g.groupby("model")["ape"].mean()
            rows.append({"target": t, "days_ahead": h, "lstm_mape_pct": r["lstm_w@compact"], "lightgbm_mape_pct": r["lgbm@compact"],
                         "ensemble_mape_pct": r["ensemble"],
                         "ensemble_gain_vs_best_member_pp": min(r["lstm_w@compact"], r["lgbm@compact"]) - r["ensemble"]})
    save(pd.DataFrame(rows), "ensemble_vs_members.csv", "data/processed/backtest_processed.parquet (12-month window)")
    post = json.loads((PROCESSED / "postprocess.json").read_text())
    info = json.loads((PROCESSED / "fixed_split_info.json").read_text())
    rows = []
    for t in TGT:
        for h in range(1, 8):
            w = post["ensemble"][t][str(h)]
            rows.append({"target": t, "days_ahead": h, "w_lstm": w, "w_lightgbm": 1 - w,
                         "w_lstm_fixed_split": info["post"]["ensemble"][t][str(h)]})
    save(pd.DataFrame(rows), "ensemble_weights.csv", "data/processed/postprocess.json ('ensemble': live weights fitted on "
         "the last 180 days of backtest rows) and fixed_split_info.json ('post': weights fitted on the fixed-split validation)")


def calibration(bt_win) -> pd.DataFrame:
    rows = []
    for (m, t, h), g in bt_win[bt_win["model"].isin(QMODELS)].groupby(["model", "target", "h"]):
        g = g.dropna(subset=["actual", "p10_raw", "p90_raw"])
        ins = lambda lo, hi: float(((g["actual"] >= lo) & (g["actual"] <= hi)).mean() * 100)   # noqa: E731
        rows.append({"model": LABEL[m], "model_key": m, "target": t, "days_ahead": int(h), "unit": unit_of(t),
                     "coverage_before_pct": ins(g["p10_raw"], g["p90_raw"]), "coverage_after_pct": ins(g["p10"], g["p90"]),
                     "width_before": float((g["p90_raw"] - g["p10_raw"]).mean()), "width_after": float((g["p90"] - g["p10"]).mean()),
                     "target_coverage_pct": 80.0, "n_forecasts": len(g)})
    df = pd.DataFrame(rows)
    df["width_change_pct"] = (df["width_after"] / df["width_before"] - 1) * 100
    return save(df, "calibration.csv", "data/processed/backtest_processed.parquet (p10_raw/p90_raw = before, p10/p90 = after "
                "per-horizon adaptive conformal calibration; 12-month window)")


# ============================================================================ 11-12. curves & ladder
def training_curves(lgb_curve) -> None:
    rows = []
    for kind, name in (("lstm", "LSTM"), ("tft", "TFT")):
        d = latest_dir(kind)
        h = pd.read_csv(d / "history.csv")
        best = int(h["val_loss"].idxmin()) + 1
        rows.append(pd.DataFrame({"model": name, "epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                                  "learning_rate": h["lr"], "epoch_seconds": h["seconds"], "refit_loss_train_plus_val": h.get("refit_loss"),
                                  "is_best_epoch": h["epoch"] == best, "loss_unit": "mean pinball, standardised ratio",
                                  "source_dir": str(d.relative_to(ROOT))}))
    save(pd.concat(rows, ignore_index=True), "training_curves.csv", "models/lstm/<latest>/history.csv and models/tft/<latest>/history.csv")
    save(lgb_curve, "lightgbm_training_curve.csv", "NOT stored by the project; recomputed in memory with the same fixed split "
         "and hyper-parameters as ablation.csv (all features). Not the production model")


def ladder(model_metrics_df) -> pd.DataFrame:
    m = model_metrics_df[model_metrics_df["window"] == "rolling_12m_backtest"]
    order = ["seasonal_naive", "lgbm@compact", "lstm_w@compact", "tft@v1", "ensemble", "grid_india"]
    rows = []
    prev = None
    for i, k in enumerate(order, 1):
        r = m[m["model_key"] == k].set_index("target")["mape_pct"]
        row = {"step": i, "model": LABEL[k], "model_key": k, "solar_peak_mape_pct": r["solar_peak"],
               "evening_peak_mape_pct": r["evening_peak"], "energy_mape_pct": r["energy"],
               "overall_mape_pct": r["overall_mean_of_3_targets"]}
        row["overall_change_vs_previous_step_pp"] = row["overall_mape_pct"] - prev if prev is not None else np.nan
        row["improves_on_previous_step"] = (row["overall_mape_pct"] < prev) if prev is not None else np.nan
        prev = row["overall_mape_pct"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rank_overall (1=best)"] = df["overall_mape_pct"].rank().astype(int)
    return save(df, "model_ladder.csv", "model_metrics.csv (rolling 12-month backtest), steps in the requested order")


# ============================================================================ 13. worked example
def worked_example(daily, fc, cfgs) -> None:
    from src.models.deep import DeepForecaster

    D = EXAMPLE_DAY
    rows = []

    def add(section, name, value, unit="", date=D, h=np.nan, target="", scaled=np.nan, mu=np.nan, sd=np.nan,
            p10=np.nan, p50=np.nan, p90=np.nan, model="", source=""):
        rows.append({"section": section, "name": name, "date": pd.Timestamp(date).date() if date is not None else None,
                     "days_ahead": h, "target": target, "model": model, "value": value, "unit": unit, "scaled_value": scaled,
                     "scaler_mean": mu, "scaler_sd": sd, "p10": p10, "p50": p50, "p90": p90, "source": source})

    row = daily.loc[D]
    for t in TGT:
        add("1 raw target value", t, row[t], unit_of(t), target=t, source="features.parquet (cleaned Grid-India value; "
            f"imputed={bool(row[t + '_imputed'])})")
    for c in [c for c in daily.columns if c.endswith("_ai") or c.endswith("_ai_anom")]:
        add("2 all-India weather (observed, demand-weighted)", c, row[c], source="features.parquet")
    add("2 all-India weather (observed, demand-weighted)", "weather_regime", ev.weather_regimes(daily).loc[D],
        source="src.evaluate.weather_regimes")
    # lag / rolling values for the forecast issued on D (origin = D): ratios to the 7-day reference level, and in raw units
    tab = fb.make_tabular(daily, fc, [D])
    for t in TGT:
        ref = float(tab[f"ref_{t}"].iloc[0])
        add("3 lag/rolling (origin = example day)", f"{t}_ref7 (7-day mean level)", ref, unit_of(t), target=t,
            source="src.features.build.reference_levels")
        for f in ("last", "roll30", "std7", "std30"):
            v = float(tab[f"{t}_{f}"].iloc[0])
            add("3 lag/rolling (origin = example day)", f"{t}_{f} (ratio to ref7)", v, "ratio", target=t, source="make_tabular")
            add("3 lag/rolling (origin = example day)", f"{t}_{f} (raw units)", v * ref, unit_of(t), target=t, source="ratio x ref7")
        for lag in ("lag7", "lag14", "lag364"):
            for _, r in tab.iterrows():
                v = float(r[f"{t}_{lag}"]) if pd.notna(r[f"{t}_{lag}"]) else np.nan
                add("3 lag/rolling (origin = example day)", f"{t}_{lag} (ratio to ref7)", v, "ratio", date=r["date"],
                    h=int(r["h"]), target=t, source="make_tabular (value of the day h ahead minus lag)")
                add("3 lag/rolling (origin = example day)", f"{t}_{lag} (raw units)", v * ref, unit_of(t), date=r["date"],
                    h=int(r["h"]), target=t, source="ratio x ref7")
    # what the LSTM reads
    m = DeepForecaster.load(latest_dir("lstm"))
    a = m._arrays(daily, fc, [D])
    e, d, _ = m._scale(a)
    for i, n in enumerate(m.enc_feats):
        add("4 LSTM encoder input, last day of the 60-day window", n, float(a["enc"][0, -1, i]), "raw",
            scaled=float(e[0, -1, i]), mu=float(m.enc_mu[i]), sd=float(m.enc_sd[i]), source="make_sequences + scaler.npz "
            "(scaled = clip((raw-mu)/sd, -10, 10))")
    for h in range(HORIZON):
        for i, n in enumerate(m.dec_feats):
            add("5 LSTM decoder input (known-future features)", n, float(a["dec"][0, h, i]), "raw",
                date=D + pd.Timedelta(days=h + 1), h=h + 1, scaled=float(d[0, h, i]), mu=float(m.dec_mu[i]),
                sd=float(m.dec_sd[i]), source="make_sequences + scaler.npz")
    # forecasts
    hist = pd.read_parquet(PROCESSED / "forecast_history.parquet")
    lat = pd.read_parquet(PROCESSED / "latest_forecast.parquet")
    for lab, fdf in (("6 forecast issued on the example day (origin 2026-11-14)", hist[hist["origin"] == D]),
                     ("7 latest stored forecast (origin %s)" % lat["origin"].max().date(), lat)):
        for _, r in fdf.sort_values(["model", "target", "h"]).iterrows():
            add(lab, f"{r['model']} forecast", r["actual"], unit_of(r["target"]), date=r["date"], h=int(r["h"]),
                target=r["target"], model=r["model"], p10=r["p10"], p50=r["p50"], p90=r["p90"],
                source="forecast_history.parquet" if lab.startswith("6") else "latest_forecast.parquet")
    df = pd.DataFrame(rows).rename(columns={"value": "value (raw / actual where known)"})
    save(df, "worked_example_2026-11-14.csv", "features.parquet, make_tabular/make_sequences (src.features.build), "
         "models/lstm/<latest>/scaler.npz, forecast_history.parquet, latest_forecast.parquet")


# ============================================================================ main
def main():
    t00 = time.time()
    set_seed(SEED)
    OUT.mkdir(exist_ok=True)
    print("Export ->", OUT)
    cfgs = {k: yaml.safe_load((latest_dir(k) / "config.yaml").read_text()) for k in ("lgbm", "lstm", "tft")}
    daily, fc = fb.load_daily()

    p_bt, src_bt = find_file("data/processed/backtest_processed.parquet")
    p_fx, src_fx = find_file("data/processed/fixed_split_predictions.parquet")
    print("backtest_processed.parquet from:", src_bt, "| fixed_split_predictions.parquet from:", src_fx)
    bt_all = pd.read_parquet(p_bt)
    bt_win = ev.eval_window(bt_all)
    fixed = pd.read_parquet(p_fx)
    fixed = fixed[fixed["date"] >= pd.Timestamp(fx.TEST_START)].dropna(subset=["actual"])
    gi_fixed = bt_all[(bt_all["model"] == "grid_india") & (bt_all["date"] >= pd.Timestamp(fx.TEST_START))]
    fixed = pd.concat([fixed, gi_fixed], ignore_index=True)
    if src_fx != "this project":
        note(f"fixed_split_predictions.parquet is not in this project; read from {src_fx} (its fixed_split_tables.parquet and "
             "fixed_split_info.json are identical to this project's; MAPEs reproduce fixed_split_tables.parquet exactly).")
    SRC = {"backtest": src_bt, "fixed": src_fx}

    print("1 model cards");           cards = model_cards(cfgs, daily)
    print("2 model metrics");         mm = model_metrics(bt_win, bt_all, fixed)
    print("3 by horizon");            by_horizon(bt_win)
    print("4 predictions long");      predictions_long(bt_all)
    print("5-6 error tables");        error_tables(bt_win, daily)
    print("7 ablation + LightGBM curve (in-memory fits)"); abl, lgb_curve = ablation(cfgs)
    print("8 importance");            imp = importance(cfgs, daily, fc)
    print("9 ensemble");              ensemble_tables(bt_win)
    print("10 calibration");          cal = calibration(bt_win)
    print("11 curves");               training_curves(lgb_curve)
    print("12 ladder");               lad = ladder(mm)
    print("13 worked example");       worked_example(daily, fc, cfgs)

    checks(mm, cfgs, abl, cal, lad, imp, bt_win)
    write_docs(SRC)
    zip_up()
    print(f"done in {time.time() - t00:.0f}s")


def checks(mm, cfgs, abl, cal, lad, imp, bt_win) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100 + "\nCHECK 1: headline numbers (12-month rolling backtest, MAPE %)\n" + "=" * 100)
    exp = {"ensemble": (2.40, 1.79, 2.00, 2.06), "grid_india": (4.73, 3.08, 3.87, 3.89), "seasonal_naive": (4.74, 3.15, 3.93, None)}
    r = mm[mm["window"] == "rolling_12m_backtest"]
    rows, ok_all = [], True
    for k, e in exp.items():
        g = r[r["model_key"] == k].set_index("target")["mape_pct"]
        got = (g["solar_peak"], g["evening_peak"], g["energy"], g["overall_mean_of_3_targets"])
        for name, x, y in zip(("solar", "evening", "energy", "overall"), got, e):
            if y is None:
                continue
            ok = abs(round(x, 2) - y) < 0.005 + 1e-9
            ok_all &= ok
            rows.append({"model": k, "target": name, "expected": y, "reproduced": round(x, 3), "match": "OK" if ok else "DIFFERS"})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print("-> all headline numbers reproduce" if ok_all else "-> SOME HEADLINE NUMBERS DIFFER (see table)")

    print("\n" + "=" * 100 + "\nCHECK 2: feature counts\n" + "=" * 100)
    L, S, T = cfgs["lgbm"], cfgs["lstm"], cfgs["tft"]
    sets = {"LightGBM": set(L["features"]), "LSTM": set(S["enc_features"]) | set(S["dec_features"]),
            "TFT": set(T["enc_features"]) | set(T["dec_features"])}
    allf = set().union(*sets.values())
    gc = pd.Series([ai_group(f) for f in allf]).value_counts()
    rows = [("distinct (all models)", len(allf), 95), ("LightGBM", len(sets["LightGBM"]), 87), ("LSTM distinct", len(sets["LSTM"]), 67),
            ("LSTM encoder", len(S["enc_features"]), 48), ("LSTM decoder", len(S["dec_features"]), 60),
            ("TFT distinct", len(sets["TFT"]), 53), ("TFT encoder", len(T["enc_features"]), 34), ("TFT decoder", len(T["dec_features"]), 46)]
    rows += [(f"group {g} (of 95)", int(gc.get(g, 0)), v) for g, v in (("Weather", 35), ("Calendar", 26), ("Lag", 12), ("Rolling", 9), ("Other", 13))]
    t = pd.DataFrame(rows, columns=["item", "found", "expected"])
    t["match"] = np.where(t["found"] == t["expected"], "OK", "DIFFERS")
    print(t.to_string(index=False))
    from src.walkthrough import feature_group as wg
    wc = pd.Series([wg(f) for f in allf]).value_counts().to_dict()
    print("(src.walkthrough.feature_group would give:", wc, ")")

    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    for t_ in TGT:
        g = r[(r["target"] == t_) & r["model_key"].isin(HEAD)].set_index("model")["mape_pct"]
        print(f"  {t_:13s} best: {g.idxmin()} ({g.min():.2f} %)   worst: {g.idxmax()} ({g.max():.2f} %)")
    eg = pd.read_csv(OUT / "errors_by_group.csv")
    e = eg[(eg["model_key"] == "ensemble") & (eg["n_days"] >= 10)]
    e = e.assign(ratio=e["mape_pct"] / e.groupby("target")["mape_pct"].transform("mean"))
    top = e.sort_values("mape_pct", ascending=False).head(3)
    print("  biggest ensemble error-by-group findings (groups with >= 10 days):")
    for _, x in top.iterrows():
        print(f"    {x['target']:13s} {x['group_type']}: {x['group']} -> MAPE {x['mape_pct']:.2f} % ({x['n_days']} days)")
    lo = e.sort_values("mape_pct").head(1).iloc[0]
    print(f"    lowest: {lo['target']} {lo['group_type']}: {lo['group']} -> {lo['mape_pct']:.2f} %")
    a = abl[abl["variant"].isin(["all_features", "no_weather", "no_weather_incl_ref_levels (extra)"])]
    print("  effect of removing weather (LightGBM, fixed split, test MAPE %):")
    print(a.pivot_table(index="target", columns="variant", values="test_mape_pct").round(2).to_string())
    print(a[a["variant"] == "no_weather"][["target", "test_change_vs_all_pp", "test_change_vs_all_pct"]].round(2).to_string(index=False))
    allf_ = abl[abl["variant"] == "all_features"].set_index("target")["test_mape_pct"]
    st = pd.read_parquet(PROCESSED / "fixed_split_tables.parquet")
    st = st[(st.kind == "overall") & (st.model == "lgbm@compact")].set_index("target")["MAPE"]
    print("  sanity: all-features fresh fit test MAPE vs stored fixed-split LightGBM:",
          {t_: (round(allf_[t_], 3), round(st[t_], 3)) for t_ in TGT})
    cb = cal[cal["model_key"] == "ensemble"]
    print(f"  ensemble coverage 12m: before {cb['coverage_before_pct'].mean():.1f} % -> after {cb['coverage_after_pct'].mean():.1f} % (target 80)")
    ec, _ = find_file("data/processed/eval_coverage.parquet")
    if ec is not None:
        e0 = pd.read_parquet(ec)
        j = cb.merge(e0[e0["model"] == "ensemble"], left_on=["target", "days_ahead"], right_on=["target", "h"])
        d1, d2 = (j["coverage_before_pct"] - j["before"]).abs().max(), (j["coverage_after_pct"] - j["after"]).abs().max()
        print(f"  cross-check vs stored eval_coverage.parquet (ensemble): max |diff| before {d1:.3f}, after {d2:.3f} pp")
    print("  ladder:")
    print(lad[["step", "model", "overall_mape_pct", "overall_change_vs_previous_step_pp", "improves_on_previous_step"]].round(3).to_string(index=False))
    bad = lad[lad["improves_on_previous_step"] == False]   # noqa: E712
    if len(bad):
        msg = ("model_ladder.csv: the requested order is NOT monotone: " + "; ".join(
            f"{x['model']} is worse than the previous step ({x['overall_change_vs_previous_step_pp']:+.2f} pp)" for _, x in bad.iterrows()))
        note(msg)


def write_docs(SRC) -> None:
    lines = ["# ai_data: files, columns, source", "",
             "Generated by `scripts/export_ai_data.py` (read-only, seed 42, no network). Percent columns are percent; "
             "`unit` is MW (peaks) or MU (energy). 12-month window = forecasts issued 2025-11-01 .. 2026-11-07 "
             "(Grid-India: target days from 2025-11-01).", "",
             f"Stored-file sources actually used: backtest_processed.parquet <- {SRC['backtest']}; "
             f"fixed_split_predictions.parquet <- {SRC['fixed']}; everything else from this project.", ""]
    for n, s in SOURCES.items():
        lines.append(f"- **{n}** - columns: {README_COLS[n]}. Source: {s}.")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    static = [
        "Feature groups: the requested split (weather 35, calendar 26, lag 12, rolling 9, other 13) is NOT what "
        "`src/walkthrough.feature_group` gives (31/25/9/3/27). It is reproduced exactly by `ai_group()` in the script: Lag = "
        "`*lag*` and `*_last`; Rolling = `*roll*`, `*_std7`, `*_std30`; Weather = weather variables plus `d_*` weather deltas "
        "and `wx_source`; Calendar = calendar/festival/holiday features incl. `ref_holidays`; Other = the rest (target `_r` and "
        "`_mask` inputs, `h`, `h_frac`, `is_imputed`, `ref_imputed`, `ref_tmax_ai`, `ref_cdd_ai`, `ref_app_tmax_ai`). So 'no weather' "
        "still keeps the three `ref_*` weather reference levels; an extra row `no_weather_incl_ref_levels (extra)` removes them too.",
        "Ablation uses the fixed split (fit origins to 2024-12, validation Jan-Mar 2025, test from 2025-04-01) and "
        "`models/lgbm/best_params_fixed_split.yaml`, not the production hyper-parameters (the production model is trained "
        "on all data, so it has no held-out test). Validation MAPE is optimistic: the validation slice drives early stopping "
        "and the conformal term. `weather_only` / `calendar_only` also remove `h` (it is in group Other).",
        "The all-features ablation fit reproduces the stored fixed-split LightGBM test MAPE to within 0.04 pp (3.34/2.30/2.81 "
        "vs 3.31/2.27/2.77 for solar/evening/energy): the stored fit is not byte-reproducible from the current feature "
        "build (the weather archive has grown since). Test origins are cut at 2026-11-07 to match the stored split.",
        "Validation set for LSTM permutation importance and TFT variable-selection/attention: the last 15 % of origins (the "
        "model's own validation definition). The shipped models were re-fitted on train+validation afterwards, so this "
        "slice is not unseen data; read the importances as 'what the model relies on', not as out-of-sample skill.",
        "LightGBM validation loss per iteration is not stored by the project; `lightgbm_training_curve.csv` is recomputed in "
        "memory on the fixed split (not the production model).",
        "Training time: for LSTM/TFT `training_seconds` in model_cards = sum of per-epoch seconds in history.csv (main loop "
        "only); `extra.training_log_seconds_incl_refit` has the training_log.jsonl figure. LightGBM time is from training_log.jsonl.",
        "Trained-model parameter counts for LSTM/TFT come from instantiating the stored networks; LightGBM 'parameters' = total "
        "trees (sum of the 9 boosters).",
        "metrics_by_horizon.csv excludes Grid-India: its `h` counts days from its forecast-week start, not from an issue date.",
        "predictions_long.csv holds the 6 headline models only (not the 13 other backtest variants). Grid-India has no P10/P90.",
        "Fixed-test-split bias % and Grid-India rows are computed here from fixed_split_predictions.parquet and the Grid-India rows "
        "of backtest_processed.parquet (date >= 2025-04-01; Grid-India days start 2025-05-01 in the backtest file).",
        "worst_days.csv / errors_by_group.csv: a target day's error is the mean over its available days-ahead forecasts. "
        "Weather condition and tmax_ai quartiles use OBSERVED weather (evaluation labels only).",
        "worked_example: lag/rolling values are computed for forecast origin 2026-11-14 (the example day as the last observed "
        "day); LSTM scaled values are the last encoder day (2026-11-14) and the 7 decoder days after it. A forecast issued on "
        "2026-11-14 and the latest stored forecast (origin 2026-11-21) are both included; `actual` is in the value column where known.",
    ]
    (OUT / "MISSING.md").write_text("# Missing, not feasible, or different from the request\n\n"
                                    + "\n".join(f"- {s}" for s in static + MISSING) + "\n")


def zip_up() -> None:
    import zipfile

    z = ROOT / "ai_data.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(ROOT))
    print(f"\nzip: {z}  ({z.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
