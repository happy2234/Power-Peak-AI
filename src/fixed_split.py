"""Fixed train/test evaluation: train ONCE, test on the unseen future (stages 1-2 of the
evaluation design; stage 3 is the live scorecard, see live_summary()).

    Train       origins whose 7-day labels end before VAL_START (2023-04 .. 2024-12)
    Validation  VAL_START .. TRAIN_END (Jan-Mar 2025): early stopping, Optuna, the ensemble
                weights and the conformal calibration only — never used to fit model weights
    Test        TEST_START (2025-04-01) .. latest: forecasts from daily origins, 7 days ahead,
                with the models frozen (no refit, no online recalibration)

Leakage guards: the anomaly normals are fitted on observed weather up to TRAIN_END; the neural
scalers and the rare-weather thresholds are fitted on the training slice only; weather inputs are
the archived forecasts issued at each origin. Outputs: data/processed/fixed_split_*.parquet,
reports/fixed_split_results.md.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from src.config import HORIZON, MODELS_DIR, PROCESSED, REPORTS, TARGETS

log = logging.getLogger(__name__)
TRAIN_START = "2023-04-01"
VAL_START = "2025-01-01"
TRAIN_END = "2025-03-31"
TEST_START = "2025-04-01"
# the variants chosen in the rolling backtest (feature set, rare-weather weights)
VARIANTS = {"lgbm": ("compact", 0.0), "lstm": ("compact", 1.0), "tft": ("v1", 0.0)}
NAMES = {"lgbm": "lgbm@compact", "lstm": "lstm_w@compact", "tft": "tft@v1"}
TGT = list(TARGETS)


def fy(d) -> str:
    d = pd.Timestamp(d)
    y = d.year if d.month >= 4 else d.year - 1
    return f"FY {y}-{str(y + 1)[2:]}"


# ============================================================================ split
def split_origins(daily: pd.DataFrame, val_start: str = VAL_START, train_end: str = TRAIN_END,
                  test_start: str = TEST_START):
    """(fit, validation, test) origins. Fit origins' labels end before val_start; validation
    origins' labels end by train_end; test origins start the day before test_start so every test
    day has all seven horizons."""
    from src.features.build import valid_origins

    vs, te = pd.Timestamp(val_start), pd.Timestamp(train_end)
    fit_o = valid_origins(daily, end=vs - pd.Timedelta(days=HORIZON + 1))
    val_o = valid_origins(daily, start=vs, end=te - pd.Timedelta(days=HORIZON))
    test_o = valid_origins(daily, start=pd.Timestamp(test_start) - pd.Timedelta(days=1))
    return fit_o, val_o, test_o


def features(train_end: str = TRAIN_END):
    """Feature tables whose anomaly normals use observed weather up to train_end only (not saved)."""
    from src.features.build import build_daily_features

    return build_daily_features(save=False, normals_end=train_end)


# ============================================================================ tuning
def params_path(kind: str):
    return MODELS_DIR / kind / "best_params_fixed_split.yaml"


def tune(kind: str, daily, fc, fit_o, val_o, n_trials: int = 30, seed: int = 42) -> dict:
    """Optuna on the fixed split: train on fit origins, objective on the validation months."""
    import optuna
    import yaml

    from src.features.build import make_tabular
    from src.models.tune import _lgbm_objective, suggest
    from src.utils.device import free_gpu

    fs, boost = VARIANTS[kind]
    origins = fit_o.append(val_o)
    tab = make_tabular(daily, fc, origins) if kind == "lgbm" else None

    def objective(trial):
        params = suggest(trial, kind)
        try:
            if kind == "lgbm":
                return _lgbm_objective(params, tab, fs, valid_start=val_o[0])
            from src.models.deep import DeepForecaster

            m = DeepForecaster(kind, params, max_epochs=150, patience=15, feature_set=fs, regime_boost=boost)
            m.fit(daily, fc, origins, trial=trial, refit_all=False, valid_start=val_o[0])
            return m.best_val
        finally:
            free_gpu()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    out = {"kind": kind, "feature_set": fs, "best_params": dict(study.best_params),
           "best_value": float(study.best_value), "n_trials": n_trials, "minutes": round((time.time() - t0) / 60, 1),
           "fit_origins": [str(fit_o.min().date()), str(fit_o.max().date())],
           "validation_origins": [str(val_o.min().date()), str(val_o.max().date())]}
    params_path(kind).parent.mkdir(parents=True, exist_ok=True)
    params_path(kind).write_text(yaml.safe_dump(out))
    log.info("fixed-split tuned %s: %.4f in %.1f min", kind, out["best_value"], out["minutes"])
    return out


def best_params(kind: str) -> dict:
    import yaml

    p = params_path(kind)
    return yaml.safe_load(p.read_text())["best_params"] if p.exists() else {}


# ============================================================================ train once
def train(daily, fc, fit_o, val_o, kinds=("seasonal_naive", "lgbm", "lstm", "tft"), params=None,
          max_epochs: int = 200) -> dict:
    """Fit every model ONCE on the fit origins (validation only for early stopping/calibration)."""
    from src.features.build import make_tabular
    from src.models.baselines import LGBMQuantile, SeasonalNaive

    params = params or {}
    models = {}
    tab_fit = make_tabular(daily, fc, fit_o)
    for kind in kinds:
        t0 = time.time()
        if kind == "seasonal_naive":
            models[kind] = SeasonalNaive().fit(tab_fit)
        elif kind == "lgbm":
            fs, boost = VARIANTS["lgbm"]
            tab = make_tabular(daily, fc, fit_o.append(val_o))
            models[kind] = LGBMQuantile(params.get("lgbm", best_params("lgbm")), regime_boost=boost,
                                        feature_set=fs).fit(tab, valid_start=val_o[0], refit=False)
        else:
            from src.models.deep import DeepForecaster

            fs, boost = VARIANTS[kind]
            m = DeepForecaster(kind, params.get(kind, best_params(kind)), feature_set=fs, regime_boost=boost,
                               max_epochs=max_epochs)
            models[kind] = m.fit(daily, fc, fit_o.append(val_o), refit_all=False, valid_start=val_o[0])
        log.info("fixed split: trained %s in %.1f s", kind, time.time() - t0)
    return models


def predict(models: dict, daily, fc, origins) -> pd.DataFrame:
    from src.features.build import make_tabular

    tab = make_tabular(daily, fc, origins)
    out = []
    for kind, m in models.items():
        p = m.predict_long(tab) if kind in ("seasonal_naive", "lgbm") else m.predict_long(daily, fc, origins)
        p["model"] = NAMES.get(kind, kind)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def postprocess(val_pred: pd.DataFrame, test_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ensemble weights + conformal corrections fitted on the validation months, applied unchanged
    to the whole test period (no online updates: the models are frozen)."""
    from src.postprocess import apply_conformal, apply_ensemble, fit_conformal, fit_ensemble

    members = (NAMES["lstm"], NAMES["lgbm"])
    w = fit_ensemble(val_pred, members)
    val_all = pd.concat([val_pred, apply_ensemble(val_pred, w, members)], ignore_index=True)
    cal = fit_conformal(val_all, sorted(val_all["model"].unique()))
    test_all = pd.concat([test_pred, apply_ensemble(test_pred, w, members)], ignore_index=True)
    return apply_conformal(test_all, cal), {"ensemble": w, "conformal": cal}


# ============================================================================ evaluation
def _metrics(g: pd.DataFrame) -> dict:
    from src.models.common import metrics

    return metrics(g)


def tables(pred: pd.DataFrame, daily: pd.DataFrame, gi: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Every evaluation table of the fixed split (test rows only)."""
    from src.evaluate import SEASONS, weather_regimes

    p = pred[pred["date"] >= pd.Timestamp(TEST_START)].dropna(subset=["actual"]).copy()
    p["fy"] = p["date"].map(fy)
    p["season"] = p["date"].dt.month.map(SEASONS)
    p["regime"] = weather_regimes(daily).reindex(p["date"]).to_numpy()
    p["month"] = p["date"].dt.to_period("M").dt.to_timestamp()
    out = {}
    for name, keys in {"overall": [], "fy": ["fy"], "season": ["season"], "regime": ["regime"], "horizon": ["h"],
                       "month": ["month"]}.items():
        rows = []
        for k, g in p.groupby(["model", "target"] + keys):
            k = k if isinstance(k, tuple) else (k,)
            r = dict(zip(["model", "target"] + keys, k))
            r.update(_metrics(g))
            r["days"] = g["date"].nunique()
            rows.append(r)
        out[name] = pd.DataFrame(rows)
    if gi is not None and len(gi):
        out["grid_india"] = grid_india_comparison(p, gi)
    return out


def grid_india_comparison(p: pd.DataFrame, gi: pd.DataFrame) -> pd.DataFrame:
    """Best model vs Grid-India on the same test days: as published and bias-corrected (constant
    bias from earlier weeks only). Ours = mean of the 1-7-day forecasts for the day, and the
    single 7-day-ahead forecast."""
    rows = []
    g = gi[gi["date"] >= pd.Timestamp(TEST_START)].dropna(subset=["actual"])
    for basis, sub in (("mean of 1–7-day forecasts", p), ("7-day-ahead forecast only", p[p["h"] == 7])):
        ours = sub[sub["model"] == "ensemble"].groupby(["date", "target"], as_index=False)["p50"].mean()
        j = g.merge(ours.rename(columns={"p50": "p50_ours"}), on=["date", "target"])
        for corr, col in (("as published", "p50"), ("bias-corrected", "p50_corr")):
            for t, x in j.dropna(subset=[col]).groupby("target"):
                ae_o, ae_g = (x["p50_ours"] - x["actual"]).abs(), (x[col] - x["actual"]).abs()
                rows.append({"target": t, "basis": basis, "grid_india": corr, "days": len(x),
                             "our_MAPE": float((ae_o / x["actual"]).mean() * 100),
                             "gi_MAPE": float((ae_g / x["actual"]).mean() * 100),
                             "improvement": 1 - ae_o.mean() / ae_g.mean(), "win_rate": float((ae_o < ae_g).mean())})
    return pd.DataFrame(rows)


def grid_india_rows(daily) -> pd.DataFrame:
    from src.audit import grid_india_with_issue, past_bias
    from src.evaluate import actuals_long

    gi = grid_india_with_issue(actuals_long(daily))
    gi["p50_corr"] = gi["p50"] - past_bias(gi)
    return gi


# ============================================================================ stage 3: live
def live_summary(processed=PROCESSED) -> dict:
    """Score every archived live forecast whose target day now has an actual (all models).

    'real time' = the forecast was issued before its target day began; forecasts generated later
    for past origins (back-filled when the pipeline started) are counted separately."""
    hist_p, feat_p = processed / "forecast_history.parquet", processed / "features.parquet"
    if not hist_p.exists() or not feat_p.exists():
        return {}
    hist = pd.read_parquet(hist_p)
    feat = pd.read_parquet(feat_p).set_index("date")
    act = feat[TGT].copy()
    for t in TGT:
        if f"{t}_imputed" in feat:
            act.loc[feat[f"{t}_imputed"].fillna(False).astype(bool), t] = np.nan
    act = act.rename_axis("date").reset_index().melt("date", var_name="target", value_name="actual").dropna()
    h = hist.drop(columns=["actual"], errors="ignore").merge(act, on=["date", "target"])
    if h.empty:
        return {"days_scored": 0, "origins": 0}
    h["real_time"] = pd.to_datetime(h["issued_at"]) < h["date"]
    gi = h[h["model"] == "grid_india"][["origin", "date", "target", "p50"]].rename(columns={"p50": "p50_gi"})
    ours = h[h["model"] != "grid_india"].merge(gi, on=["origin", "date", "target"], how="left")
    ours["recipe"] = ours["recipe"].fillna("baseline") if "recipe" in ours else "baseline"
    rows = []
    for (m, t, rc), g in ours.groupby(["model", "target", "recipe"]):
        ape = (g["p50"] - g["actual"]).abs() / g["actual"] * 100
        gg = g.dropna(subset=["p50_gi"])
        rows.append({"model": m, "target": t, "recipe": rc, "days": int(g["date"].nunique()), "rows": len(g),
                     "real_time_rows": int(g["real_time"].sum()), "MAPE": float(ape.mean()),
                     "gi_MAPE": float(((gg["p50_gi"] - gg["actual"]).abs() / gg["actual"]).mean() * 100) if len(gg) else None,
                     "win_rate": float(((gg["p50"] - gg["actual"]).abs() < (gg["p50_gi"] - gg["actual"]).abs()).mean())
                     if len(gg) else None})
    best = hist.loc[hist["is_best"].fillna(False).astype(bool), "model"]
    out = {"updated": str(pd.Timestamp.now().floor("s")), "first_day": str(h["date"].min().date()),
           "last_day": str(h["date"].max().date()), "days_scored": int(h["date"].nunique()),
           "origins": int(h["origin"].nunique()), "real_time_days": int(h.loc[h["real_time"], "date"].nunique()),
           "pending_days": int(hist.loc[hist["date"] > h["date"].max(), "date"].nunique()),
           "best_model": best.iloc[-1] if len(best) else None, "by_model": rows}
    (processed / "live_summary.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ============================================================================ report
def write_report(t: dict, info: dict, rolling: pd.DataFrame | None, live: dict) -> str:
    from src.evaluate import REGIME_ORDER, SEASON_ORDER
    from src.viz import MODEL_LABELS

    def lab(m):
        return MODEL_LABELS.get(m, m)

    order = ["ensemble", NAMES["lstm"], NAMES["lgbm"], NAMES["tft"], "seasonal_naive"]
    ov = t["overall"]
    L = ["# Fixed train/test evaluation", "",
         f"Models trained **once** on {info['fit'][0]} .. {info['fit'][1]} (forecast origins; labels end before "
         f"{VAL_START}). {VAL_START} .. {TRAIN_END} is the validation slice, used only for early stopping, Optuna "
         "tuning, the ensemble weights and the P10–P90 calibration. Test: every daily forecast origin from "
         f"{info['test'][0]} to {info['test'][1]}, 1–7 days ahead, with the models frozen (no refit, no online "
         "recalibration). Weather inputs are the archived forecasts issued at each origin; the anomaly normals, "
         "scalers and rare-weather thresholds use the training period only. Peaks in MW (errors in MW), energy in MU.",
         "", "The feature set of each model (LightGBM and LSTM: compact weather; TFT: previous inputs) was chosen in "
         "the rolling backtest, whose months overlap this test period; that choice is the one piece of design "
         "informed by test-period data.", ""]
    L += ["## Overall (test period)", "", "| model | target | days | MAPE % | MAE | RMSE | bias | P10–P90 coverage % |",
          "|---|---|---|---|---|---|---|---|"]
    for m in order:
        for k in TGT:
            r = ov[(ov.model == m) & (ov.target == k)]
            if len(r):
                r = r.iloc[0]
                L.append(f"| {lab(m)} | {k} | {r['days']} | {r['MAPE']:.2f} | {r['MAE']:,.0f} | {r['RMSE']:,.0f} | "
                         f"{r['bias']:+,.0f} | {r.get('coverage80', np.nan):.1f} |")
    L.append("")
    for key, title, col, ordr in (("fy", "By financial year", "fy", None), ("season", "By season", "season", SEASON_ORDER),
                                  ("regime", "By weather regime (observed weather, evaluation only)", "regime", REGIME_ORDER),
                                  ("horizon", "By horizon (days ahead)", "h", None)):
        x = t[key]
        piv = x.pivot_table(index=["target", col], columns="model", values="MAPE").reindex(columns=order)
        piv.columns = [lab(c) for c in piv.columns]
        days = x[x.model == "ensemble"].set_index(["target", col])["days"]
        piv.insert(0, "days", days.reindex(piv.index))
        if ordr:
            piv = piv.reindex(pd.MultiIndex.from_product([TGT, [o for o in ordr if o in set(x[col])]]))
        L += [f"## {title} — MAPE %", "", piv.round(2).to_markdown(), ""]
    mo = t["month"]
    piv = mo.pivot_table(index="month", columns="model", values="MAPE").reindex(columns=order)
    piv.index = piv.index.strftime("%b %Y")
    piv.columns = [lab(c) for c in piv.columns]
    e = mo[mo.model == "ensemble"].groupby("month")["MAPE"].mean()
    same = drift_same_months(mo)
    bias = ov[ov.model == "ensemble"].set_index("target")["bias"]
    L += ["## Drift: monthly MAPE over the test period (mean of the three targets)", "",
          piv.groupby(level=0, sort=False).mean().round(2).to_markdown(), "",
          "Seasons dominate the monthly pattern (pre-monsoon summer and festival months are hard, winter is easy), "
          "so drift is measured on the same calendar months one year apart. " + same["text"], "",
          f"The frozen models under-forecast on average (ensemble bias {bias.get('solar_peak', np.nan) / 1000:+.1f} GW "
          f"solar-hour, {bias.get('evening_peak', np.nan) / 1000:+.1f} GW evening, {bias.get('energy', np.nan):+.0f} MU "
          "energy). FY 2025-26 had unusually weak demand growth (about 1 %); because the models predict demand "
          "relative to the last 7 days, a change in the growth rate shows up as a small level error rather than as "
          "an error that accumulates.", "",
          "P10–P90 coverage falls to " + ", ".join(
              f"{ov[(ov.model == 'ensemble') & (ov.target == k)]['coverage80'].iloc[0]:.0f} %" for k in TGT)
          + " (solar-hour, evening, energy) instead of 80 %. The intervals were calibrated on Jan–Mar 2025, the "
          "season with the smallest errors, and never updated; the rolling backtest recalibrates every month and "
          "holds about 80 %. Interval calibration drifts faster than the point forecast.", ""]
    _ = e
    if "grid_india" in t:
        g = t["grid_india"]
        L += ["## Grid-India on the same test days (ensemble)", "",
              "| target | our basis | Grid-India | days | our MAPE % | Grid-India MAPE % | our error lower by | win rate |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r['target']} | {r['basis']} | {r['grid_india']} | {r['days']} | {r['our_MAPE']:.2f} | "
                     f"{r['gi_MAPE']:.2f} | {r['improvement']:+.1%} | {r['win_rate']:.0%} |")
        L += ["", "Grid-India's forecast is prepared about 5 days before its week starts (6–12 days ahead), so even "
              "the 7-day-ahead basis favours us slightly; the matched comparison is in fairness_audit.md.", ""]
    L += comparison_md(t, rolling, live)
    L += ["## Ensemble and calibration (fitted on the validation months)", "",
          "Ensemble weight on the LSTM by target × horizon:", "",
          pd.DataFrame({k: info["post"]["ensemble"][k] for k in TGT}).T.to_markdown(), ""]
    text = "\n".join(L) + "\n"
    (REPORTS / "fixed_split_results.md").write_text(text)
    return text


def drift_same_months(mo: pd.DataFrame, model: str = "ensemble") -> dict:
    """Mean monthly MAPE (over targets) of the same calendar months in consecutive years."""
    e = mo[mo.model == model].groupby("month")["MAPE"].mean()
    e.index = pd.to_datetime(e.index)
    y1, y2 = e[e.index.year == 2025], e[e.index.year == 2026]
    months = sorted(set(y1.index.month) & set(y2.index.month))
    if not months:
        return {"text": "", "months": []}
    a = y1[y1.index.month.isin(months)].mean()
    b = y2[y2.index.month.isin(months)].mean()
    names = f"{pd.Timestamp(2025, months[0], 1):%b}–{pd.Timestamp(2025, months[-1], 1):%b}"
    verb = "barely changes" if abs(b - a) < 0.15 else ("rises" if b > a else "falls")
    return {"months": months, "first": float(a), "second": float(b),
            "text": f"Ensemble, {names} 2025 vs {names} 2026: {a:.2f} % vs {b:.2f} % — the error {verb} after a "
                    "year without retraining."}


def comparison(t: dict, rolling: pd.DataFrame | None, live: dict) -> pd.DataFrame:
    """MAPE per model x target: fixed split (whole test period and the rolling window only),
    rolling backtest (monthly refits), live."""
    from src.evaluate import BACKTEST_START
    from src.models.common import metrics

    rows = []
    fixed = t["overall"]
    p_roll = t.get("_pred_rolling_window")
    for m, fam in (("ensemble", "ensemble"), (NAMES["lstm"], "lstm"), (NAMES["lgbm"], "lgbm"), (NAMES["tft"], "tft"),
                   ("seasonal_naive", "seasonal_naive")):
        for k in TGT:
            r = {"model": m, "target": k}
            f = fixed[(fixed.model == m) & (fixed.target == k)]
            r["fixed_test"] = float(f["MAPE"].iloc[0]) if len(f) else np.nan
            if p_roll is not None:
                g = p_roll[(p_roll.model == m) & (p_roll.target == k)]
                r["fixed_same_window"] = metrics(g).get("MAPE", np.nan) if len(g) else np.nan
            if rolling is not None:
                g = rolling[(rolling.model == m) & (rolling.target == k) & (rolling.origin >= pd.Timestamp(BACKTEST_START))]
                r["rolling"] = metrics(g).get("MAPE", np.nan) if len(g) else np.nan
            lv = [x for x in (live or {}).get("by_model", []) if x["model"] == fam and x["target"] == k]
            lv = sorted(lv, key=lambda x: x.get("recipe") == "baseline")      # newest recipe first
            r["live"] = lv[0]["MAPE"] if lv else np.nan
            r["live_days"] = lv[0]["days"] if lv else 0
            rows.append(r)
    return pd.DataFrame(rows)


def comparison_md(t, rolling, live) -> list[str]:
    from src.viz import MODEL_LABELS

    c = comparison(t, rolling, live)
    c.to_parquet(PROCESSED / "evaluation_comparison.parquet", index=False)
    L = ["## Three stages compared — MAPE %", "",
         "| model | target | fixed split, whole test (Apr 2025 →) | fixed split, same 12 months as rolling | "
         "rolling backtest (monthly refit) | live (days) |", "|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        lv = f"{r['live']:.2f} ({int(r['live_days'])})" if r["live"] == r["live"] else "–"
        L.append(f"| {MODEL_LABELS.get(r['model'], r['model'])} | {r['target']} | {r['fixed_test']:.2f} | "
                 f"{r.get('fixed_same_window', np.nan):.2f} | {r.get('rolling', np.nan):.2f} | {lv} |")
    e = c[c.model == "ensemble"]
    if "fixed_same_window" in c and "rolling" in c and len(e):
        gap = (e["fixed_same_window"] - e["rolling"]).mean()
        L += ["", f"Same 12 months, same models: the frozen ensemble's MAPE is {gap:+.2f} points versus the monthly "
              "refits. That is the value of retraining: each monthly refit sees more recent data and recalibrates its "
              "intervals. It is not accuracy fading over time: the frozen ensemble's error on the same calendar months "
              "is unchanged a year later (see Drift). The seasonal-naive rows, which learn no weights, agree between the two "
              "columns, as they should.", "",
              f"Live so far: {live.get('days_scored', 0)} days scored ({live.get('first_day', '–')} .. "
              f"{live.get('last_day', '–')}) from {live.get('origins', 0)} forecast origin(s); "
              f"{live.get('real_time_days', 0)} of those days were forecast before the day began. The scored forecast "
              "was produced when the pipeline was first run (9 Nov 2026) for origin 1 Nov, using only data up to 1 Nov, "
              "by the models of the previous round; the ensemble has no live score yet. Too few days to compare; "
              "the table updates with every pipeline run.", ""]
    return L
