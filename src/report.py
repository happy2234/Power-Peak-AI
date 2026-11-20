"""reports/results.md: leaderboard, calibration, weather/season evaluation, before/after."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED, REPORTS, TARGETS

TLAB = {"solar_peak": "solar-hour peak", "evening_peak": "evening peak", "energy": "energy"}


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def coverage(d: pd.DataFrame, lo="p10", hi="p90") -> float:
    d = d.dropna(subset=["actual", lo, hi])
    return float(((d["actual"] >= d[lo]) & (d["actual"] <= d[hi])).mean() * 100) if len(d) else np.nan


def summary_table(ev: pd.DataFrame, models) -> pd.DataFrame:
    from src.models.common import metrics

    rows = []
    for m in models:
        g = ev[ev["model"] == m]
        if g.empty:
            continue
        r = {"model": m}
        for t in TARGETS:
            r[f"{TLAB[t]} MAPE %"] = metrics(g[g["target"] == t]).get("MAPE", np.nan)
        r["overall MAPE %"] = np.mean([r[f"{TLAB[t]} MAPE %"] for t in TARGETS])
        r["P10–P90 coverage %"] = coverage(g) if g["p10"].notna().any() else np.nan
        r["coverage before calibration %"] = coverage(g, "p10_raw", "p90_raw") if "p10_raw" in g and g["p10_raw"].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def _md(df: pd.DataFrame, nd=2) -> str:
    return df.round(nd).to_markdown()


def write_results(raw, processed, daily, best, selection, plog, rec, status, timing, v1,
                  compute: str, plain: pd.DataFrame | None = None) -> str:
    from src.evaluate import (REGIME_ORDER, SEASON_ORDER, SEASONS, by_horizon,
                              eval_window, head_to_head, heatwave_mask, leaderboard, mape_by,
                              mape_table, regime_thresholds, tag_days, weather_regimes)

    ev = eval_window(processed)
    ev_raw = eval_window(raw)
    lgbm_c = selection["ensemble_members"][1]
    lstm_c = selection["ensemble_members"][0]
    tft_c = selection.get("chosen", {}).get("tft", "tft")
    from src.evaluate import parse_variant

    order = {"seasonal_naive": 0, "lgbm": 1, "lstm": 2, "tft": 3, "ensemble": 4, "grid_india": 5}
    final = sorted(ev["model"].unique(), key=lambda m: (order.get(parse_variant(m)[0], 9), m))
    L = ["# Backtest results", "",
         f"Rolling-origin backtest, reported window {ev['date'].min():%Y-%m-%d} .. {ev['date'].max():%Y-%m-%d} "
         "(forecasts issued from 2025-11-01): models refit monthly on all data known before the month, daily "
         "forecast origins, 7 days ahead. Weather inputs are archived forecasts issued at the origin (no hindsight). "
         "Six earlier months (2025-05 .. 2025-10) were backtested the same way; their out-of-sample predictions "
         "are used only to fit the ensemble weights, the conformal calibration and the record-probability "
         "recalibration of the following months (each month uses only rows whose actuals were already published).",
         "", f"**Best model (lowest overall MAPE): `{best}`.** Peaks in MW, energy in MU.", ""]

    # ---- summary
    st = summary_table(ev, final)
    L += ["## Summary (12 reported months)", "", _md(st), "",
          "Coverage = share of days inside P10–P90 (target 80 %) after per-horizon conformal calibration; "
          "the last column is the model's own interval before that step. Grid-India publishes a point forecast only.", ""]

    # ---- model changes
    L += ["## Model changes and acceptance rule", "",
          "A change is kept only if the overall MAPE (mean over the three targets, 12 reported months) does "
          "not get worse. Each model family was backtested with three weather feature sets — `v1` (the previous "
          "round's inputs: temperature, humidity, rain, radiation, CDD), `compact` (v1 + night temperature, HDD, "
          "North-India fog, daytime cloud, dew point, heat index, gusts, 3-day rain and four anomalies) and "
          "`full` (all 81 weather columns) — each re-tuned with Optuna, with and without rare-weather sample "
          "weights (`_w`). The variant with the lowest overall MAPE is kept (ties: the simpler one).", ""]
    if v1 is not None:
        om_v1 = leaderboard(eval_window(v1)).pivot_table(index="model", columns="target", values="MAPE").mean(axis=1)
        L += ["Previous round (same 12 months): " + ", ".join(f"{m} {om_v1[m]:.3f}" for m in ("lgbm", "lstm", "tft")
                                                        if m in om_v1) + ".", ""]
    L += ["| family | variant | weather features | rare-weather weights | overall MAPE % | kept |",
          "|---|---|---|---|---|---|"]
    for d in selection["variants"]:
        L.append(f"| {d['family']} | {d['variant']} | {d['feature_set']} | {'yes' if d['regime_weights'] else 'no'} | "
                 f"{d['overall_MAPE']:.3f} | {'**yes**' if d['chosen'] else ''} |")
    om = selection["overall_mape"]
    bm = min(om[lgbm_c], om[lstm_c])
    L += ["", f"Ensemble of {lstm_c} + {lgbm_c}: overall MAPE {om['ensemble']:.3f} vs {bm:.3f} for the better member "
          f"→ {'kept' if om['ensemble'] <= bm else 'not kept'}.", ""]

    # ---- per target leaderboards
    lb = leaderboard(ev)
    for t in TARGETS:
        L += [f"## {t}", "", "| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in lb[lb["target"] == t].sort_values("MAPE").iterrows():
            L.append(f"| {r['model']} | {int(r['n'])} | {_f(r['MAPE'])} | {_f(r['MAE'], 0)} | {_f(r['RMSE'], 0)} | "
                     f"{_f(r['bias'], 0)} | {_f(r.get('pinball'), 0)} | {_f(r.get('coverage80'), 1)} |")
        L.append("")

    # ---- ensemble weights
    w = plog.dropna(subset=["ensemble"]).iloc[-1]["ensemble"] if plog["ensemble"].notna().any() else None
    if w:
        wt = pd.DataFrame({t: w[t] for t in TARGETS}).T
        L += [f"## Ensemble weights (weight on {lstm_c}; the rest on {lgbm_c}), last reported month", "",
              _md(wt), "", "Weights are re-learned every month on the previous 180 days of out-of-sample "
              "predictions (grid 0..1 in steps of 0.05, minimising MAPE of the P50).", ""]

    # ---- horizon tables
    L += ["## MAPE by horizon (days ahead)", "", _md(by_horizon(ev)), ""]
    cov_raw = by_horizon(ev.assign(p10=ev["p10_raw"], p90=ev["p90_raw"]), "coverage80")
    cov_cal = by_horizon(ev, "coverage80")
    from src.postprocess import ACI_GAMMA, CAL_WINDOW_DAYS

    L += ["## P10–P90 coverage by horizon: before → after conformal calibration (target 80 %)", "",
          f"Split-conformal per model × target × horizon on the previous {CAL_WINDOW_DAYS} days of residuals "
          f"(past only), with a monthly adaptive level (adaptive conformal inference, γ = {ACI_GAMMA}): each "
          "month the quantile level moves by γ × (0.2 − last month's miss rate), using only forecasts whose "
          "actuals were already published.", ""]
    keep = [m for m in (best, lstm_c, lgbm_c, tft_c) if m in ev["model"].unique()]
    keep = list(dict.fromkeys(keep))
    rows = []
    for t in TARGETS:
        for m in keep:
            if (t, m) in cov_raw.index:
                rows.append(pd.Series({f"h{h}": f"{cov_raw.loc[(t, m), h]:.1f} → {cov_cal.loc[(t, m), h]:.1f}"
                                       for h in cov_raw.columns}, name=f"{t} / {m}"))
    L += [pd.DataFrame(rows).to_markdown(), ""]
    cov = pd.concat([cov_raw.stack().rename("before"), cov_cal.stack().rename("after")], axis=1).reset_index()
    cov.columns = ["target", "model", "h", "before", "after"]
    cov.to_parquet(PROCESSED / "eval_coverage.parquet", index=False)
    spread = {}
    for lab, tbl in (("before", cov_raw), ("after", cov_cal)):
        sub = tbl.loc[[i for i in tbl.index if i[1] in keep]]
        spread[lab] = (sub.min().min(), sub.max().max(), (sub - 80).abs().mean().mean())
    if plain is not None:
        cp = by_horizon(plain, "coverage80")
        sub = cp.loc[[i for i in cp.index if i[1] in keep]]
        spread["plain"] = (sub.min().min(), sub.max().max(), (sub - 80).abs().mean().mean())
    L += [f"Range across horizons/targets for these models: before {spread['before'][0]:.1f}–{spread['before'][1]:.1f} % "
          f"(mean |coverage − 80| = {spread['before'][2]:.1f} pts); after {spread['after'][0]:.1f}–{spread['after'][1]:.1f} % "
          f"(mean |coverage − 80| = {spread['after'][2]:.1f} pts)."
          + (f" Plain split-conformal without the adaptive step: {spread['plain'][0]:.1f}–{spread['plain'][1]:.1f} % "
             f"(mean |coverage − 80| = {spread['plain'][2]:.1f} pts) — it over-covers because errors in the "
             "preceding months (summer, smaller training set) are larger than in the months being forecast. "
             f"The window ({CAL_WINDOW_DAYS} d) and γ were chosen by comparing a few settings on this backtest, "
             "so the 'after' coverage is slightly optimistic." if "plain" in spread else ""), ""]

    # ---- weather & season
    th = regime_thresholds(daily)
    regimes = weather_regimes(daily, th)
    seasons = pd.Series([SEASONS[m] for m in daily.index.month], index=daily.index)
    order_models = [m for m in ("grid_india", "seasonal_naive", lgbm_c, lstm_c, tft_c, "ensemble") if m in final]
    L += ["## MAPE by weather regime (observed weather; evaluation only)", "",
          "Each day gets one label, first match in this order: heavy rain/storm (all-India rain ≥ "
          f"{th['storm_precip_ai']:.1f} mm or gusts ≥ {th['storm_gust_ai']:.0f} km/h, top 5 %), cold/fog (Nov–Feb, "
          f"North-India night ≤ {th['cold_tmin_nr']:.0f} °C or fog proxy ≥ {th['fog_nr']}), hot (all-India max ≥ "
          f"{th['hot_tmax_ai']:.1f} °C or heat index ≥ {th['hot_heat_index_ai']:.1f} °C, top 10 %), rainy (≥ 2.5 mm), "
          "cloudy (daytime cloud ≥ 60 %), humid (dew point ≥ 22 °C), normal. `days` = distinct target days.", ""]
    by_reg = mape_by(ev[ev["model"].isin(order_models)], regimes, REGIME_ORDER)
    by_sea = mape_by(ev[ev["model"].isin(order_models)], seasons, SEASON_ORDER)
    for t in TARGETS:
        tb = mape_table(by_reg, order_models)
        L += [f"### {t}", "", _md(tb.xs(t, level="target")), ""]
    L += ["## MAPE by season", ""]
    for t in TARGETS:
        tb = mape_table(by_sea, order_models)
        L += [f"### {t}", "", _md(tb.xs(t, level="target")), ""]
    saved = [by_reg.assign(kind="regime", version="after"), by_sea.assign(kind="season", version="after")]

    # ---- before / after
    if v1 is not None:
        v1e = eval_window(v1)
        v1e = v1e[v1e["model"].isin(["lgbm", "lstm", "tft"])].assign(model=lambda d: d["model"] + " (before)")
        now = ev[ev["model"].isin([lgbm_c, lstm_c, tft_c, "ensemble"])].copy()
        now["model"] = now["model"] + " (after)"
        both = pd.concat([v1e, now], ignore_index=True)
        cols = ["lgbm (before)", f"{lgbm_c} (after)", "lstm (before)", f"{lstm_c} (after)", "tft (before)",
                f"{tft_c} (after)", "ensemble (after)"]
        hw = heatwave_mask(daily)
        lab2 = regimes.copy()
        L += ["## Before / after: previous models vs the all-weather models", "",
              "Same reported window and same day labels. *Before* = the models of the previous backtest "
              "(temperature, humidity, rain and radiation features only, no ensemble or per-horizon calibration).", ""]
        for t in TARGETS:
            tb = mape_table(mape_by(both, lab2, REGIME_ORDER), cols)
            hwt = mape_by(both, hw.map({True: "hottest 5 % of days", False: "other"}))
            hwt = mape_table(hwt, cols).xs(t, level="target").loc[["hottest 5 % of days"]]
            allt = mape_table(mape_by(both, pd.Series("all days", index=daily.index)), cols).xs(t, level="target")
            L += [f"### {t}", "", _md(pd.concat([allt, tb.xs(t, level="target"), hwt])), ""]
        saved += [mape_by(both, lab2, REGIME_ORDER).assign(kind="regime_before_after"),
                  mape_by(both, hw.map({True: "hottest 5 % of days", False: "other"})).assign(kind="hot5_before_after"),
                  mape_by(both, pd.Series("all days", index=daily.index)).assign(kind="all_before_after")]
        hot = mape_table(mape_by(both, hw.map({True: "hot5", False: "other"})), cols)
        L += ["### The heatwave weakness", ""]
        for t in ("solar_peak", "energy", "evening_peak"):
            r = hot.loc[(t, "hot5")]
            L.append(f"- {t}: before, LSTM {r['lstm (before)']:.2f} % vs LightGBM {r['lgbm (before)']:.2f} % on the "
                     f"hottest 5 % of days; after, {lstm_c} {r[f'{lstm_c} (after)']:.2f} % vs {lgbm_c} "
                     f"{r[f'{lgbm_c} (after)']:.2f} %, ensemble {r['ensemble (after)']:.2f} %.")
        L.append("")

    tbl = pd.concat(saved, ignore_index=True)
    tbl["label"] = tbl["label"].astype(str)
    tbl.to_parquet(PROCESSED / "eval_weather.parquet", index=False)
    regimes.rename("regime").to_frame().reset_index().to_parquet(PROCESSED / "weather_regimes.parquet", index=False)

    # ---- worst days
    b = ev[ev["model"] == best].dropna(subset=["actual"]).copy()
    b["ape"] = (b["p50"] - b["actual"]).abs() / b["actual"] * 100
    wd = b.groupby(["date", "target"])["ape"].mean().reset_index().sort_values("ape", ascending=False).head(15)
    tags = tag_days(wd["date"], daily, regimes).reset_index(drop=True)
    wd = pd.concat([wd.reset_index(drop=True).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")), tags], axis=1)
    L += [f"## Worst days for {best} (mean APE over the 7 forecasts of that day), with weather and holidays", "",
          wd.round(2).to_markdown(index=False), "",
          "`tmax_anom` = all-India max temperature minus its normal for the date; rain is demand-weighted all-India "
          "and the wettest region; all observed (reanalysis), for explanation only.", ""]

    # ---- record probability
    L += [f"## Record-peak probability: raw vs recalibrated (model {best}, {status['origins']} origins)", "",
          "Recalibration maps are fitted month by month on earlier origins whose 7-day outcome was already "
          "known (Platt = logistic on logit(p); isotonic = monotone step function).", "",
          "| record | method | events | Brier | climatology | skill |", "|---|---|---|---|---|---|"]
    for kind, res in rec.items():
        for m, r in res.items():
            L.append(f"| {kind.replace('_', '-')} | {m} | {r['events']} | {r['brier']:.4f} | "
                     f"{r['brier_climatology']:.4f} | {r['skill']:+.3f} |")
    L += ["", f"Live gauges use Platt scaling. Status: **{'Experimental — not yet reliable' if status['experimental'] else 'reliable'}** "
          f"(skill must exceed {status['threshold']} for both records)." , ""]

    # ---- head-to-head (original basis; see the fairness audit)
    L += ["## Head-to-head vs Grid-India (original basis)", "",
          "Our mean of the 1–7-day-ahead forecasts for each day vs Grid-India's value for that day. This basis "
          "favours our model (Grid-India's forecast is prepared ~5 days before its week starts); see "
          "[fairness_audit.md](fairness_audit.md) for the matched comparison.", "",
          "| target | days | our MAE | Grid-India MAE | improvement | win rate |", "|---|---|---|---|---|---|"]
    j = head_to_head(ev, best)
    for t, g in j.groupby("target"):
        L.append(f"| {t} | {len(g)} | {g['ae_ours'].mean():,.0f} | {g['ae_gi'].mean():,.0f} | "
                 f"{1 - g['ae_ours'].mean() / g['ae_gi'].mean():+.1%} | {g['ours_wins'].mean():.0%} |")
    L.append("")

    # ---- compute
    L += ["## Compute", "", f"Device: {compute}. Backtest wall time (18 monthly refits incl. warm-up): "
          + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timing.items()), ""]
    text = "\n".join(L) + "\n"
    (REPORTS / "results.md").write_text(text)
    return text
