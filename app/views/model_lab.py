"""Model lab: leaderboard, errors and coverage by horizon, accuracy by weather, calibration."""
import common
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.config import TARGET_LABELS
from src.evaluate import REGIME_ORDER, SEASON_ORDER
from src.insights import brier, reliability
from src.viz import (BACKTEST_SOURCE, COLORS, MODEL_LABELS, NEUTRAL, actual_trace, band_traces, coverage_before_after, grid_india_trace,
                     horizon_lines, main_models, mape_bars, model_color, our_models, reliability_plot, scale, style, training_curve)

ui.page("Model lab", "Every model's accuracy by target, horizon and weather, how well its ranges hold, and how "
                     "the models are evaluated." if ui.full_mode() else
                     "How accurate each model is, and how the models are evaluated.", badges=[ui.badge("12-month backtest"), ui.updated_badge()])
bt = common.backtest()
if bt is None:
    ui.empty_state("No backtest results yet. Run the backtest (about an hour on a GPU):", "backtest")
    ui.stop()
best = common.backtest_best()
models = our_models(bt["model"].unique())
MB = True   # this page keeps the Plotly mode bar (zoom, download)

ALL_TABS = ["Leaderboard", "By horizon", "By weather", "Calibration", "Evaluation design", "Drift"]
# presentation mode (ui.app_mode) shows two of the six; the other four are built only in full mode
shown = ALL_TABS if ui.full_mode() else [t for t in ALL_TABS if t in ui.PRESENTATION_LAB_TABS]
tabs = dict(zip(shown, st.tabs(shown)))

# ----------------------------------------------------------------------------- leaderboard
if "Leaderboard" in tabs:
    with tabs["Leaderboard"]:
        lb = common.leaderboard()
        wide = lb.pivot_table(index="model", columns="target", values="MAPE")
        wide["overall"] = wide[list(TARGET_LABELS)].mean(axis=1)
        cov = lb.pivot_table(index="model", values="coverage80", aggfunc="mean") if "coverage80" in lb else None
        wide = wide.sort_values("overall")
        top, gi_o = wide.index[0], wide["overall"].get("grid_india")
        ui.headline(f"*{MODEL_LABELS.get(top, top)}* has the lowest error over the 12 reported months: "
                    f"{wide.loc[top, 'overall']:.2f} % MAPE on average"
                    + (f", against {gi_o:.2f} % for Grid-India." if gi_o == gi_o and gi_o is not None else "."))
        metrics = lb.assign(model_name=lb["model"].map(lambda m: MODEL_LABELS.get(m, m)),
                            target_name=lb["target"].map(TARGET_LABELS))
        ui.download_csv(metrics, "model-metrics", label="Download metrics", primary=True)
        tbl = pd.DataFrame({"Model": [MODEL_LABELS.get(m, m) for m in wide.index],
                            "Solar-hour peak MAPE %": wide["solar_peak"].to_numpy(),
                            "Evening peak MAPE %": wide["evening_peak"].to_numpy(),
                            "Energy MAPE %": wide["energy"].to_numpy(),
                            "Overall MAPE %": wide["overall"].to_numpy(),
                            "P10–P90 coverage %": (cov.reindex(wide.index)["coverage80"].to_numpy()
                                                   if cov is not None else np.nan)})
        main = set(main_models(lb)) | {"grid_india", "seasonal_naive"}
        ui.table(tbl[[m in main for m in wide.index]], max_height=None, name="leaderboard")
        with st.expander(f"All {len(tbl)} variants"):
            ui.table(tbl, max_height=None, name="leaderboard-all")
        st.caption("Forecasts issued from 1 Nov 2025, 1–7 days ahead, scored against published actuals. Grid-India publishes "
                   "a single value, so it has no coverage. The ensemble blends LSTM and LightGBM with weights per target and "
                   "horizon, re-learned each month from earlier months only.")
        ui.terms("MAPE", "coverage")
        sel = common.model_selection()
        if sel and sel.get("variants") and "variant" in sel["variants"][0]:
            with st.expander("Which variants were kept (a change stays only if overall MAPE does not get worse)"):
                ui.table(pd.DataFrame([{"Model": MODEL_LABELS.get(v["variant"], v["variant"]),
                                        "Weather features": v["feature_set"],
                                        "Rare-weather weights": "yes" if v["regime_weights"] else "no",
                                        "Overall MAPE %": v["overall_MAPE"], "Kept": "yes" if v["chosen"] else ""}
                                       for v in sel["variants"]]), max_height=None)
                st.caption("v1 = the previous round's weather inputs; compact = v1 plus the most relevant cold/fog, rain, "
                           "cloud, humidity, wind and anomaly inputs; full = all 81 weather columns.")

        with st.expander("Actual vs forecast, day by day"):
            c1, c2, c3 = st.columns([1.9, 1.1, 0.9])
            target = c1.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                                          key="ml_target", label_visibility="collapsed") or "evening_peak"
            m = c2.selectbox("Model", models, index=models.index(best) if best in models else 0,
                             format_func=lambda x: MODEL_LABELS.get(x, x), label_visibility="collapsed")
            h = c3.slider("Days ahead", 1, 7, 1)
            g = bt[(bt["model"] == m) & (bt["target"] == target) & (bt["h"] == h)].sort_values("date")
            div, unit = scale(target)
            fig = go.Figure()
            fig.add_trace(band_traces(g["date"], g["p10"] / div, g["p90"] / div))
            fig.add_trace(go.Scatter(x=g["date"], y=g["p50"] / div, name=MODEL_LABELS.get(m, m),
                                     line=dict(color=COLORS["ours"], width=1.5)))
            fig.add_trace(actual_trace(g["date"], g["actual"] / div, unit))
            gi = bt[(bt["model"] == "grid_india") & (bt["target"] == target)]
            if len(gi):
                fig.add_trace(grid_india_trace(gi["date"], gi["p50"] / div, unit, line=False))
            mape = ((g["p50"] - g["actual"]).abs() / g["actual"]).mean() * 100
            ui.chart(style(fig, f"{h}-day-ahead {TARGET_LABELS[target].lower()} tracks the actuals within {mape:.1f} % on average",
                           420, unit, legend="direct", subtitle=f"{unit}; backtest forecasts with P10–P90 range", source=BACKTEST_SOURCE), modebar=MB)

        with st.expander("Training history"):
            curves = common.training_curves()
            latest = {}
            for name in sorted(curves):
                latest[name.split("_")[0]] = name
            cols = st.columns(max(len(latest), 1))
            for c, (kind, name) in zip(cols, latest.items()):
                with c:
                    ui.chart(training_curve(curves[name], f"{kind.upper()}",
                                            subtitle=f"Training and validation loss, run {name.split('_', 1)[1]}"),
                             modebar=MB, half=True)
            ui.terms("pinball loss")
            log = common.training_log()
            if log is not None:
                cols = [c for c in ("model", "feature_set", "regime_boost", "epochs", "train_seconds", "device", "at")
                        if c in log]
                ui.table(log[cols].tail(8))

# ----------------------------------------------------------------------------- by horizon
if "By horizon" in tabs:
    with tabs["By horizon"]:
        target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                                      key="hz_target", label_visibility="collapsed") or "evening_peak"
        show = main_models(common.leaderboard())
        by = bt[bt["model"] == best]
        m1 = 100 * ((by[(by.target == target) & (by.h == 1)].p50 - by[(by.target == target) & (by.h == 1)].actual).abs()
                    / by[(by.target == target) & (by.h == 1)].actual).mean()
        m7 = 100 * ((by[(by.target == target) & (by.h == 7)].p50 - by[(by.target == target) & (by.h == 7)].actual).abs()
                    / by[(by.target == target) & (by.h == 7)].actual).mean()
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.chart(horizon_lines(bt, target, lambda g: 100 * ((g.p50 - g.actual).abs() / g.actual).mean(), "MAPE (%)",
                                   f"Error grows from {m1:.1f} % one day ahead to {m7:.1f} % at seven days",
                                   best=best, models=show, subtitle=f"MAPE %, {TARGET_LABELS[target].lower()}, best variant "
                                                                    "of each model; all variants in the Leaderboard"), modebar=MB,
                     half=True)
        with c2:
            bb = by[by.target == target]
            cv = bb.groupby("h").apply(lambda g: 100 * ((g.actual >= g.p10) & (g.actual <= g.p90)).mean())
            ui.chart(horizon_lines(bt, target, lambda g: 100 * ((g.actual >= g.p10) & (g.actual <= g.p90)).mean(),
                                   "% inside P10–P90", f"The P10–P90 range holds {cv.min():.0f}–{cv.max():.0f} % of days "
                                                       "at every horizon (target 80 %)",
                                   ref_line=80, best=best, models=show, subtitle="Share of days inside P10–P90, %"),
                     modebar=MB, half=True)
        ui.terms("MAPE", "P10–P90")

# ----------------------------------------------------------------------------- by weather
if "By weather" in tabs:
    with tabs["By weather"]:
        ev = common.eval_weather()
        if ev is None:
            ui.empty_state("Accuracy by weather comes from the backtest. Run it:", "backtest")
        else:
            target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get,
                                          default="evening_peak", key="wx_target", label_visibility="collapsed") or "evening_peak"
            t = ev[(ev["kind"] == "regime") & (ev["target"] == target)]
            ms = ["grid_india"] + [x for x in our_models(t["model"].unique()) if x != "seasonal_naive"]
            order = [o for o in REGIME_ORDER if o in set(t["label"])]
            b = t[t["model"] == best].set_index("label")["MAPE"]
            worst = b.idxmax() if len(b) else ""
            ui.chart(mape_bars(t, ms, order, f"Hardest conditions for {common.model_label(best)}: {worst} days",
                               best=best, subtitle="MAPE %, days labelled by observed weather (evaluation only)"),
                     modebar=MB)
            parts = []
            for kind in ("all_before_after", "regime_before_after", "hot5_before_after"):
                x = ev[(ev["kind"] == kind) & (ev["target"] == target)]
                if kind == "hot5_before_after":
                    x = x[x["label"] == "hottest 5 % of days"]
                parts.append(x)
            x = pd.concat(parts)
            if len(x):
                st.subheader("Before and after the all-weather features")
                orderx = [o for o in ["all days"] + REGIME_ORDER + ["hottest 5 % of days"] if o in set(x["label"])]
                piv = x.pivot_table(index="label", columns="model", values="MAPE").reindex(orderx)

                def _key(c):
                    name, when = c.rsplit(" (", 1)
                    fam = name.partition("@")[0].removesuffix("_w")
                    return ({"lgbm": 0, "lstm": 1, "tft": 2, "ensemble": 3}.get(fam, 9), when != "before)")
                cols = sorted(piv.columns, key=_key)
                days = x.drop_duplicates("label").set_index("label")["days"]
                out = pd.DataFrame({"Days by observed weather": piv.index, "Days": days.reindex(piv.index).to_numpy()})
                for c in cols:
                    name, when = c.rsplit(" (", 1)
                    fam = MODEL_LABELS.get(name.partition("@")[0].removesuffix("_w"), name)
                    out[f"{fam} {when.rstrip(')')}"] = piv[c].to_numpy()
                ui.table(out, max_height=None)
                st.caption("MAPE % over the same 12 months. Before = the previous models (temperature, humidity, rain and "
                           "radiation only, no ensemble). After = the variant of each model kept in this round, as "
                           "named in the Leaderboard.")
            s = ev[(ev["kind"] == "season") & (ev["target"] == target)]
            if len(s):
                sb = s[s["model"] == best].set_index("label")["MAPE"]
                hard = sb.idxmax().split(" (")[0] if len(sb) else "–"
                ui.chart(mape_bars(s, ms, [o for o in SEASON_ORDER if o in set(s["label"])],
                                   f"{hard.capitalize()} is the hardest season for {common.model_label(best)}", best=best,
                                   subtitle="MAPE % by season"), modebar=MB)

# ----------------------------------------------------------------------------- calibration
if "Calibration" in tabs:
    with tabs["Calibration"]:
        target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                                      key="cal_target", label_visibility="collapsed") or "evening_peak"
        cov = common.eval_coverage()
        if cov is None:
            ui.empty_state("Coverage tables come from the backtest. Run it:", "backtest")
        else:
            avail = [x for x in our_models(cov["model"].unique()) if x != "seasonal_naive"]
            cm = st.selectbox("Model", avail, index=avail.index(best) if best in avail else 0,
                              format_func=lambda x: MODEL_LABELS.get(x, x), key="cov_model", label_visibility="collapsed")
            g = cov[(cov["target"] == target) & (cov["model"] == cm)]
            ui.chart(coverage_before_after(cov, target, cm,
                                           f"Per-horizon calibration brings coverage from {g['before'].min():.0f}–"
                                           f"{g['before'].max():.0f} % to {g['after'].min():.0f}–{g['after'].max():.0f} %",
                                           subtitle="Share of days inside P10–P90 by days ahead, %; target 80 %"), modebar=MB)
            st.caption("Before: the model's own range was too wide one day ahead and too narrow six to seven days ahead. "
                       "After: each range is widened or narrowed per target and horizon using the errors of the previous "
                       "90 days, re-tuned monthly, using past data only.")
            ui.terms("P10–P90", "coverage")

        st.subheader("Record-peak probability")
        cal = common.record_calibration()
        if cal is None or cal.empty:
            ui.empty_state("Calibration results come from the backtest. Run it:", "backtest")
        else:
            cal = cal[cal["origin"] >= pd.Timestamp("2025-11-01")]
            ver = st.segmented_control("Probabilities", ["Calibrated", "Raw"], default="Calibrated",
                                       label_visibility="collapsed",
                                       help="Raw: simulated from the forecast ranges. Calibrated: corrected with a logistic "
                                            "fit to earlier outcomes (Platt scaling).")
            sfx = "_platt" if ver != "Raw" and "p_all_time_platt" in cal else ""
            c1, c2 = st.columns(2, gap="large")
            for col, kind, lab in [(c1, "all_time", "all-time peak"), (c2, "year", "the year's peak")]:
                rel = reliability(cal[f"p_{kind}{sfx}"], cal[f"hit_{kind}"])
                b = brier(cal[f"p_{kind}{sfx}"], cal[f"hit_{kind}"])
                with col:
                    ui.chart(reliability_plot(rel, f"Exceeding {lab} within 7 days: skill {b['skill']:+.2f}",
                                              subtitle=f"{b['n']} origins, {b['events']} events; Brier {b['brier']:.3f} vs "
                                                       f"{b['brier_climatology']:.3f} for the base rate"), modebar=MB,
                             half=True)
            meta = common.latest_meta() or {}
            if (meta.get("record_status") or {}).get("experimental", True):
                ui.badges(ui.badge("Experimental", "warn"))
            st.caption("Record days are rare, so these estimates are noisy. The Overview labels the probabilities "
                       "experimental until skill exceeds 0.05 for both records.")
            ui.terms("Brier skill")

# ----------------------------------------------------------------------------- drift
if "Drift" in tabs:
    with tabs["Drift"]:
        fxt = common.fixed_split_tables()
        if fxt is None:
            ui.empty_state("No fixed train/test results yet. Run the fixed-split evaluation:", "fixed_split")
        else:
            target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get,
                                          default="evening_peak", key="dr_target", label_visibility="collapsed") or "evening_peak"
            mo = fxt[(fxt["kind"] == "month") & (fxt["target"] == target)].copy()
            mo["month"] = pd.to_datetime(mo["month"])
            show = [m for m in our_models(mo["model"].unique())]
            fig = go.Figure()
            for m in show:
                g = mo[mo["model"] == m].sort_values("month")
                fig.add_trace(go.Scatter(x=g["month"], y=g["MAPE"], name=MODEL_LABELS.get(m, m), mode="lines+markers",
                                         line=dict(color=model_color(m, "ensemble"), width=2.5 if m == "ensemble" else 1.5),
                                         marker=dict(size=5)))
            r = bt[(bt["model"] == "ensemble") & (bt["target"] == target)].dropna(subset=["actual"]).copy()
            if len(r):
                r["month"] = r["date"].dt.to_period("M").dt.to_timestamp()
                rm = r.groupby("month").apply(lambda g: 100 * ((g.p50 - g.actual).abs() / g.actual).mean())
                fig.add_trace(go.Scatter(x=rm.index, y=rm.values, name="Ensemble, refit monthly", mode="lines",
                                         line=dict(color=NEUTRAL["strong"], width=2, dash="dot")))
            from src.fixed_split import drift_same_months

            dm = drift_same_months(mo)
            if dm.get("months"):
                verb = ("rises" if dm["second"] > dm["first"] + 0.15 else "falls" if dm["second"] < dm["first"] - 0.15
                        else "stays level")
                ttl = (f"Frozen since March 2025, the ensemble's error {verb} a year later: {dm['first']:.2f} % vs "
                       f"{dm['second']:.2f} % over the same months")
            else:
                ttl = "Monthly error of the frozen models"
            ui.chart(style(fig, ttl, 420, "MAPE (%)", legend=True,
                           subtitle=f"Monthly MAPE %, {TARGET_LABELS[target].lower()}, models trained once on FY 2023-24 "
                                    "and FY 2024-25; dotted: the same ensemble refit every month"), modebar=MB)
            from src.fixed_split import drift_same_months

            st.caption("Drift = how accuracy changes when a model is not retrained. Seasons dominate the monthly pattern, "
                       "so compare the same months a year apart. " + drift_same_months(
                           fxt[fxt["kind"] == "month"].assign(month=lambda x: pd.to_datetime(x["month"])))["text"]
                       + " FY 2025-26 had unusually weak demand growth (about 1 %).")
            fyt = fxt[(fxt["kind"] == "fy") & (fxt["target"] == target)]
            if len(fyt):
                piv = fyt.pivot_table(index="model", columns="fy", values="MAPE").reindex(show)
                piv.index = [MODEL_LABELS.get(m, m) for m in piv.index]
                piv.index.name = "Model"
                ui.kicker("MAPE % by financial year")
                ui.table(piv, max_height=None)

# ----------------------------------------------------------------------------- evaluation design
if "Evaluation design" in tabs:
    with tabs["Evaluation design"]:
        info = common.fixed_split_info() or {}
        live = common.live_summary() or {}
        st.subheader("Three stages of evaluation")
        last_day = (info.get("test") or ["", str(bt["date"].max().date())])[1]
        ui.timeline([("train", "Train", "FY 2023-24 + Apr–Dec 2024", "2023-04-01", "2024-12-31"),
                     ("val", "Validation", "Jan–Mar 2025", "2025-01-01", "2025-03-31"),
                     ("test", "Test (models frozen)", "FY 2025-26 and FY 2026-27", "2025-04-01", "2026-11-01"),
                     ("live", "Live", "since 02 Nov", "2026-11-02", max(last_day, "2026-11-08"))],
                    ticks=["2023-04-01", "2025-04-01", "2026-11-02"])
        ui.table(pd.DataFrame([
            {"Stage": "1–2. Fixed split", "Question it answers": "How good is a model trained once, on unseen future data?",
             "Training": "Once, on Apr 2023 – Dec 2024 (Jan–Mar 2025 for early stopping and calibration)",
             "Tested on": "Every day from 1 Apr 2025, 1–7 days ahead"},
            {"Stage": "Rolling backtest", "Question it answers": "How good is the system as it is operated, retrained monthly?",
             "Training": "Refit every month on all data known before that month",
             "Tested on": "Forecasts issued 1 Nov 2025 – 8 Nov 2026"},
            {"Stage": "3. Live", "Question it answers": "Does it hold up in real time?",
             "Training": "The deployed models, updated by the daily pipeline",
             "Tested on": f"{live.get('days_scored', 0)} days scored so far, {live.get('real_time_days', 0)} of them "
                          "forecast before the day began"}]), max_height=None)
        comp = common.evaluation_comparison()
        if comp is None:
            ui.empty_state("The comparison needs the fixed-split evaluation. Run it:", "fixed_split")
        else:
            e = comp[comp["model"] == "ensemble"]
            gap = (e["fixed_same_window"] - e["rolling"]).mean() if "fixed_same_window" in e else float("nan")
            if gap != gap:
                ui.headline("Fixed split, rolling backtest and live compared.")
            elif gap > 0.02:
                ui.headline(f"Not retraining costs the ensemble *{gap:.2f} MAPE points* over the same 12 months.")
            else:
                ui.headline(f"Over the same 12 months the frozen ensemble is within *{abs(gap):.2f} MAPE points* of the "
                            "monthly-refit one: little drift so far.")
            out = pd.DataFrame({"Model": [MODEL_LABELS.get(m, m) for m in comp["model"]],
                                "Target": comp["target"].map({"solar_peak": "Solar-hour", "evening_peak": "Evening",
                                                              "energy": "Energy"}),
                                "Fixed, all test": comp["fixed_test"],
                                "Fixed, same 12 months": comp.get("fixed_same_window"),
                                "Refit monthly": comp.get("rolling"),
                                "Live": comp["live"], "Live days": comp["live_days"]})
            ui.kicker("MAPE % by evaluation")
            ui.table(out, max_height=None, name="evaluation-comparison")
            from src.fixed_split import drift_same_months

            fxt2 = common.fixed_split_tables()
            same = drift_same_months(fxt2[fxt2["kind"] == "month"].assign(month=lambda x: pd.to_datetime(x["month"]))) \
                if fxt2 is not None else {"text": ""}
            st.caption("Same model designs and inputs in every column. Fixed split: trained once, never refit. Rolling: "
                       "refit every month. The gap between the two middle columns is the value of retraining: more data and "
                       "monthly recalibration. It is not accuracy fading over time: " + same["text"] + " Live: forecasts "
                       f"saved by the daily pipeline and scored once actuals are published; so far {live.get('days_scored', 0)} "
                       "days from one back-filled origin, made by the previous round's models, too few to compare.")
ui.footer()
