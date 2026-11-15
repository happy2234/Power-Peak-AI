"""Our forecasts vs Grid-India's week-ahead forecast: live, backtest, and bias-corrected/matched."""
import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import fmt
from src.config import TARGET_LABELS
from src.viz import BACKTEST_SOURCE, COLORS, MODEL_LABELS, REF, grid_india_trace, our_models, scale, style

ui.page("Our forecast vs Grid-India", "How our forecast compares with Grid-India's week-ahead forecast: live, over 12 months "
                        "of backtest, and on a matched, bias-corrected basis.", badges=[ui.updated_badge()])


def win_rate_card():
    """The 'Win rate vs Grid-India' card that used to sit on the Overview: last 30 days of the backtest."""
    bt = common.backtest()
    if bt is None:
        return
    acc = common.recent_accuracy(common.backtest_best())
    if acc and acc.get("win") == acc.get("win"):
        d = acc["win"] - acc["win_prev"] if acc.get("win_prev") == acc.get("win_prev") else None
        card, _ = st.columns([1, 2])
        with card:
            ui.kpis([{"label": "Win rate vs Grid-India", "value": f"{acc['win']:.0f}", "unit": "%", "help": "win rate",
                      **({"delta": f"{d:+.0f} pts", "good": d >= 0 if abs(d) >= 0.5 else None} if d is not None else {}),
                      "note": "last 30 days vs the 30 before"}])


def live_tab():
    lv = common.live_scorecard()
    if lv is None or lv.empty:
        ui.empty_state("No live days scored yet. Each daily run scores the days whose actuals have arrived:", "pipeline")
        return
    lv = lv.sort_values("origin").groupby(["date", "target"], as_index=False).last()
    lv["recipe"] = lv["recipe"].fillna("baseline") if "recipe" in lv else "baseline"
    recipes = sorted(lv["recipe"].unique(), key=lambda r: r != "baseline")
    changes = common.model_changes() or []
    if changes:
        c = changes[-1]
        ui.note(f"Model recipe switched from {c['from']} to {c['to']} on {fmt.date(c['date'])}. Rows below are split "
                "by the recipe that issued each forecast, so results before and after the switch stay separate.")
    rows = []
    for (rc, t), g in lv.groupby(["recipe", "target"]):
        gi = g.dropna(subset=["ae_gi"])
        rows.append({**({"Recipe": rc} if len(recipes) > 1 else {}), "Target": TARGET_LABELS[t], "Days scored": len(g),
                     "Our mean absolute error": fmt.value(g["ae_ours"].mean(), t),
                     "Grid-India mean absolute error": fmt.value(gi["ae_gi"].mean(), t) if len(gi) else "–",
                     "Days our model was closer": f"{(gi['ae_ours'] < gi['ae_gi']).mean():.0%}" if len(gi) else "–"})
    ui.table(pd.DataFrame(rows), max_height=None)
    recent = lv.sort_values(["date", "target"], ascending=[False, True]).head(21)
    with st.expander("Most recent scored days"):
        _recent_table(recent)
    st.caption("For each day, the most recent of our forecasts issued before it is scored; Grid-India's is its "
               "week-ahead forecast.")


def _recent_table(recent):
    ui.table(pd.DataFrame({
        "Date": recent["date"], "Target": recent["target"].map(TARGET_LABELS), "Recipe": recent["recipe"],
        "Days ahead": recent["h"].astype(int),
        "Actual": [fmt.value(v, t) for v, t in zip(recent["actual"], recent["target"])],
        "Our model": [fmt.value(v, t) for v, t in zip(recent["p50_ours"], recent["target"])],
        "Grid-India": [fmt.value(v, t) for v, t in zip(recent["p50_gi"], recent["target"])],
        "Our error": [fmt.value(v, t) for v, t in zip(recent["ae_ours"], recent["target"])],
        "Grid-India error": [fmt.value(v, t) for v, t in zip(recent["ae_gi"], recent["target"])]}))


def scorecard_view(bt, model, target, corrected: bool, key: str):
    n = bt.loc[(bt["model"] == "grid_india") & (bt["target"] == target), "date"].nunique()
    with st.spinner(f"Scoring {n} backtest days…"):
        sc = common.scorecard(model, target, corrected)
    if sc.empty:
        ui.empty_state("No days to compare for this model and target. Pick another model.", None)
        return
    div, unit = scale(target)
    imp = 1 - sc["ae_ours"].mean() / sc["ae_gi"].mean()
    gi_name = "Grid-India (bias-corrected)" if corrected else "Grid-India"
    ui.headline(f"{common.model_label(model)} was closer than {gi_name} on *{sc['ours_wins'].mean():.0%} of "
                f"{len(sc)} days*, with a {abs(imp):.0%} {'lower' if imp >= 0 else 'higher'} mean absolute error.")
    ui.kpis([{"label": "Win rate", "value": f"{100 * sc['ours_wins'].mean():.0f}", "unit": "%", "help": "win rate"},
             {"label": "Our mean absolute error", "value": fmt.value(sc["ae_ours"].mean(), target, unit=False), "unit": unit},
             {"label": f"{gi_name} mean absolute error", "value": fmt.value(sc["ae_gi"].mean(), target, unit=False),
              "unit": unit}])
    ours = common.model_label(model)
    t_cum, t_err = st.tabs(["Cumulative error", "Daily error"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["cum_ae_ours"] / div, name=ours, line=dict(color=COLORS["ours"], width=2.5)))
    fig.add_trace(grid_india_trace(sc["date"], sc["cum_ae_gi"] / div, unit, name=gi_name).update(mode="lines"))
    with t_cum:
        ui.chart(style(fig, f"Over the period, {gi_name} accumulates "
                          f"{sc['cum_ae_gi'].iloc[-1] / max(sc['cum_ae_ours'].iloc[-1], 1e-9):.1f}× our total error",
                     340, unit, legend="direct", subtitle=f"Cumulative absolute error, {unit} (lower is better). "
                                                          "Source: Grid-India (NLDC), own backtest"), key=f"cum_{key}")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sc["date"], y=(sc["p50_ours"] - sc["actual"]) / div, name=ours,
                         marker=dict(color=COLORS["ours"])))
    fig.add_trace(grid_india_trace(sc["date"], (sc["p50_gi"] - sc["actual"]) / div, unit, name=gi_name, line=False))
    fig.add_hline(y=0, line=dict(color=REF, width=1))
    bias_gi = (sc["p50_gi"] - sc["actual"]).mean() / div
    with t_err:
        ui.chart(style(fig, f"{gi_name} errs {'high' if bias_gi > 0 else 'low'} on average ({bias_gi:+.1f} {unit})",
                     300, unit, subtitle=f"Daily error, forecast minus actual, {unit}", source=BACKTEST_SOURCE), key=f"err_{key}")
    tbl = sc[["date", "actual", "p50_ours", "p50_gi", "advantage"]].copy()
    for c in tbl.columns[1:]:
        tbl[c] = tbl[c] / div
    names = {"date": "Date", "actual": f"Actual ({unit})", "p50_ours": f"Ours ({unit})",
             "p50_gi": f"Grid-India ({unit})", "advantage": f"Advantage ({unit})"}
    with st.expander("Best and worst days against Grid-India"):
        ui.kicker("Our five best days against Grid-India")
        ui.table(tbl.nlargest(5, "advantage").rename(columns=names), max_height=None, name=f"best-days-{key}")
        ui.kicker("Grid-India's five best days against us")
        ui.table(tbl.nsmallest(5, "advantage").rename(columns=names), max_height=None, name=f"worst-days-{key}")


win_rate_card()
bt = common.backtest()
tabs = st.tabs(["Live scorecard", "Backtest", "Bias-corrected"])
with tabs[0]:
    live_tab()
if bt is None or "grid_india" not in set(bt["model"]):
    for t in tabs[1:]:
        with t:
            ui.empty_state("The head-to-head comes from the backtest. Run it:", "backtest")
    ui.stop()

models = our_models(bt["model"].unique())
best = common.backtest_best()
with tabs[1]:
    c1, c2 = st.columns([1.4, 1])
    target = c1.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                                  key="sc_target", label_visibility="collapsed") or "evening_peak"
    model = c2.selectbox("Our model", models, index=models.index(best) if best in models else 0,
                         format_func=lambda m: MODEL_LABELS.get(m, m), key="sc_model", label_visibility="collapsed")
    scorecard_view(bt, model, target, corrected=False, key="raw")
    st.caption("Our forecast for a day = the mean of our 1–7-day-ahead backtest forecasts for that day; Grid-India's "
               "week-ahead forecast is prepared about 5 days before its week starts. This basis favours our model — "
               "the Bias-corrected tab shows the matched comparison.")

with tabs[2]:
    c1, c2 = st.columns([1.4, 1])
    target2 = c1.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                                   key="sc_target2", label_visibility="collapsed") or "evening_peak"
    model2 = c2.selectbox("Our model", models, index=models.index(best) if best in models else 0,
                          format_func=lambda m: MODEL_LABELS.get(m, m), key="sc_model2", label_visibility="collapsed")
    fa = common.fairness_audit()
    if fa is not None:
        labels = {"0 original": "Original basis (mean of our 1–7-day forecasts)",
                  "1 single h=7": "One forecast per day (our 7-day-ahead)",
                  "2 matched issue (lgbm_long)": "Matched issue time and lead",
                  "4 matched + GI bias-corrected": "Matched + Grid-India bias removed"}
        rows = []
        for key, lab in labels.items():
            for r in fa["steps"].get(key, []):
                if r["target"] == target2:
                    rows.append({"Comparison": lab, "Days": r["days"], "Our MAPE %": r["our_MAPE"],
                                 "Grid-India MAPE %": r["gi_MAPE"], "Our error lower by %": 100 * r["improvement"],
                                 "Days we were closer %": 100 * r["win_rate"]})
        if rows:
            last = rows[-1]
            ui.headline(f"On a matched, bias-corrected basis our error is *{last['Our error lower by %']:.0f} % lower* "
                        f"than Grid-India's for the {TARGET_LABELS[target2].lower()} "
                        f"(originally {rows[0]['Our error lower by %']:.0f} %).")
            ui.table(pd.DataFrame(rows), max_height=None)
            st.caption("Grid-India prepares its forecast about 5 days before the week starts, so it forecasts 6–12 days "
                       "ahead. Matched = our long-lead LightGBM issued on the same day with the same information; "
                       "bias removed = Grid-India's average over-forecast, measured on earlier weeks only, subtracted. "
                       "Details: reports/fairness_audit.md.")
    else:
        ui.empty_state("No matched comparison yet. Run the fairness audit:", "audit")
    ui.kicker("Day-by-day, Grid-India bias-corrected")
    try:
        scorecard_view(bt, model2, target2, corrected=True, key="corr")
    except Exception as e:  # listing index missing
        ui.empty_state(f"Bias correction needs the Grid-India listing index ({type(e).__name__}). Back-fill the reports:", "backfill")
ui.footer()
