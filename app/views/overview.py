"""Overview: the landing page, kept to five things. Title, status and the 'Update data' button;
one headline sentence; three KPI cards; the forecast chart (one target at a time, Grid-India on
request) with the reasons in a collapsed 'Why this forecast?'; a link to how it works.
Drivers live on 7-day forecast, pipeline status on Data pipeline, win rate on vs Grid-India."""
import common
import pandas as pd
import streamlit as st

import ui
from src.config import readonly
from src import fmt
from src.app_data import kpis
from src.insights import week_headline, week_insights
from src.viz import forecast_fan

STALE_DAYS = 2
TARGETS = {"solar_peak": "Solar-hour peak", "evening_peak": "Evening peak", "energy": "Energy"}

d, fc, meta = common.daily(), common.latest_forecast(), common.latest_meta()
if ui.page_with_action("Overview", "All-India electricity demand: the latest actuals and the forecast for the "
                                   "next 7 days.", key="fetch_overview"):
    ui.pipeline_panel()

issued = pd.Timestamp(meta["issued_at"]) if meta and meta.get("issued_at") else None
if issued is not None:
    age = (pd.Timestamp.now().normalize() - issued.normalize()).days
    if age > STALE_DAYS:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"Forecast is {age} days old. Update data to refresh." if not readonly()
                        else f"Forecast is {age} days old. This is a read-only snapshot, refreshed daily.")
            if ui.fetch_enabled() and ui.update_button("fetch_stale"):
                ui.pipeline_panel()

if d is None:
    ui.empty_state("No demand data yet. Run the daily pipeline to download it:", "pipeline")
    ui.stop()

best = common.best_model()
k = kpis(d)
last = k["last_date"]
f = fc[fc["model"] == best] if fc is not None else None
have_fc = f is not None and len(f) > 0

# ---- 2. one headline sentence
head = week_headline(fc, best, (meta or {}).get("records")) if have_fc else None
ui.headline(head or f"Latest daily peak {k['latest_peak'] / 1000:,.1f} GW on {fmt.date(last)}. No forecast yet.")

# ---- 3. three KPI cards
items = []
if have_fc:
    p = f[f["target"].isin(["solar_peak", "evening_peak"])].groupby("date")["p50"].max()
    items.append({"label": "Next 7 days peak", "value": f"{p.max() / 1000:,.1f}", "unit": "GW",
                  "note": f"highest forecast, {pd.Timestamp(p.idxmax()):%a %d %b}"})
items.append({"label": "Latest actual peak", "value": f"{k['latest_peak'] / 1000:,.1f}", "unit": "GW",
              "note": f"{last:%a %d %b}"})
acc = common.recent_accuracy(common.backtest_best()) if common.backtest() is not None else {}
if acc and acc.get("mape") == acc.get("mape"):
    items.append({"label": "Our accuracy", "value": f"{acc['mape']:.2f}", "unit": "% MAPE",
                  "note": "last 30 days", "help": "MAPE"})
ui.kpis(items)

# ---- 4. the forecast chart
with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
    target = st.segmented_control("Target", list(TARGETS), format_func=TARGETS.get, default="solar_peak",
                                  key="home_target", label_visibility="collapsed") or "solar_peak"
    compare = st.checkbox("Compare with Grid-India", value=False, key="home_gi") if have_fc else False
if not have_fc:
    ui.empty_state("No forecast yet. Run the daily pipeline to make one:", "pipeline")
else:
    hist = d[target].dropna().iloc[-30:]
    gi = common.grid_india() if compare else None
    gi = gi[gi["date"] > hist.index.max()] if gi is not None else None
    div, unit, dec = fmt.scale(target)
    g = f[f["target"] == target].sort_values("date")
    if len(g):
        mean_next, mean_last = g["p50"].mean() / div, hist.iloc[-7:].mean() / div
        chg = mean_next / mean_last - 1
        trend = "rise" if chg > 0.01 else "ease" if chg < -0.01 else "hold steady"
        title = (f"{TARGETS[target]} expected to {trend}: {mean_next:,.{dec}f} {unit} a day on average "
                 f"({chg:+.1%} vs the last 7 days)")
    else:
        title = TARGETS[target]
    ui.chart(forecast_fan(hist, fc, target, models=[best], best=best, gi=gi, title=title, height=400,
                          subtitle=f"{unit}; last 30 days of actuals, then our forecast with its likely range"
                                   + (" and Grid-India's week-ahead forecast" if compare else "")))
    reasons = week_insights(fc, common.latest_explanations(), best, target)
    if reasons:
        with st.expander("Why this forecast?", expanded=False):
            ui.insights(reasons)

# ---- 5. how it works
st.page_link(st.session_state["pp_pages"]["pipeline"], label="How it works", icon=":material/arrow_forward:")
ui.footer()
