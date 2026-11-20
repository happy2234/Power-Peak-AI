"""What-if simulator: temperature changes per region and extra holidays -> forecast change."""
import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import fmt
from src.config import TARGET_LABELS
from src.viz import COLORS, NEUTRAL, band_traces, scale, style

ui.page("What-if simulator", "Change the temperature or add holidays and see how the 7-day forecast responds.",
        badges=[ui.badge("LightGBM")])


@st.cache_resource(show_spinner="Loading the LightGBM model…")
def _model():
    from src.models import registry

    return registry.load("lgbm")


@st.cache_data(ttl=3600, show_spinner="Loading the 7-day weather forecast…")
def _inputs(origin: str):
    from src.features.build import load_daily
    from src.insights import base_raw_weather
    from src.predict import live_raw_weather

    from src.predict import load_raw_weather

    daily, fc = load_daily()
    o = pd.Timestamp(origin)
    raw, src = load_raw_weather(o), "the Open-Meteo forecast the published forecast used"
    if raw is None:
        raw, src = live_raw_weather(o), "Open-Meteo live forecast"
    if raw is None:
        raw, src = base_raw_weather(daily, fc, o), "archived forecast / persistence"
    return daily, fc, raw, src


meta, model = common.latest_meta(), _model()
if model is None or meta is None:
    ui.empty_state("The simulator needs the LightGBM model and a forecast. Train the model, then run the pipeline:",
                   "train_lgbm")
    ui.stop()

from src.insights import REGION_NAMES, Scenario, what_if  # noqa: E402

daily, fc, raw, wx_src = _inputs(meta["origin"])
origin = pd.Timestamp(meta["origin"])
days = [origin + pd.Timedelta(days=h) for h in range(1, 8)]

with st.form("scenario", border=False), ui.card("scenario"):
    ui.kicker("Scenario")
    c1, c2, c3 = st.columns([1.2, 1.4, 1], gap="large")
    all_dt = c1.slider("All-India temperature change (°C)", -5.0, 5.0, 0.0, 0.5)
    hol = c3.multiselect("Treat as holiday", days, format_func=fmt.date)
    reg = {}
    with c2.expander("Per region (adds to the all-India change)"):
        for r in ["nr", "wr", "sr", "er", "ner"]:
            reg[r] = st.slider(REGION_NAMES[r], -5.0, 5.0, 0.0, 0.5, key=f"dt_{r}")
    st.form_submit_button("Run scenario", type="primary", icon=":material/play_arrow:")

delta = {"all": all_dt} if all_dt else {}
delta.update({r: v for r, v in reg.items() if v})
with st.spinner("Running the scenario through the model…"):
    res = what_if(model, daily, fc, origin, Scenario(temp_delta=delta, holidays=hol), raw_weather=raw)

target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                              label_visibility="collapsed") or "evening_peak"
r = res[res["target"] == target].sort_values("date")
div, unit = scale(target)
dec = fmt.scale(target)[2]
avg = r["delta_p50"].mean()
changed = bool(delta or hol)
ui.headline(f"This scenario changes the {TARGET_LABELS[target].lower()} by *{avg / div:+,.{dec}f} {unit}* a day on average."
            if changed else "Set a temperature change or a holiday, then run the scenario.")
ui.kpis([
    {"label": "Average change (P50)", "value": f"{avg / div:+,.{dec}f}", "unit": unit},
    {"label": "Largest change (P50)", "value": f"{r['delta_p50'].abs().max() / div:,.{dec}f}", "unit": unit},
    {"label": "Scenario weekly peak" if target != "energy" else "Scenario weekly energy",
     "value": f"{(r['p50_scen'].max() if target != 'energy' else r['p50_scen'].sum()) / div:,.{dec}f}", "unit": unit},
])

fig = go.Figure()
for lab, sfx, color, dash in [("Base", "base", COLORS["actual"], "dot"), ("Scenario", "scen", COLORS["ours"], "solid")]:
    fig.add_trace(band_traces(r["date"], r[f"p10_{sfx}"] / div, r[f"p90_{sfx}"] / div, color=color if sfx == "scen" else NEUTRAL["mid"],
                              name=f"{lab} P10–P90"))
    fig.add_trace(go.Scatter(x=r["date"], y=r[f"p50_{sfx}"] / div, name=lab, mode="lines+markers",
                             line=dict(color=color, width=2.5 if sfx == "scen" else 1.5, dash=dash), marker=dict(size=6)))
big = r.loc[r["delta_p50"].abs().idxmax()] if len(r) else None
ttl = (f"Largest change on {pd.Timestamp(big['date']):%a %d %b}: {big['delta_p50'] / div:+,.{dec}f} {unit}"
       if changed and big is not None else f"{TARGET_LABELS[target]}: base forecast")
ui.chart(style(fig, ttl, 380, unit, legend="direct",
               subtitle=f"{unit}, P50 with P10–P90 range. LightGBM model, origin {fmt.date(origin)}, base weather: {wx_src}"))

tbl = r[["date", "p50_base", "p50_scen", "delta_p50"]].copy()
tbl[["p50_base", "p50_scen", "delta_p50"]] /= div
with st.expander("Daily numbers"):
    ui.table(tbl.rename(columns={"date": "Date", "p50_base": f"Base P50 ({unit})", "p50_scen": f"Scenario P50 ({unit})",
                                 "delta_p50": f"Change ({unit})"}), max_height=None)
st.caption("Temperature changes shift the regional max, mean, minimum and apparent temperatures and the dew point, "
           "then flow through to degree days, heat index and the anomaly features. Effects outside the range seen "
           "in training are uncertain.")
ui.footer()
