"""Evening peak: will the evening (non-solar-hour) peak exceed the daytime peak, and stress alerts."""
import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.viz import ACCENT, ALERT, COLORS, HISTORY_SOURCE, NEUTRAL, REF, band_traces, style

ui.page("Evening peak", "When the evening peak will beat the daytime peak, and which evenings risk unusually "
                        "high demand.", badges=[ui.updated_badge()])
d = common.daily()
if d is None:
    ui.empty_state("No demand data yet. Run the daily pipeline to download it:", "pipeline")
    ui.stop()
rng = ui.period(d, key="ev_period", default="1 year")
h = d.loc[rng[0]:rng[1]] if rng else d


def _growth(col):
    s = d[col].dropna()
    end = s.index.max()
    a = s.loc[end - pd.Timedelta(days=364):].mean()
    b = s.loc[end - pd.Timedelta(days=729):end - pd.Timedelta(days=365)].mean()
    return a / b - 1 if b == b and b else float("nan")


ge, gs = _growth("evening_peak"), _growth("solar_peak")
if ge == ge and gs == gs:
    faster = "faster" if ge > gs else "slower"
    ui.headline(f"The evening peak is growing *{faster}* than the daytime peak: {ge:+.1%} vs {gs:+.1%} over the "
                "last 12 months.")

st.subheader("Next 7 days")
s = common.latest_stress()
if s is None:
    ui.empty_state("No evening-peak forecast yet. Run the daily pipeline to make one:", "pipeline")
else:
    thr = s["threshold_top5"].iloc[0] / 1000
    n_alert = int(s["alert"].sum())
    if n_alert:
        ui.note(f"Evening stress alert: on {n_alert} of the next 7 days the evening peak is forecast to be in the "
                f"top 5 % of history (above {thr:,.1f} GW).", alert=True)
    else:
        ui.note(f"No stress alerts this week: no evening is likely to reach the top 5 % of history ({thr:,.1f} GW).")
    fig = go.Figure()
    fig.add_trace(band_traces(s["date"], s["gap_p10"] / 1000, s["gap_p90"] / 1000, name="Gap P10–P90"))
    fig.add_trace(go.Scatter(x=s["date"], y=s["gap_p50"] / 1000, name="Gap", mode="lines+markers",
                             line=dict(color=COLORS["ours"], width=2.5), marker=dict(size=6)))
    fig.add_hline(y=0, line=dict(color=REF, width=1))
    above = int((s["gap_p50"] > 0).sum())
    ui.chart(style(fig, f"The evening peak is forecast above the daytime peak on {above} of 7 days", 300, "GW",
                   legend=False, subtitle="Evening minus solar-hour peak, GW; P50 with P10–P90 range"))
    with st.expander("Daily numbers"):
        t = s[["date", "solar_p50", "evening_p50", "gap_p50", "p_evening_above_solar", "p_evening_top5", "alert"]].copy()
        for c in ["solar_p50", "evening_p50", "gap_p50"]:
            t[c] = t[c] / 1000
        t["alert"] = t["alert"].map({True: "Alert", False: "–"})
        t["p_evening_above_solar"] = 100 * t["p_evening_above_solar"]
        t["p_evening_top5"] = 100 * t["p_evening_top5"]
        ui.table(t.rename(columns={"date": "Date", "solar_p50": "Solar-hour peak (GW)", "evening_p50": "Evening peak (GW)",
                                   "gap_p50": "Gap (GW)", "p_evening_above_solar": "P(evening > solar-hour) %",
                                   "p_evening_top5": "P(evening in top 5 %) %", "alert": "Stress alert"}), max_height=None)

with st.expander("History"):
    t_lvl, t_gap = st.tabs(["Daytime vs evening", "Daily gap"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=(h["solar_peak"] / 1000).rolling(7, min_periods=3).mean(), name="Solar-hour peak",
                             line=dict(color=NEUTRAL["strong"], width=2)))
    fig.add_trace(go.Scatter(x=h.index, y=(h["evening_peak"] / 1000).rolling(7, min_periods=3).mean(), name="Evening peak",
                             line=dict(color=ACCENT, width=2.5)))
    gp = ((h["evening_peak"] - h["solar_peak"]) / 1000).dropna()
    g0, g1 = gp.iloc[:90].mean(), gp.iloc[-90:].mean()
    ttl = (f"The evening runs {abs(g1):.1f} GW {'above' if g1 > 0 else 'below'} the daytime peak lately "
           f"({g0:+.1f} GW at the start of the period)" if len(gp) > 180 else "Daytime and evening peaks")
    with t_lvl:
        ui.chart(style(fig, ttl, 360, "GW",
                   legend="direct", subtitle="7-day mean, GW", source=HISTORY_SOURCE))
    gap = (h["evening_peak"] - h["solar_peak"]) / 1000
    fig = go.Figure(go.Bar(x=gap.index, y=gap, name="Daily gap", showlegend=False,
                           marker=dict(color=[ALERT if v > 0 else NEUTRAL["faint"] for v in gap]),
                           hovertemplate="%{x|%d %b %Y}: %{y:+.1f} GW<extra></extra>"))
    fig.add_trace(go.Scatter(x=gap.index, y=gap.rolling(30, min_periods=7).mean(), line=dict(color=COLORS["actual"], width=2),
                             name="30-day mean"))
    with t_gap:
        ui.chart(style(fig, f"The evening set the daily peak on {(gap > 0).mean():.0%} of days in this period", 320, "GW",
                   legend="direct", subtitle="Evening minus solar-hour peak, GW; red: the evening was higher", source=HISTORY_SOURCE))
st.caption("As solar capacity grows, daytime demand on the grid falls and stress moves to the hours after sunset. "
           "Stress alert: the evening peak is likely to be in the top 5 % of all evenings since April 2023.")
ui.footer()
