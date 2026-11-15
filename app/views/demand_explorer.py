"""Demand explorer: trends, seasonality and states."""
import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.analytics import festival_effects, relative_to_local, state_growth, yoy_overlay
from src.config import TARGET_LABELS
from src.viz import (ACCENT, CATEGORICAL, COLORS, HISTORY_SOURCE, NEUTRAL, REF, axis_title, hbar, month_year_heatmap, rgba,
                     scale, style)

ui.page("Demand explorer", "Demand history, seasonal and holiday patterns, and growth by state.", badges=[ui.updated_badge()])
d = common.daily()
if d is None:
    ui.empty_state("No demand data yet. Run the daily pipeline to download it:", "pipeline")
    ui.stop()
c1, c2 = st.columns([1.4, 1])
target = c1.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="energy",
                              key="ex_target", label_visibility="collapsed") or "energy"
with c2:
    rng = ui.period(d, key="ex_period", default="2 years")
div, unit = scale(target)
dec = 0 if unit == "MU" else 1
h = d.loc[rng[0]:rng[1]] if rng else d
tabs = st.tabs(["Trends", "Seasonality", "States"])

# ----------------------------------------------------------------------------- trends
with tabs[0]:
    s = h[target] / div
    last = d[target].dropna()
    end_ = last.index.max()
    l12 = last.loc[end_ - pd.Timedelta(days=364):].mean()
    p12 = last.loc[end_ - pd.Timedelta(days=729):end_ - pd.Timedelta(days=365)].mean()
    growth = l12 / p12 - 1 if p12 == p12 and p12 else float("nan")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Daily", line=dict(color=NEUTRAL["faint"], width=1),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(30, min_periods=10).mean(), name="30-day mean",
                             line=dict(color=COLORS["ours"], width=2.5)))
    imp = h[f"{target}_imputed"] if f"{target}_imputed" in h else None
    if imp is not None and imp.any():
        fig.add_trace(go.Scatter(x=s.index[imp], y=s[imp], mode="markers", name="Imputed",
                                 marker=dict(symbol="x", size=7, color=NEUTRAL["strong"])))
    ui.chart(style(fig, f"{TARGET_LABELS[target]} is {growth:+.1%} higher over the last 12 months than the 12 before"
                        if growth == growth else TARGET_LABELS[target], 400, unit, legend="direct",
                   subtitle=f"{unit}; daily values (grey) and 30-day mean", source=HISTORY_SOURCE))

    with st.expander("Year on year"):
        y = yoy_overlay(d, target)
        years = sorted(y["year"].unique())
        fig = go.Figure()
        for i, yr in enumerate(years):
            g = y[y["year"] == yr]
            latest = yr == years[-1]
            fig.add_trace(go.Scatter(x=g["doy"], y=g["value"] / div, name=str(yr), mode="lines",
                                     line=dict(color=COLORS["ours"] if latest else [NEUTRAL["faint"], NEUTRAL["mid"], NEUTRAL["strong"]][
                                         min(len(years) - 2 - i, 2) if len(years) > 1 else 0], width=2.5 if latest else 1.5)))
        axis_title(fig, x="day of year")
        if len(years) > 1:
            piv = y.pivot_table(index="doy", columns="year", values="value")
            cur, prev = piv[years[-1]], piv[years[:-1]].max(axis=1)
            both = cur.notna() & prev.notna()
            share = (cur[both] > prev[both]).mean() if both.any() else float("nan")
            ttl = (f"{years[-1]} has set a new high for the date on {share:.0%} of days so far"
                   if share == share else "Year-on-year comparison")
        else:
            ttl = "Year-on-year comparison"
        ui.chart(style(fig, ttl, 380, unit, legend="direct", subtitle=f"{unit}, 7-day mean by day of year; "
                                                                      f"{years[-1]} in blue", source=HISTORY_SOURCE))

        yearly = d[target].groupby(d.index.year).agg(["mean", "max", "count"])
        yearly["mean"] /= div
        yearly["max"] /= div
        yearly["growth"] = 100 * yearly["mean"].pct_change()
        yearly.index.name = "Year"
        ui.table(yearly.rename(columns={"mean": f"Mean ({unit})", "max": f"Maximum ({unit})", "count": "Days",
                                        "growth": "Growth of mean %"}), max_height=None)
        st.caption("Partial years (2023 from April, the current year to date) are not directly comparable on the mean.")

# ----------------------------------------------------------------------------- seasonality
with tabs[1]:
    ser = d[target].dropna() / div
    mm = ser.groupby(ser.index.month).mean()
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    ui.chart(month_year_heatmap(ser, f"Demand is highest in {months[int(mm.idxmax()) - 1]} and lowest in "
                                     f"{months[int(mm.idxmin()) - 1]}", unit,
                                subtitle=f"Monthly mean {TARGET_LABELS[target].lower()}, {unit}"))
    r = 100 * (relative_to_local(d[target]) - 1)
    r = r.loc[h.index]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    med = [r[r.index.day_of_week == i].median() for i in range(7)]
    c1, c2 = st.columns(2, gap="large")
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_trace(go.Box(y=r[r.index.day_of_week == i], name=n, marker=dict(color=ACCENT if i < 6 else NEUTRAL["strong"], size=3),
                             boxpoints="outliers", line=dict(width=1.2), fillcolor=rgba(ACCENT, 0.08)))
    fig.add_hline(y=0, line=dict(color=REF, width=1))
    with c1:
        ui.chart(style(fig, f"Sundays run {med[6]:+.1f} % against a typical day", 360, "%", legend=False,
                       subtitle="% vs the centred 15-day median, by weekday", source=HISTORY_SOURCE), half=True)
    fe = festival_effects(d, target)
    with c2:
        if len(fe):
            big = fe.loc[fe["effect_pct"].abs().idxmax()]
            fe = fe.sort_values("effect_pct", ascending=False)
            ui.chart(hbar(fe["event"], fe["effect_pct"],
                          [f"{v:+.1f} % (n={n})" for v, n in zip(fe["effect_pct"], fe["n_days"])],
                          title=f"{big['event']} moves demand most ({big['effect_pct']:+.1f} %)",
                          subtitle="% on the day vs the centred 15-day median; n = occurrences",
                          source=HISTORY_SOURCE, hover="%{y}: %{x:+.1f} %", height=360), half=True)
    st.caption("Effects are measured against a centred 15-day rolling median, which removes season and trend. "
               "Festival effects rest on 3–5 occurrences each and are approximate.")

# ----------------------------------------------------------------------------- states
with tabs[2]:
    st_df = common.states()
    if st_df is None:
        ui.empty_state("No state-level data yet. Back-fill the daily reports to build it:", "backfill")
    else:
        end = rng[1] if rng else st_df["date"].max()
        g = state_growth(st_df, end=end)
        if g.empty:
            ui.empty_state("State growth needs two years of data before the end of the selected period. Choose a longer period.", None)
        else:
            c1, c2 = st.columns(2, gap="large")
            top = g.head(12)
            with c1:
                ui.chart(hbar(top["state"], top["growth_pct"], [f"{v:+.1f} %" for v in top["growth_pct"]], ACCENT,
                              title=f"{g.iloc[0]['state']} is growing fastest ({g.iloc[0]['growth_pct']:+.1f} %)",
                              subtitle=f"Energy, 12 months to {end:%b %Y} vs the previous 12, %",
                              source=HISTORY_SOURCE, hover="%{y}: %{x:+.1f} %", height=440), half=True)
            big = g.sort_values("energy_mu", ascending=False).head(12)
            with c2:
                ui.chart(hbar(big["state"], big["energy_mu"] / 1000, [f"{v / 1000:,.0f} TWh" for v in big["energy_mu"]],
                              CATEGORICAL[1], title=f"{big.iloc[0]['state']} uses the most electricity",
                              subtitle="Energy met, last 12 months, TWh", source=HISTORY_SOURCE,
                              hover="%{y}: %{x:,.0f} TWh", height=440), half=True)
            choices = sorted(st_df["state"].unique())
            default = [x for x in ["Maharashtra", "UP", "Gujarat", "Tamil Nadu"] if x in choices][:4]
            c1, c2 = st.columns([2, 1])
            sel = c1.multiselect("States (up to 6)", choices, default=default, max_selections=6)
            metric = c2.segmented_control("Metric", ["max_demand_mw", "energy_met_mu"], default="max_demand_mw",
                                          format_func={"max_demand_mw": "Peak (GW)", "energy_met_mu": "Energy (MU)"}.get,
                                          key="st_metric") or "max_demand_mw"
            dd = st_df[st_df["state"].isin(sel)]
            if rng:
                dd = dd[(dd["date"] >= rng[0]) & (dd["date"] <= rng[1])]
            fig = go.Figure()
            chg = {}
            for i, x in enumerate(sel):
                v = dd[dd["state"] == x].set_index("date")[metric].rolling(7, min_periods=3).mean()
                if metric == "max_demand_mw":
                    v = v / 1000
                v = v.dropna()
                if len(v) > 60:
                    chg[x] = v.iloc[-30:].mean() / v.iloc[:30].mean() - 1
                fig.add_trace(go.Scatter(x=v.index, y=v, name=x, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2)))
            u = "GW" if metric == "max_demand_mw" else "MU"
            lead = max(chg, key=chg.get) if chg else None
            ttl = (f"{lead} has grown most of the selected states over the period ({chg[lead]:+.0%})" if lead
                   else "Selected states")
            ui.chart(style(fig, ttl, 380, u, legend="direct", subtitle=f"7-day mean, {u}", source=HISTORY_SOURCE))
            ui.kicker("All states")
            tbl = g.assign(peak_mw=g["peak_mw"] / 1000)
            ui.table(tbl.rename(columns={"region": "Region", "state": "State", "energy_mu": "Energy, last 12 months (MU)",
                                         "prev_energy_mu": "Energy, previous 12 months (MU)", "growth_pct": "Growth %",
                                         "peak_mw": "Peak, last 12 months (GW)"}))
ui.footer()
