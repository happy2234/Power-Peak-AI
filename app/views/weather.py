"""Weather impact: accuracy and demand by weather condition and season, and the next 7 days."""
import common
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.config import TARGET_LABELS
from src.evaluate import REGIME_ORDER, SEASON_ORDER, SEASONS
from src.insights import weather_drivers
from src.viz import (CATEGORICAL, COLORS, MODEL_LABELS, NEUTRAL, axis_title, band_traces, mape_bars, our_models, scale, style,
                     weather_week)

ui.page("Weather impact", "How weather moves demand, how accurate each forecast is in each kind of weather, and "
                          "the weather behind the next 7 days.", badges=[ui.updated_badge()])
d = common.daily()
if d is None or "tmax_ai" not in d:
    ui.empty_state("No weather data yet. Run the daily pipeline to download it:", "pipeline")
    ui.stop()
target = st.segmented_control("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, default="evening_peak",
                              key="wx_target", label_visibility="collapsed") or "evening_peak"
div, unit = scale(target)
best = common.backtest_best()
ev = common.eval_weather()
seas = pd.Series([SEASONS[m] for m in d.index.month], index=d.index)


def accuracy(kind: str, order: list[str], noun: str):
    if ev is None:
        ui.empty_state("Accuracy by weather comes from the backtest. Run it:", "backtest")
        return
    t = ev[(ev["kind"] == kind) & (ev["target"] == target)]
    ms = (["grid_india"] if "grid_india" in set(t["model"]) else []) + [
        m for m in our_models(t["model"].unique()) if m != "seasonal_naive"]
    order = [o for o in order if o in set(t["label"])]
    b = t[t["model"] == best].set_index("label")["MAPE"].reindex(order)
    g = t[t["model"] == "grid_india"].set_index("label")["MAPE"].reindex(order)
    hard = b.idxmax() if b.notna().any() else "–"
    gap = (g - b).dropna()
    ttl = (f"{common.model_label(best)} is least accurate on {hard.split(' (')[0]} days, yet beats Grid-India in "
           f"{(gap > 0).sum()} of {len(gap)} {noun}s" if len(gap) else f"Accuracy by {noun}")
    ui.chart(mape_bars(t, ms, order, ttl, best=best, subtitle=f"MAPE %, {TARGET_LABELS[target].lower()}, 12-month "
                                                              "backtest; days labelled by observed weather"))
    ui.terms("MAPE")
    piv = t.pivot_table(index="label", columns="model", values="MAPE").reindex(order).reindex(columns=ms)
    out = pd.DataFrame({noun.capitalize(): piv.index,
                        "Days": t.drop_duplicates("label").set_index("label")["days"].reindex(order).to_numpy()})
    for m in ms:
        out[f"{MODEL_LABELS.get(m, m)} MAPE %"] = piv[m].to_numpy()
    with st.expander(f"MAPE by {noun}, all models"):
        ui.table(out, max_height=None)


def scatter(col: str, lab: str, key: str):
    """Demand vs one weather variable, coloured by season, plus the de-trended view."""
    rel = 100 * (d[target] / d[target].rolling(31, center=True, min_periods=10).mean() - 1)
    xa = d[f"{col}_anom"] if f"{col}_anom" in d else d[col]
    corr = {s: rel[seas == s].corr(xa[seas == s]) for s in SEASON_ORDER}
    strongest = max(((s, v) for s, v in corr.items() if v == v), key=lambda kv: abs(kv[1]), default=("", 0))
    c1, c2 = st.columns(2, gap="large")
    fig = go.Figure()
    for i, s in enumerate(SEASON_ORDER):
        m = seas == s
        fig.add_trace(go.Scatter(x=d.loc[m, col], y=d.loc[m, target] / div, mode="markers", name=s.split(" (")[0],
                                 marker=dict(size=5, color=CATEGORICAL[i], opacity=0.6, line=dict(width=0)),
                                 text=d.index[m].strftime("%d %b %Y"),
                                 hovertemplate="%{text}: %{x:.1f}, %{y:,.1f}<extra></extra>"))
    axis_title(fig, x=lab)
    fig.update_layout(hovermode="closest")
    with c1:
        ui.chart(style(fig, f"{TARGET_LABELS[target]} against {lab.split(' (')[0].lower()}", 400, unit,
                       subtitle=f"{unit}, each dot a day since April 2023, by season",
                       source="Grid-India (NLDC), Open-Meteo"), key=f"sc1_{key}", half=True)
    fig = go.Figure()
    for i, s in enumerate(SEASON_ORDER):
        m = seas == s
        fig.add_trace(go.Scatter(x=xa[m], y=rel[m], mode="markers", name=s.split(" (")[0], showlegend=False,
                                 marker=dict(size=5, color=CATEGORICAL[i], opacity=0.6, line=dict(width=0)),
                                 text=d.index[m].strftime("%d %b %Y"),
                                 hovertemplate="%{text}: %{x:+.1f}, %{y:+.1f} %<extra></extra>"))
    axis_title(fig, x=f"{lab.split(' (')[0]}: difference from normal" if f"{col}_anom" in d else lab)
    fig.update_layout(hovermode="closest")
    with c2:
        ui.chart(style(fig, f"Strongest link in {strongest[0].split(' (')[0].lower()} (r = {strongest[1]:+.2f})"
                       if strongest[0] else "Relative to the surrounding month", 400, "%",
                       subtitle="Demand vs its 31-day centred mean, %; weather vs its normal",
                       source="Grid-India (NLDC), Open-Meteo"),
                 key=f"sc2_{key}", half=True)
    st.caption("Right: removes growth and the seasonal level from demand, and the seasonal normal from the weather, "
               "so the cloud shows the short-term effect. Correlation by season: "
               + ", ".join(f"{s.split(' (')[0]} {v:+.2f}" for s, v in corr.items() if v == v)
               + ". Rain and clouds cut rooftop solar, which is behind the meter, so they can raise daytime grid "
                 "demand even as they cool the air.")


VARS = {"Max temperature, all India (°C)": "tmax_ai", "Night temperature, North India (°C)": "tmin_nr",
        "Heat index, all India (°C)": "heat_index_ai", "Rain, all India (mm)": "precip_ai",
        "Rain over 3 days, all India (mm)": "rain3_ai", "Daytime cloud cover, all India (%)": "cloud_day_ai",
        "Dew point, all India (°C)": "dew_mean_ai", "Relative humidity, all India (%)": "rh_mean_ai",
        "Wind gusts, all India (km/h)": "gust_max_ai", "Fog likelihood, North India (0–1)": "fog_nr"}
VARS = {k: v for k, v in VARS.items() if v in d}

tabs = st.tabs(["By condition", "By season", "Next 7 days"])
with tabs[0]:
    accuracy("regime", REGIME_ORDER, "condition")
    st.caption("Conditions, first match wins: heavy rain/storm (top 5 % rain or gusts), cold/fog (Nov–Feb, North India "
               "night ≤ 8 °C or fog likely), hot (top 10 % max temperature or heat index), rainy (≥ 2.5 mm), cloudy "
               "(daytime cloud ≥ 60 %), humid (dew point ≥ 22 °C), normal. Labels use observed weather and serve "
               "evaluation only; the models see only weather forecasts issued before each forecast.")
    with st.expander("How weather moves demand"):
        lab = st.selectbox("Weather variable", list(VARS), key="wx_var")
        scatter(VARS[lab], lab, "cond")

with tabs[1]:
    accuracy("season", SEASON_ORDER, "season")
    m = d[target].groupby(seas).mean().reindex(SEASON_ORDER) / div
    fig = go.Figure(go.Bar(x=[s.split(" (")[0] for s in SEASON_ORDER], y=m.to_numpy(),
                           marker=dict(color=COLORS["ours"], cornerradius=3),
                           text=[f"{v:,.{0 if unit == 'MU' else 1}f}" for v in m], textposition="outside",
                           textfont=dict(size=11, color=NEUTRAL["muted"]), cliponaxis=False))
    ui.chart(style(fig, f"Demand is highest in {m.idxmax().split(' (')[0].lower()}", 320, unit, legend=False,
                   subtitle=f"Mean {TARGET_LABELS[target].lower()} by season since April 2023, {unit}",
                   source="Grid-India (NLDC)"))

with tabs[2]:
    wx, fc, ex = common.latest_weather(), common.latest_forecast(), common.latest_explanations()
    if wx is None or fc is None:
        ui.empty_state("No live weather forecast yet. Run the daily pipeline to fetch it:", "pipeline")
    else:
        wx = wx.sort_values("date")
        is_best = fc["is_best"].fillna(False).astype(bool) if "is_best" in fc else pd.Series(False, index=fc.index)
        shown = fc.loc[is_best, "model"].iloc[0] if is_best.any() else common.best_model()
        g = fc[(fc["model"] == shown) & (fc["target"] == target)].sort_values("date")
        hot = wx.loc[wx["tmax_ai"].idxmax()] if "tmax_ai" in wx else None
        wet = wx.loc[wx["precip_ai"].idxmax()] if "precip_ai" in wx else None
        bits = []
        if hot is not None:
            bits.append(f"warmest on {pd.Timestamp(hot['date']):%a %d %b} ({hot['tmax_ai']:.1f} °C all-India max"
                        + (f", {hot['tmax_ai_anom']:+.1f} °C vs normal" if "tmax_ai_anom" in wx else "") + ")")
        if wet is not None and wet["precip_ai"] >= 1:
            bits.append(f"wettest on {pd.Timestamp(wet['date']):%a %d %b} ({wet['precip_ai']:.1f} mm)")
        ui.headline("The week ahead: " + ("; ".join(bits) if bits else "no notable weather") + ".")
        fig = go.Figure()
        if len(g):
            fig.add_trace(band_traces(g["date"], g["p10"] / div, g["p90"] / div))
            fig.add_trace(go.Scatter(x=g["date"], y=g["p50"] / div, name="Forecast", mode="lines+markers",
                                     line=dict(color=COLORS["ours"], width=2.5), marker=dict(size=6)))
        fig.update_xaxes(tickformat="%a %d")
        if len(g):
            top = g.loc[g["p50"].idxmax()]
            ttl = (f"{TARGET_LABELS[target]} highest on {pd.Timestamp(top['date']):%a %d %b}: "
                   f"{top['p50'] / div:,.{0 if unit == 'MU' else 1}f} {unit}")
        else:
            ttl = f"{TARGET_LABELS[target]} forecast"
        ui.chart(style(fig, ttl, 280, unit, legend=False,
                       subtitle=f"{unit}, P50 with P10–P90 range, {common.model_label(shown)}"))
        figs = weather_week(wx)
        cols = st.columns(2, gap="large")
        for i, f in enumerate(figs):
            with cols[i % 2]:
                ui.chart(f, key=f"wk_{i}", half=True)
        src = wx["weather_source"].iloc[0] if "weather_source" in wx else ""
        st.caption(f"Weather source: {src}. Hover a point to see its difference from the normal for that date.")
        if ex is not None and len(ex):
            e = ex[ex["target"] == target].sort_values("date")
            ui.kicker("Top weather drivers, in plain words")
            ui.table(pd.DataFrame({"Day": [pd.Timestamp(x).strftime("%a %d %b") for x in e["date"]],
                                   "Effect on the forecast vs the recent level": [
                                       " · ".join(weather_drivers(c, unit)) or "no notable weather effect"
                                       for c in e["contrib"]]}), max_height=None)
            st.caption(f"How much each kind of weather moves the LightGBM forecast, {unit}. Weekday, holidays and the recent "
                       "trend are on the 7-day forecast page.")
ui.footer()
