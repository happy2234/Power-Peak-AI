"""7-day forecast: one tab per target with the fan chart, then 'Why this forecast?' and the daily
numbers in collapsed expanders, then the drivers by day; the whole forecast downloads as one CSV."""
import common
import pandas as pd
import streamlit as st

import ui
from src import fmt
from src.config import TARGET_LABELS
from src.insights import weather_drivers, week_insights
from src.viz import MODEL_LABELS, drivers_bar, forecast_fan, scale

d, fc, meta = common.daily(), common.latest_forecast(), common.latest_meta()
ui.page("7-day forecast", "Solar-hour peak, evening peak and daily energy for the next 7 days, with the range we "
                          "expect and what drives each day.", badges=[ui.updated_badge()])
if d is None or fc is None:
    ui.empty_state("No forecast yet. Run the daily pipeline to make one:", "pipeline")
    ui.stop()

best = common.best_model()
models = [m for m in ["ensemble", "lgbm", "lstm", "tft"] if m in fc["model"].unique()]
gi = common.grid_india()
ex = common.latest_explanations()
origin = fc["origin"].min()


def _export(fc: pd.DataFrame) -> pd.DataFrame:
    """The whole forecast in display units: one row per day, target and model."""
    out = fc[["date", "target", "model", "p10", "p50", "p90"]].copy()
    div = out["target"].map(lambda t: fmt.scale(t)[0])
    for c in ("p10", "p50", "p90"):
        out[c] = (out[c] / div).round(1)
    out["unit"] = out["target"].map(lambda t: fmt.scale(t)[1])
    out["target"] = out["target"].map(TARGET_LABELS)
    out["model"] = out["model"].map(lambda m: MODEL_LABELS.get(m, m))
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out.insert(0, "issued_at", (meta or {}).get("issued_at", ""))
    return out.sort_values(["target", "model", "date"])


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
show = c1.multiselect("Models", models, default=[best] if best in models else models[:1],
                      format_func=lambda m: MODEL_LABELS.get(m, m),
                      help="The P10–P90 range is shown for the best model; other models add their P50 line.") or [best]
with c2:
    ui.download_csv(_export(fc), "forecast", label="Download forecast (CSV)", primary=True)

tabs = st.tabs(["Solar-hour peak", "Evening peak", "Energy"])
for tab, target in zip(tabs, ["solar_peak", "evening_peak", "energy"]):
    with tab:
        div, unit = scale(target)
        dec = 0 if unit == "MU" else 1
        hist = d[target].dropna().iloc[-21:]
        g = gi[gi["date"] > hist.index.max()] if gi is not None else None
        b = fc[(fc["model"] == best) & (fc["target"] == target)].sort_values("date")
        if len(b):
            hi = b.loc[b["p50"].idxmax()]
            title = (f"{TARGET_LABELS[target]} highest on {pd.Timestamp(hi['date']):%a %d %b}: "
                     f"{hi['p50'] / div:,.{dec}f} {unit} (likely range {hi['p10'] / div:,.{dec}f}–{hi['p90'] / div:,.{dec}f})")
        else:
            title = TARGET_LABELS[target]
        ui.chart(forecast_fan(hist, fc, target, models=show, best=best, gi=g, title=title, height=380,
                              subtitle=f"{unit}; last 21 days of actuals, then the forecast issued after "
                                       f"{fmt.date(origin)}, P50 with P10–P90"))

        t = fc[(fc["target"] == target) & fc["model"].isin(show + ["grid_india"])]
        rows = []
        for day, gg in t.groupby("date"):
            r = {"Date": pd.Timestamp(day)}
            for _, x in gg.iterrows():
                lab = MODEL_LABELS.get(x["model"], x["model"])
                r[f"{lab} P50 ({unit})"] = x["p50"] / div
                if pd.notna(x["p10"]):
                    r[f"{lab} P10–P90 ({unit})"] = f"{x['p10'] / div:,.{dec}f}–{x['p90'] / div:,.{dec}f}"
            rows.append(r)
        reasons = week_insights(fc, ex, best, target)
        if reasons:
            with st.expander("Why this forecast?"):
                ui.insights(reasons)
        with st.expander("Daily numbers"):
            ui.table(pd.DataFrame(rows), max_height=None, name=f"forecast-{target}")

        if ex is None:
            ui.empty_state("Driver explanations come with the LightGBM forecast. Run the pipeline:", "pipeline")
            continue
        e = ex[ex["target"] == target].sort_values("date")
        if e.empty:
            continue
        st.subheader("Drivers by day")
        c1, c2 = st.columns([1, 1.15], gap="large")
        with c1:
            ui.kicker("Weather effects by day")
            ui.table(pd.DataFrame({
                "Day": [pd.Timestamp(x).strftime("%a %d %b") for x in e["date"]],
                "Main weather effects vs the recent level": [" · ".join(weather_drivers(c, unit)) or "None notable"
                                                             for c in e["contrib"]]}), max_height=None,
                     name=f"weather-effects-{target}")
        with c2:
            day = st.select_slider("Day", options=list(e["date"]), format_func=fmt.date, key=f"day_{target}")
            row = e[e["date"] == day].iloc[0]
            contrib = sorted(row["contrib"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            lead = contrib[0] if contrib else ("", 0)
            ui.chart(drivers_bar(contrib, unit, f"{pd.Timestamp(day):%a %d %b}: {lead[0]} matters most "
                                                f"({lead[1]:+.1f} {unit})",
                                 subtitle=f"vs the recent 7-day level, {unit}; blue raises, grey lowers"), half=True)
st.caption("Peaks in GW, energy in MU. P10–P90: the range we expect on 8 days out of 10. Grid-India publishes a single "
           "value; its energy figure is the total of its 15-minute forecast. Drivers come from the LightGBM model, "
           "grouped by kind (for example, all North India temperature inputs count as Heat in North India).")
ui.footer()
