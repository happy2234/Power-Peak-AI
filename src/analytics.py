"""Descriptive analytics used by the exploration pages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.calendar import FESTIVAL_DATES, national_holidays

FEST_NAMES = {"diwali": "Diwali", "holi": "Holi", "eid_ul_fitr": "Eid al-Fitr", "eid_ul_adha": "Eid al-Adha",
              "dussehra": "Dussehra", "durga_puja": "Durga Puja", "pongal": "Pongal",
              "ganesh_chaturthi": "Ganesh Chaturthi", "christmas": "Christmas"}


def relative_to_local(s: pd.Series, window: int = 15) -> pd.Series:
    """Value / centred rolling median: isolates day effects from season and trend."""
    return s / s.rolling(window, center=True, min_periods=window // 2).median()


def festival_effects(d: pd.DataFrame, target: str) -> pd.DataFrame:
    r = relative_to_local(d[target])
    rows = []
    for f, dates in FESTIVAL_DATES.items():
        v = r.reindex(dates).dropna()
        if len(v):
            rows.append({"event": FEST_NAMES[f], "n_days": len(v), "effect_pct": 100 * (v.mean() - 1)})
    nat = [k for k in national_holidays(range(d.index.year.min(), d.index.year.max() + 1))]
    v = r.reindex(pd.DatetimeIndex(nat)).dropna()
    rows.append({"event": "Any national holiday", "n_days": len(v), "effect_pct": 100 * (v.mean() - 1)})
    sun = r[r.index.day_of_week == 6].dropna()
    rows.append({"event": "Sunday", "n_days": len(sun), "effect_pct": 100 * (sun.mean() - 1)})
    return pd.DataFrame(rows).sort_values("effect_pct")


def yoy_overlay(d: pd.DataFrame, target: str, smooth: int = 7) -> pd.DataFrame:
    s = d[target].rolling(smooth, min_periods=1).mean()
    return pd.DataFrame({"year": s.index.year, "doy": s.index.day_of_year, "value": s.values,
                         "date": s.index}).dropna()


def state_growth(st: pd.DataFrame, end=None, days: int = 365) -> pd.DataFrame:
    """Energy met per state: last `days` vs the `days` before (YoY growth)."""
    st = st.copy()
    end = pd.Timestamp(end) if end is not None else st["date"].max()
    cur = st[(st["date"] > end - pd.Timedelta(days=days)) & (st["date"] <= end)]
    prev = st[(st["date"] > end - pd.Timedelta(days=2 * days)) & (st["date"] <= end - pd.Timedelta(days=days))]
    a = cur.groupby(["region", "state"])["energy_met_mu"].sum()
    b = prev.groupby(["region", "state"])["energy_met_mu"].sum()
    out = pd.DataFrame({"energy_mu": a, "prev_energy_mu": b}).dropna()
    out = out[(out["prev_energy_mu"] > 100) & ~out.index.get_level_values("state").str.contains("Bulk|Railways")]
    out["growth_pct"] = 100 * (out["energy_mu"] / out["prev_energy_mu"] - 1)
    out["peak_mw"] = cur.groupby(["region", "state"])["max_demand_mw"].max()
    return out.reset_index().sort_values("growth_pct", ascending=False)


def weather_correlations(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["solar_peak", "evening_peak", "energy", "tmax_ai", "tmean_ai", "app_tmax_ai", "cdd_ai",
            "rh_mean_ai", "precip_ai", "rad_ai"]
    cols = [c for c in cols if c in d]
    # de-trend demand with its 365-day centred mean so growth does not masquerade as weather
    x = d[cols].copy()
    for t in ("solar_peak", "evening_peak", "energy"):
        if t in x:
            x[t] = x[t] / x[t].rolling(365, center=True, min_periods=90).mean()
    return x.corr()


def cdd_fit(d: pd.DataFrame, target: str = "energy") -> dict:
    """Slope of detrended demand vs CDD (all-India), for the Weather page caption."""
    y = d[target] / d[target].rolling(365, center=True, min_periods=90).mean()
    m = pd.concat([d["cdd_ai"], y], axis=1).dropna()
    if len(m) < 30:
        return {}
    b = np.polyfit(m.iloc[:, 0], m.iloc[:, 1], 1)
    return {"pct_per_cdd": 100 * b[0], "r": float(m.corr().iloc[0, 1])}
