"""Feature engineering + leakage-safe sample construction.

Two layers:
  1. Daily tables (one row per date, no notion of "forecast origin"):
       features.parquet       targets, flags, calendar, OBSERVED weather (regional + all-India)
       weather_fc.parquet     archived weather FORECASTS, one row per (date, lead)
  2. Sample builders that take a forecast origin t (last day with known actuals) and build
     inputs for target days d = t+1 .. t+7 using ONLY information available at t:
       * targets/observed weather up to and including t
       * weather forecasts issued at (or before) t: lead >= h for day t+h
       * calendar of any date (deterministic)
     -> make_tabular() for LightGBM, make_sequences() for the neural nets.

Targets are modelled as ratios to the origin's reference level ref = mean(y[t-6..t]) so that
models can extrapolate the ~5 %/yr growth trend and forecast new records.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED, REGIONS, TARGETS
from src.features.calendar import calendar_features

log = logging.getLogger(__name__)

CDD_BASE = 24.0      # cooling-degree base (°C, daily mean temperature)
HDD_BASE = 18.0      # heating-degree base (°C) — winter heating in the North
RAIN_DAY_MM = 2.5    # IMD "rainy day" threshold
REF_DAYS = 7
# Fixed cut-off for the climatological normals behind the anomaly features: observed weather up
# to the last origin whose labels are known before the first backtest refit (2025-11-01 - 8 d),
# the same cut-off used for hyper-parameter tuning. Every backtest/live origin is later, so the
# normals never contain information from after a forecast's issue time.
NORMALS_END = "2025-10-24"
N_HARMONICS = 3

# raw (per city / region) daily variables
WX_BASE = ["tmax", "tmean", "app_tmax", "rh_mean", "precip", "rad",
           "tmin", "dew_mean", "cloud_mean", "cloud_day", "wind_max", "gust_max", "dpd_morning"]
# derived per unit (region, city): degree days, heat index, rain-day flag, fog proxy
WX_DERIVED = ["cdd", "hdd", "heat_index", "rain_day", "fog"]
AI_VARS = ["tmax", "tmin", "tmean", "app_tmax", "heat_index", "rh_mean", "dew_mean", "precip",
           "rain_day", "cloud_mean", "cloud_day", "rad", "wind_max", "gust_max", "cdd", "hdd", "fog"]
REG_VARS = ["tmax", "tmin", "cdd", "hdd", "precip", "cloud_day", "fog"]
# anomaly = value - climatological normal for that place and day of year
ANOM_COLS = ([f"{v}_ai" for v in ("tmax", "tmin", "rh_mean", "dew_mean", "precip", "cloud_day", "rad", "wind_max")]
             + [f"{v}_{r}" for v in ("tmax", "tmin", "precip") for r in REGIONS])
# per-day weather features used by models
WX_FEATS = ([f"{v}_ai" for v in AI_VARS] + [f"{v}_{r}" for v in REG_VARS for r in REGIONS]
            + [f"{c}_anom" for c in ANOM_COLS])
# 3-day rain sum (d-2..d): built per origin, mixing observed (<= origin) and forecast days
RAIN3_COLS = ["rain3_ai"] + [f"rain3_{r}" for r in REGIONS]
# raw regional weather kept alongside (not model inputs) so scenarios can be re-derived
WX_RAW = [f"{v}_{r}" for v in WX_BASE for r in REGIONS]
CAL_FEATS = ["dow", "is_saturday", "is_sunday", "is_weekend", "month", "doy_sin", "doy_cos",
             "dow_sin", "dow_cos", "is_national_holiday", "fest_diwali", "fest_holi",
             "fest_eid_ul_fitr", "fest_eid_ul_adha", "fest_dussehra", "fest_durga_puja",
             "fest_pongal", "fest_ganesh_chaturthi", "fest_christmas", "diwali_window",
             "is_festival", "is_holiday", "days_to_festival", "days_from_festival", "is_bridge"]
TGT = list(TARGETS)


# ============================================================================ weather
def region_weights(clean: pd.DataFrame, end: str = "2024-03-31") -> dict[str, float]:
    """Each region's share of all-India energy met over the first financial year (fixed, so it
    is known before every backtest origin)."""
    first = clean[clean["date"] <= pd.Timestamp(end)]
    tot = {r: first[f"energy_met_mu_{r}"].sum() for r in REGIONS}
    s = sum(tot.values())
    return {r: tot[r] / s for r in REGIONS}


def heat_index(t_c, dew_c):
    """NOAA heat index (°C) from air temperature and dew point (Rothfusz regression with the
    simple formula below 26.7 °C, i.e. 80 °F). NaN where the dew point is unknown."""
    t_c, dew_c = np.asarray(t_c, float), np.asarray(dew_c, float)
    rh = 100 * np.exp(17.625 * dew_c / (243.04 + dew_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
    rh = np.clip(rh, 0, 100)
    T = t_c * 9 / 5 + 32
    simple = 0.5 * (T + 61 + (T - 68) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * T + 10.14333127 * rh - 0.22475541 * T * rh - 6.83783e-3 * T * T
            - 5.481717e-2 * rh * rh + 1.22874e-3 * T * T * rh + 8.5282e-4 * T * rh * rh
            - 1.99e-6 * T * T * rh * rh)
    hi = np.where(T >= 80, full, simple)
    return (hi - 32) * 5 / 9


def fog_proxy(tmin, dpd_morning):
    """0..1 fog likelihood: early-morning air near saturation (dew-point depression < 3 °C) on a
    cool night (tmin < 18 °C, full weight below 12 °C). Open-Meteo has no archived visibility."""
    sat = np.clip((3.0 - np.asarray(dpd_morning, float)) / 3.0, 0, 1)
    cool = np.clip((18.0 - np.asarray(tmin, float)) / 6.0, 0, 1)
    return sat * cool


def _derive_unit(out: pd.DataFrame, u: str) -> None:
    """In place: degree days, heat index, rain-day flag and fog proxy for one unit (region/city)."""
    col = lambda v: out[f"{v}_{u}"] if f"{v}_{u}" in out else pd.Series(np.nan, index=out.index)  # noqa: E731
    out[f"cdd_{u}"] = (col("tmean") - CDD_BASE).clip(lower=0)
    out[f"hdd_{u}"] = (HDD_BASE - col("tmean")).clip(lower=0)
    out[f"heat_index_{u}"] = heat_index(col("tmax"), col("dew_mean"))
    p = col("precip")
    out[f"rain_day_{u}"] = np.where(p.isna(), np.nan, (p >= RAIN_DAY_MM).astype(float))
    out[f"fog_{u}"] = fog_proxy(col("tmin"), col("dpd_morning"))


def _doy_design(dates) -> np.ndarray:
    doy = pd.DatetimeIndex(dates).day_of_year.to_numpy(float)
    cols = [np.ones_like(doy)]
    for k in range(1, N_HARMONICS + 1):
        w = 2 * np.pi * k * doy / 365.25
        cols += [np.sin(w), np.cos(w)]
    return np.stack(cols, 1)


def fit_normals(obs: pd.DataFrame, cols, end: str = NORMALS_END) -> dict[str, list[float]]:
    """Smooth day-of-year climatology per column (mean + 3 annual harmonics), least squares on
    OBSERVED weather up to `end` only. obs: indexed by date."""
    o = obs[obs.index <= pd.Timestamp(end)]
    X = _doy_design(o.index)
    out = {}
    for c in cols:
        if c not in o:
            continue
        y = o[c].to_numpy(float)
        ok = ~np.isnan(y)
        if ok.sum() >= 2 * X.shape[1]:
            out[c] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0].tolist()
    return out


def normal_at(normals: dict, col: str, dates) -> np.ndarray:
    if col not in normals:
        return np.full(len(dates), np.nan)
    return _doy_design(dates) @ np.asarray(normals[col])


def derive_weather(reg: pd.DataFrame, weights: dict[str, float], dates=None,
                   normals: dict | None = None) -> pd.DataFrame:
    """Regional columns <var>_<region> -> derived variables per region + demand-weighted
    all-India values (+ anomalies when `normals` and the row dates are given). Pure function of
    its inputs, reused for observed, forecast and what-if weather. Missing raw variables give NaN."""
    out = reg.copy()
    for v in WX_BASE:
        for r in REGIONS:
            if f"{v}_{r}" not in out:
                out[f"{v}_{r}"] = np.nan
    for r in REGIONS:
        _derive_unit(out, r)
    w = np.array([weights[r] for r in REGIONS])
    new = {}
    for v in WX_BASE + WX_DERIVED:
        vals = out[[f"{v}_{r}" for r in REGIONS]].to_numpy(float)
        ok = ~np.isnan(vals)
        new[f"{v}_ai"] = np.where(ok.any(1), np.nansum(np.nan_to_num(vals) * w, 1)
                                  / np.maximum((ok * w).sum(1), 1e-9), np.nan)
    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    if dates is None:
        dates = out["date"] if "date" in out else out.index
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    anom = {f"{c}_anom": (out[c].to_numpy(float) - normal_at(normals, c, dates)) if normals
            else np.full(len(out), np.nan) for c in ANOM_COLS}
    return pd.concat([out, pd.DataFrame(anom, index=out.index)], axis=1)


def observed_weather_daily(weights, normals: dict | None = None,
                           normals_end: str = NORMALS_END) -> tuple[pd.DataFrame, dict]:
    """Observed regional + all-India weather (date index) and the normals fitted on it."""
    from src.data.weather import load_observed_all, regional

    obs = load_observed_all()
    if obs.empty:
        raise FileNotFoundError("no observed weather cached; run python -m src.data.weather")
    reg = regional(obs, ["date"]).set_index("date")
    if normals is None:
        base = derive_weather(reg, weights)
        normals = fit_normals(base, ANOM_COLS, normals_end)
    return derive_weather(reg, weights, normals=normals), normals


def city_weather_daily(normals_end: str = NORMALS_END) -> pd.DataFrame:
    """Observed weather per city (wide: <var>_<city>) with derived variables and anomalies vs
    each city's own normal (fitted up to normals_end). For analysis and the app."""
    from src.data.weather import CITIES, load_observed_all, city_wide

    obs = load_observed_all()
    if obs.empty:
        return pd.DataFrame()
    cw = city_wide(obs, ["date"]).set_index("date")
    for c in CITIES:
        _derive_unit(cw, c)
    acols = [f"{v}_{c}" for v in ("tmax", "tmin", "precip", "cloud_day", "dew_mean") for c in CITIES]
    nrm = fit_normals(cw, acols, normals_end)
    an = {f"{c}_anom": cw[c].to_numpy(float) - normal_at(nrm, c, cw.index) for c in acols}
    return pd.concat([cw, pd.DataFrame(an, index=cw.index)], axis=1)


def forecast_weather_long(weights, normals: dict) -> pd.DataFrame:
    from src.data.weather import load_forecasts_all, regional

    fc = load_forecasts_all()
    if fc.empty:
        return pd.DataFrame(columns=["date", "lead"] + list(dict.fromkeys(WX_FEATS + WX_RAW)))
    reg = regional(fc, ["date", "lead"])
    # only use a forecast row if every region has a value (else all-India would be biased)
    reg = reg.dropna(subset=[f"tmax_{r}" for r in REGIONS]).reset_index(drop=True)
    out = derive_weather(reg, weights, normals=normals)
    return out[["date", "lead"] + list(dict.fromkeys(WX_FEATS + WX_RAW))]


def load_normals() -> dict:
    import json

    p = PROCESSED / "weather_normals.json"
    return json.loads(p.read_text())["normals"] if p.exists() else {}


# ============================================================================ daily table
def build_daily_features(save: bool = True, normals_end: str = NORMALS_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily feature table + archived weather forecasts. `normals_end`: last date of the observed
    weather used for the anomaly normals (the fixed train/test split passes its train end)."""
    clean = pd.read_parquet(PROCESSED / "daily_clean.parquet")
    weights = region_weights(clean)
    obs, normals = observed_weather_daily(weights, normals_end=normals_end)
    df = clean.set_index("date")
    cal = calendar_features(df.index)
    wx = obs.reindex(df.index)[list(dict.fromkeys(WX_FEATS + WX_RAW))]
    # observed 3-day rain sums (d-2..d), for the encoder history and evaluation
    rain3 = pd.DataFrame({f"rain3_{u}": obs[f"precip_{u}"].rolling(3, min_periods=3).sum()
                          for u in ["ai"] + REGIONS}).reindex(df.index)
    wx = pd.concat([wx, rain3], axis=1)
    keep = TGT + [f"{t}_imputed" for t in TGT] + ["is_imputed", "is_outlier"]
    daily = pd.concat([df[keep], cal[CAL_FEATS], wx], axis=1)
    daily["peak_gap"] = daily["evening_peak"] - daily["solar_peak"]
    daily.index.name = "date"
    fc = forecast_weather_long(weights, normals)
    if save:
        import json

        daily.reset_index().to_parquet(PROCESSED / "features.parquet", index=False)
        fc.to_parquet(PROCESSED / "weather_fc.parquet", index=False)
        pd.Series(weights).to_json(PROCESSED / "region_weights.json")
        (PROCESSED / "weather_normals.json").write_text(json.dumps({"end": NORMALS_END, "normals": normals}))
        city = city_weather_daily()
        if len(city):
            city.reset_index().to_parquet(PROCESSED / "weather_city.parquet", index=False)
    log.info("features: %s, weather_fc: %s", daily.shape, fc.shape)
    return daily, fc


def load_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(PROCESSED / "features.parquet").set_index("date")
    fc = pd.read_parquet(PROCESSED / "weather_fc.parquet")
    return daily, fc


# ============================================================================ helpers
def _fc_matrix(fc: pd.DataFrame, cols=None) -> dict[int, pd.DataFrame]:
    cols = cols or WX_FEATS
    return {l: g.set_index("date").reindex(columns=cols) for l, g in fc.groupby("lead")}


def weather_known_at(fc: pd.DataFrame, daily: pd.DataFrame, origins: pd.DatetimeIndex,
                     h: int, fcm: dict | None = None, cols=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Weather for days origin+h as known at each origin.

    Chain: forecast issued at origin (lead h) -> older forecasts (lead h+1..7, issued before
    origin) -> persistence (mean observed weather over origin-2..origin). Never uses observed
    weather after the origin. Returns (values indexed like origins, source code 0/1/2)."""
    cols = cols or WX_FEATS
    fcm = fcm if fcm is not None else _fc_matrix(fc, cols)
    target = origins + pd.Timedelta(days=h)
    vals = pd.DataFrame(np.nan, index=origins, columns=cols)
    src = np.full(len(origins), 2, dtype=np.int8)
    for lead in range(h, HORIZON + 1):
        if lead not in fcm:
            continue
        v = fcm[lead].reindex(target).to_numpy()
        missing = vals.isna().all(axis=1).to_numpy() & ~np.isnan(v).all(axis=1)
        vals.iloc[missing] = v[missing]
        src[missing] = 0 if lead == h else 1
    still = vals.isna().all(axis=1).to_numpy()
    if still.any():
        obs = daily.reindex(columns=cols)
        pers = obs.rolling(3, min_periods=1).mean().reindex(origins[still]).to_numpy()
        vals.iloc[still] = pers
    # Variables the forecast archive lacks for some dates (dew point, clouds and wind before
    # 2024) are filled cell by cell with the same persistence value (observed <= origin).
    part = vals.isna().to_numpy()
    if part.any():
        obs_cols = [c for c in cols if c in daily.columns]
        pers = daily[obs_cols].rolling(3, min_periods=1).mean().reindex(origins)
        pers = pers.reindex(columns=cols)
        vals = vals.mask(vals.isna(), pd.DataFrame(pers.to_numpy(), index=vals.index, columns=cols))
    return vals, src


def _hist(series: pd.Series, origins: pd.DatetimeIndex, offsets) -> np.ndarray:
    """series values at origin+offset (offset<=0 or such that date<=origin by construction)."""
    out = []
    for off in offsets:
        out.append(series.reindex(origins + pd.Timedelta(days=int(off))).to_numpy())
    return np.stack(out, 1)


# ============================================================================ extra feature groups
# Computed for every row but used only by models that select the group (LGBMQuantile(extra=...)),
# so saved models keep their exact inputs. Every value uses data at or before the origin, the
# calendar of the target day, or the forecast weather known at the origin (tests/test_accuracy_round.py).
FEST_ALIGN = ("diwali", "holi", "dussehra", "eid_ul_fitr", "eid_ul_adha", "ganesh_chaturthi")
FEST_ALIGN_DAYS = 10
EXTRA_GROUPS = {
    "interactions": ["tmax_x_weekend", "app_tmax_x_weekend", "cdd_x_premonsoon", "cdd_x_monsoon",
                     "cdd_x_postmonsoon", "cdd_x_doy_sin", "cdd_x_doy_cos"],
    "growth": [f"{k}_{f}" for k in TGT for f in ("yoy7", "yoy30", "mom30_90")],
    "lag365": [f"{k}_{f}" for k in TGT for f in ("ly", "ly_shape")],
}
EXTRA_COLS = [c for cols in EXTRA_GROUPS.values() for c in cols]


def last_year_date(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Same day last year: 364 days back (same weekday), or aligned to the same festival when the
    day lies within FEST_ALIGN_DAYS of a major moving festival (Diwali 2025 -> Diwali 2024)."""
    from src.features.calendar import FESTIVAL_DATES

    out = dates - pd.Timedelta(days=364)
    best = np.full(len(dates), FEST_ALIGN_DAYS + 1)
    for f in FEST_ALIGN:
        fd = pd.DatetimeIndex(FESTIVAL_DATES[f])
        for this in fd:
            prev = fd[(fd < this - pd.Timedelta(days=300)) & (fd > this - pd.Timedelta(days=430))]
            if not len(prev):
                continue
            off = (dates - this).days.to_numpy()
            hit = (np.abs(off) <= FEST_ALIGN_DAYS) & (np.abs(off) < best)
            if hit.any():
                out = out.where(~hit, prev[-1] + pd.to_timedelta(off, unit="D"))
                best = np.where(hit, np.abs(off), best)
    return pd.DatetimeIndex(out)


def _extra_columns(X: dict, daily: pd.DataFrame, origins: pd.DatetimeIndex, d: pd.DatetimeIndex, h: int,
                   roll30, roll90, ref7_ly, roll30_ly) -> None:
    month = d.month.to_numpy()
    wk = X["is_weekend"]
    X["tmax_x_weekend"] = X["tmax_ai"] * wk
    X["app_tmax_x_weekend"] = X["app_tmax_ai"] * wk
    for name, months in (("premonsoon", (3, 4, 5, 6)), ("monsoon", (7, 8, 9)), ("postmonsoon", (10, 11))):
        X[f"cdd_x_{name}"] = X["cdd_ai"] * np.isin(month, months)
    X["cdd_x_doy_sin"] = X["cdd_ai"] * X["doy_sin"]
    X["cdd_x_doy_cos"] = X["cdd_ai"] * X["doy_cos"]
    d_ly = last_year_date(d)
    o_ly = d_ly - pd.Timedelta(days=h)                    # the origin's position, a year earlier
    ly_week = daily[TGT].rolling(REF_DAYS, min_periods=4).mean()
    for k in TGT:
        ref = X[f"ref_{k}"]
        X[f"{k}_yoy7"] = ref / ref7_ly[k].to_numpy()
        X[f"{k}_yoy30"] = roll30[k].to_numpy() / roll30_ly[k].to_numpy()
        X[f"{k}_mom30_90"] = roll30[k].to_numpy() / roll90[k].to_numpy()
        s = daily[k]
        ly = s.reindex(d_ly).to_numpy()
        X[f"{k}_ly"] = ly / ref
        X[f"{k}_ly_shape"] = ly / ly_week[k].reindex(o_ly).to_numpy()


NONNEG = ("precip", "rain", "cloud", "rad", "wind", "gust", "cdd", "hdd", "fog")


def debias_lead(fc: pd.DataFrame, daily: pd.DataFrame, end: str, lead: int = 7, ref_lead: int = 6) -> tuple[pd.DataFrame, dict]:
    """Remove the extra offset of the archived lead-`lead` forecasts relative to lead `ref_lead`,
    per weather column: off = mean(fc_lead - observed) - mean(fc_ref - observed) over target dates
    up to `end` (training period only). Non-negative quantities are clipped at 0. Returns the
    corrected copy and the offsets."""
    out = fc.copy()
    offs = {}
    obs_idx = daily.loc[:end].index
    for c in [c for c in fc.columns if c not in ("date", "lead") and c in daily.columns]:
        obs = daily.loc[obs_idx, c]
        e = {}
        for ld in (lead, ref_lead):
            f = fc[fc["lead"] == ld].set_index("date")[c].reindex(obs_idx)
            e[ld] = (f - obs).mean()
        if pd.notna(e[lead]) and pd.notna(e[ref_lead]):
            offs[c] = float(e[lead] - e[ref_lead])
            m = out["lead"] == lead
            out.loc[m, c] = out.loc[m, c] - offs[c]
            if c.startswith(NONNEG) and not c.endswith("_anom"):
                out.loc[m, c] = out.loc[m, c].clip(lower=0)
    return out, offs


def reference_levels(daily: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """ref_k(t) = mean of target k over t-6..t (NaN-tolerant, needs >= 4 values)."""
    roll = daily[TGT].rolling(REF_DAYS, min_periods=4).mean()
    return roll.reindex(origins)


# ============================================================================ tabular samples
TARGET_BASES = ("ref7", "last", "lag7")


def make_tabular(daily: pd.DataFrame, fc: pd.DataFrame, origins, horizons=range(1, HORIZON + 1),
                 extra_holidays=None, wx_override: dict | None = None,
                 climatology=None, target_base: str = "ref7") -> pd.DataFrame:
    """Direct multi-horizon samples: one row per (origin, h).

    Columns: origin, h, date, feature columns, ref_<t>, y_<t> (ratio), actual_<t> (level).
    `wx_override`: {h: DataFrame indexed by origin with WX_FEATS} replaces forecast weather
    (live forecasts / what-if scenarios). `climatology(dates) -> DataFrame[WX_FEATS]` supplies
    weather for horizons beyond the 7-day forecast archive (wx_source 3); used only by the
    long-lead model in the Grid-India fairness audit.

    target_base: what the label is divided by. "ref7" (default) = mean of the last 7 days at the
    origin; "last" = the origin day itself (day-over-day ratio); "lag7" = the same weekday one week
    before the target day (week-over-week ratio; known at the origin for h <= 7). The features are
    the same for every base; only ref_<t> and y_<t> change."""
    assert target_base in TARGET_BASES, target_base
    origins = pd.DatetimeIndex(pd.to_datetime(origins))
    fcm = _fc_matrix(fc) if len(fc) else {}
    refs = reference_levels(daily, origins)
    roll30 = daily[TGT].rolling(30, min_periods=15).mean().reindex(origins)
    std7 = daily[TGT].rolling(7, min_periods=4).std().reindex(origins)
    std30 = daily[TGT].rolling(30, min_periods=15).std().reindex(origins)
    wxref = daily[["tmax_ai", "cdd_ai", "app_tmax_ai"]].rolling(REF_DAYS, min_periods=4).mean().reindex(origins)
    ref_hol = daily["is_holiday"].rolling(REF_DAYS, min_periods=1).sum().reindex(origins)
    ref_imp = daily["is_imputed"].astype(float).rolling(REF_DAYS, min_periods=1).max().reindex(origins)
    roll90 = daily[TGT].rolling(90, min_periods=45).mean().reindex(origins)
    ref7_ly = daily[TGT].rolling(REF_DAYS, min_periods=4).mean().reindex(origins - pd.Timedelta(days=364))
    roll30_ly = daily[TGT].rolling(30, min_periods=15).mean().reindex(origins - pd.Timedelta(days=364))

    all_dates = pd.DatetimeIndex(sorted({o + pd.Timedelta(days=h) for o in origins for h in horizons}))
    cal = calendar_features(all_dates, extra_holidays=extra_holidays)[CAL_FEATS]

    # weather for every horizon first (the 3-day rain sum of day h needs days h-2..h)
    horizons = list(horizons)
    wx_h, src_h = {}, {}
    for h in sorted(set(horizons) | {j for h in horizons for j in (h - 1, h - 2) if j >= 1}):
        if wx_override is not None and h in wx_override:
            wx_h[h] = wx_override[h].reindex(origins).reindex(columns=WX_FEATS)
            src_h[h] = np.zeros(len(origins), np.int8)
        elif h > HORIZON and climatology is not None:
            wx_h[h] = climatology(origins + pd.Timedelta(days=h)).set_axis(origins)
            src_h[h] = np.full(len(origins), 3, np.int8)
        else:
            wx_h[h], src_h[h] = weather_known_at(fc, daily, origins, h, fcm)
    obs_precip = {u: daily[f"precip_{u}"] if f"precip_{u}" in daily else pd.Series(dtype=float)
                  for u in ["ai"] + REGIONS}

    frames = []
    for h in horizons:
        d = origins + pd.Timedelta(days=h)
        X = {"origin": origins.to_numpy(), "h": h, "date": d.to_numpy()}   # dict -> one DataFrame (fast)
        for k in TGT:
            ref = refs[k].to_numpy()
            s = daily[k]
            X[f"ref_{k}"] = ref
            X[f"{k}_last"] = s.reindex(origins).to_numpy() / ref
            for lag in (7, 14, 364):
                # a lag is only usable if that day is at or before the origin (matters for h > 7)
                X[f"{k}_lag{lag}"] = (s.reindex(d - pd.Timedelta(days=lag)).to_numpy() / ref
                                      if lag >= h else np.nan)
            X[f"{k}_roll30"] = roll30[k].to_numpy() / ref
            X[f"{k}_std7"] = std7[k].to_numpy() / ref
            X[f"{k}_std30"] = std30[k].to_numpy() / ref
        wx, src = wx_h[h], src_h[h]
        for c in WX_FEATS:
            X[c] = wx[c].to_numpy()
        X["wx_source"] = src
        # rain over d-2..d: observed for days <= origin, forecast (as known at origin) after
        for u in ["ai"] + REGIONS:
            tot = np.zeros(len(origins))
            for j in (h - 2, h - 1, h):
                v = (wx_h[j][f"precip_{u}"].to_numpy() if j >= 1
                     else obs_precip[u].reindex(origins + pd.Timedelta(days=j)).to_numpy())
                tot = tot + v
            X[f"rain3_{u}"] = tot
        for c in ["tmax_ai", "cdd_ai", "app_tmax_ai"]:
            X[f"ref_{c}"] = wxref[c].to_numpy()
            X[f"d_{c}"] = X[c] - X[f"ref_{c}"]
        X["ref_holidays"] = ref_hol.to_numpy()
        X["ref_imputed"] = ref_imp.to_numpy()
        calv = cal.reindex(d)
        for c in CAL_FEATS:
            X[c] = calv[c].to_numpy()
        _extra_columns(X, daily, origins, d, h, roll30, roll90, ref7_ly, roll30_ly)
        # target base (features above always use the 7-day reference level)
        if target_base != "ref7":
            for k in TGT:
                base = (daily[k].reindex(origins) if target_base == "last"
                        else daily[k].reindex(d - pd.Timedelta(days=7)))
                X[f"ref_{k}"] = base.to_numpy()
        # labels (only for evaluation/training; never used as features)
        for k in TGT:
            act = daily[k].reindex(d).to_numpy()
            X[f"actual_{k}"] = act
            X[f"y_{k}"] = act / X[f"ref_{k}"]
        frames.append(pd.DataFrame(X))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["origin", "h"]).reset_index(drop=True)


# ============================================================================ feature sets
# Which weather columns a model gets. "v1" = the previous round's inputs; "compact" adds the most
# relevant cold/fog, rain, cloud, humidity, wind and anomaly features; "full" = everything.
V1_WX = ([f"{v}_ai" for v in ("tmax", "tmean", "app_tmax", "rh_mean", "precip", "rad", "cdd")]
         + [f"tmax_{r}" for r in REGIONS] + [f"cdd_{r}" for r in REGIONS])
COMPACT_WX = V1_WX + ["tmin_ai", "hdd_ai", "tmin_nr", "hdd_nr", "fog_nr", "cloud_day_ai", "dew_mean_ai",
                      "heat_index_ai", "gust_max_ai", "rain3_ai", "tmax_ai_anom", "tmin_nr_anom",
                      "precip_ai_anom", "cloud_day_ai_anom"]
FEATURE_SETS = {"v1": V1_WX, "compact": COMPACT_WX, "full": WX_FEATS + RAIN3_COLS}
DEFAULT_SET = "full"


def model_wx(feature_set: str = DEFAULT_SET) -> list[str]:
    return list(FEATURE_SETS[feature_set])


def tabular_feature_cols(df: pd.DataFrame, feature_set: str = DEFAULT_SET, extra=()) -> list[str]:
    """Model inputs: everything except ids, labels and reference levels, the weather columns not
    in `feature_set`, and the EXTRA_GROUPS not selected in `extra`."""
    drop = {"origin", "date"} | {c for c in df.columns if c.startswith(("y_", "actual_", "ref_solar",
                                                                          "ref_evening", "ref_energy"))}
    keep_extra = {c for g in extra for c in EXTRA_GROUPS[g]}
    drop |= {c for c in EXTRA_COLS if c not in keep_extra}
    keep_wx = set(model_wx(feature_set))
    drop |= {c for c in WX_FEATS + RAIN3_COLS if c not in keep_wx}
    return [c for c in df.columns if c not in drop]


# ============================================================================ sequence samples
ENC_CAL = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_sunday", "is_saturday", "is_holiday",
           "is_festival", "diwali_window", "is_bridge"]
DEC_CAL = ENC_CAL + ["days_to_festival", "days_from_festival", "fest_diwali", "fest_holi"]
DEC_LAGS = [f"{k}_{l}" for k in TGT for l in ("lag7", "lag14", "lag364")]


def enc_feats(feature_set: str = DEFAULT_SET) -> list[str]:
    return [f"{k}_r" for k in TGT] + [f"{k}_mask" for k in TGT] + model_wx(feature_set) + ENC_CAL + ["is_imputed"]


def dec_feats(feature_set: str = DEFAULT_SET) -> list[str]:
    return (model_wx(feature_set) + ["wx_source", "d_tmax_ai", "d_cdd_ai", "d_app_tmax_ai", "ref_holidays"]
            + DEC_LAGS + DEC_CAL + ["h_frac"])


ENC_FEATS = enc_feats()
DEC_FEATS = dec_feats()


def make_sequences(daily: pd.DataFrame, fc: pd.DataFrame, origins, enc_days: int,
                   extra_holidays=None, wx_override: dict | None = None,
                   feature_set: str = DEFAULT_SET) -> dict[str, np.ndarray]:
    """Encoder/decoder tensors for each origin.

    enc  [N, enc_days, len(ENC_FEATS)]  days origin-enc_days+1 .. origin
    dec  [N, HORIZON,  len(DEC_FEATS)]  days origin+1 .. origin+HORIZON (known-future inputs)
    y    [N, HORIZON, 3] ratio targets (NaN where unknown), ref [N, 3]
    """
    origins = pd.DatetimeIndex(pd.to_datetime(origins))
    refs = reference_levels(daily, origins).to_numpy()
    N = len(origins)
    wx_cols = model_wx(feature_set)
    dfe = dec_feats(feature_set)
    enc = np.zeros((N, enc_days, len(enc_feats(feature_set))), np.float32)
    cal_hist = daily[ENC_CAL]
    for i, t in enumerate(origins):
        days = pd.date_range(t - pd.Timedelta(days=enc_days - 1), t)
        w = daily.reindex(days)
        r = w[TGT].to_numpy() / refs[i]
        mask = ~np.isnan(r)
        parts = [np.nan_to_num(r, nan=0.0), mask.astype(np.float32),
                 w.reindex(columns=wx_cols).to_numpy(), cal_hist.reindex(days).to_numpy(),
                 w[["is_imputed"]].astype(float).to_numpy()]
        enc[i] = np.concatenate(parts, 1)
    enc = np.nan_to_num(enc, nan=0.0)

    tab = make_tabular(daily, fc, origins, extra_holidays=extra_holidays, wx_override=wx_override)
    tab["h_frac"] = tab["h"] / HORIZON
    tab[DEC_LAGS] = tab[DEC_LAGS].fillna(1.0)  # ratio lags: missing (e.g. lag364 in year 1) -> neutral
    dec = tab[dfe].to_numpy(np.float32).reshape(N, HORIZON, len(dfe))
    y = tab[[f"y_{k}" for k in TGT]].to_numpy(np.float32).reshape(N, HORIZON, len(TGT))
    return {"enc": enc, "dec": np.nan_to_num(dec, nan=0.0), "y": y, "ref": refs.astype(np.float32),
            "origins": origins.to_numpy()}


def valid_origins(daily: pd.DataFrame, start=None, end=None, need_targets: bool = True,
                  min_history: int = 30) -> pd.DatetimeIndex:
    """Origins with a reference level for all targets and >= min_history days of data before.
    If need_targets, also require at least one target observed in t+1..t+7."""
    refs = reference_levels(daily, daily.index)
    ok = refs.notna().all(axis=1)
    first = daily.index.min() + pd.Timedelta(days=min_history)
    idx = daily.index[ok & (daily.index >= first)]
    if need_targets:
        idx = idx[idx + pd.Timedelta(days=1) <= daily.index.max()]
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    daily, fc = build_daily_features()
    print(daily.tail(3).T.to_string())
