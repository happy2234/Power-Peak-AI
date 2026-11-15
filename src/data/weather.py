"""Open-Meteo weather for major Indian demand centres (free, no API key).

Three sources, all cached under data/raw/weather/:
  * observed  — archive API, daily values (ERA5-based reanalysis)      obs_<city>.parquet
  * archived forecasts — Previous-Runs API: what the forecast said 1..7 days before each
    date (hourly, aggregated here to daily). This lets training/backtests use *real forecast
    error* instead of perfect hindsight weather.                         fc_<city>.parquet
  * live forecast — forecast API, next days                            (not cached; small)

Daily aggregates (Asia/Kolkata days): tmax, tmean, app_tmax (apparent temperature ~ heat
index), rh_mean, precip, rad (shortwave radiation, MJ/m²), plus WX_EXTRA (tmin, dew point,
cloud cover, wind, gusts, morning dew-point depression for fog) in fcx_/obsx_ caches.
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd
import requests

from src.config import WEATHER_DIR

log = logging.getLogger(__name__)

# city -> (lat, lon, grid region)
CITIES: dict[str, tuple[float, float, str]] = {
    "delhi": (28.61, 77.21, "nr"),
    "lucknow": (26.85, 80.95, "nr"),
    "jaipur": (26.91, 75.79, "nr"),
    "ahmedabad": (23.02, 72.57, "wr"),
    "mumbai": (19.08, 72.88, "wr"),
    "nagpur": (21.15, 79.09, "wr"),
    "chennai": (13.08, 80.27, "sr"),
    "bengaluru": (12.97, 77.59, "sr"),
    "hyderabad": (17.39, 78.49, "sr"),
    "kolkata": (22.57, 88.36, "er"),
    "guwahati": (26.14, 91.74, "ner"),  # added so every grid region has a temperature
}
WX_VARS = ["tmax", "tmean", "app_tmax", "rh_mean", "precip", "rad"]
MAX_LEAD = 7

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST = "https://api.open-meteo.com/v1/forecast"
_DAILY = ("temperature_2m_max,temperature_2m_mean,apparent_temperature_max,"
          "relative_humidity_2m_mean,precipitation_sum,shortwave_radiation_sum")
_DAILY_MAP = {"temperature_2m_max": "tmax", "temperature_2m_mean": "tmean",
              "apparent_temperature_max": "app_tmax", "relative_humidity_2m_mean": "rh_mean",
              "precipitation_sum": "precip", "shortwave_radiation_sum": "rad"}
_HOURLY = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
           "shortwave_radiation"]

_session = requests.Session()
_session.headers["User-Agent"] = "india-power-forecast/0.2 (research)"


REQUESTS = 0        # HTTP requests sent to Open-Meteo by this process (the pipeline logs it)


def _get(url: str, params: dict, retries: int = 4, pause: float = 1.0) -> dict:
    global REQUESTS
    for attempt in range(retries):
        try:
            REQUESTS += 1
            r = _session.get(url, params=params, timeout=120)
            if r.status_code == 429:  # rate limit: back off generously
                wait = 60 * (attempt + 1)
                log.warning("Open-Meteo rate limit; sleeping %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("Open-Meteo request failed (%s); retrying", e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _daily_frame(js: dict) -> pd.DataFrame:
    d = js["daily"]
    df = pd.DataFrame({_DAILY_MAP[k]: v for k, v in d.items() if k in _DAILY_MAP})
    df.insert(0, "date", pd.to_datetime(d["time"]))
    return df


def _chunks(start: dt.date, end: dt.date, days: int):
    s = start
    while s <= end:
        e = min(end, s + dt.timedelta(days=days - 1))
        yield s, e
        s = e + dt.timedelta(days=1)


# ----------------------------------------------------------------------------- observed
def fetch_observed(city: str, start: dt.date, end: dt.date, refresh_days: int = 10) -> pd.DataFrame:
    """Daily observed weather, cached; re-fetches the last `refresh_days` (archive revisions)."""
    lat, lon, _ = CITIES[city]
    path = WEATHER_DIR / f"obs_{city}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["date"])
    have_from = old["date"].min().date() if len(old) else None
    have_until = old["date"].max().date() if len(old) else None
    fetch_from = start if have_until is None else max(start, have_until - dt.timedelta(days=refresh_days))
    ranges = [(fetch_from, end)]
    if have_from is not None and have_from > start:  # cache starts late: back-fill history too
        ranges.insert(0, (start, have_from - dt.timedelta(days=1)))
    frames = [old]
    chunks = [c for a, b in ranges if a <= b for c in _chunks(a, b, 800)]
    if chunks:
        for s, e in chunks:
            js = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
                                "daily": _DAILY, "timezone": "Asia/Kolkata"})
            frames.append(_daily_frame(js))
        new = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
        new = new.dropna(subset=["tmax"]).sort_values("date").reset_index(drop=True)
        WEATHER_DIR.mkdir(parents=True, exist_ok=True)
        new.to_parquet(path, index=False)
        old = new
    out = old[(old["date"] >= pd.Timestamp(start)) & (old["date"] <= pd.Timestamp(end))]
    return out.assign(city=city)


# ----------------------------------------------------------------------------- archived fc
def _aggregate_hourly(js: dict) -> pd.DataFrame:
    """Hourly previous-run forecasts -> daily, long format (date, lead, WX_VARS...)."""
    h = js["hourly"]
    t = pd.to_datetime(h["time"])
    rows = []
    for lead in range(1, MAX_LEAD + 1):
        sfx = f"_previous_day{lead}"
        f = pd.DataFrame({v: h.get(v + sfx) for v in _HOURLY}, index=t).astype(float)
        g = f.groupby(f.index.normalize())
        daily = pd.DataFrame({
            "tmax": g["temperature_2m"].max(),
            "tmean": g["temperature_2m"].mean(),
            "app_tmax": g["apparent_temperature"].max(),
            "rh_mean": g["relative_humidity_2m"].mean(),
            "precip": g["precipitation"].sum(min_count=20),
            "rad": g["shortwave_radiation"].sum(min_count=20) * 3600 / 1e6,  # W/m² h -> MJ/m²
        })
        # a day only counts if (nearly) all 24 hours are present
        complete = g["temperature_2m"].count() >= 20
        daily = daily[complete]
        daily.index.name = "date"
        rows.append(daily.reset_index().assign(lead=lead))
    return pd.concat(rows, ignore_index=True)


def fetch_archived_forecasts(city: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Forecasts issued `lead` days before each date, lead 1..7. Cached incrementally."""
    lat, lon, _ = CITIES[city]
    path = WEATHER_DIR / f"fc_{city}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["date", "lead"])
    have_from = old["date"].min().date() if len(old) else None
    have_until = old["date"].max().date() if len(old) else None
    fetch_from = start if have_until is None else max(start, have_until - dt.timedelta(days=3))
    ranges = [(fetch_from, end)]
    if have_from is not None and have_from > start:  # cache starts late: back-fill history too
        ranges.insert(0, (start, have_from - dt.timedelta(days=1)))
    frames = [old]
    hourly = ",".join(f"{v}_previous_day{l}" for v in _HOURLY for l in range(1, MAX_LEAD + 1))
    chunks = [c for a, b in ranges if a <= b for c in _chunks(a, b, 92)]
    for s, e in chunks:
        js = _get(PREVIOUS, {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
                             "hourly": hourly, "timezone": "Asia/Kolkata"}, pause=8.0)
        frames.append(_aggregate_hourly(js))
        log.info("  %s archived forecasts %s..%s", city, s, e)
    new = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "lead"], keep="last")
    new = new.sort_values(["date", "lead"]).reset_index(drop=True)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    out = new[(new["date"] >= pd.Timestamp(start)) & (new["date"] <= pd.Timestamp(end))]
    return out.assign(city=city)


# ----------------------------------------------------------------------------- extra variables
# Added for all-weather features: night/winter temperature, dew point, clouds, wind/storms and a
# morning fog proxy. Kept in separate caches (fcx_/obsx_) so the original caches stay valid.
# Open-Meteo has no archived visibility forecast, so fog is proxied by the early-morning
# dew-point depression (T - Td, 03:00-08:00 local): fog forms when the air is close to saturation.
# Previous-run (archived) forecasts of dew point, clouds and wind only exist from early 2024;
# before that these columns are NaN (temperature-based tmin exists from 2023).
WX_EXTRA = ["tmin", "dew_mean", "cloud_mean", "cloud_day", "wind_max", "gust_max", "dpd_morning"]
_HOURLY_X = ["temperature_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m", "wind_gusts_10m"]


def _aggregate_extra(f: pd.DataFrame) -> pd.DataFrame:
    """Hourly frame (columns _HOURLY_X, local-time index) -> daily WX_EXTRA."""
    f = f.astype(float)
    day = f.index.normalize()
    hour = f.index.hour
    g = f.groupby(day)
    dpd = (f["temperature_2m"] - f["dew_point_2m"]).where((hour >= 3) & (hour <= 8))
    sun = f["cloud_cover"].where((hour >= 8) & (hour <= 17))
    out = pd.DataFrame({
        "tmin": g["temperature_2m"].min(),
        "dew_mean": g["dew_point_2m"].mean(),
        "cloud_mean": g["cloud_cover"].mean(),
        "cloud_day": sun.groupby(day).mean(),
        "wind_max": g["wind_speed_10m"].max(),
        "gust_max": g["wind_gusts_10m"].max(),
        "dpd_morning": dpd.groupby(day).mean(),
    })
    complete = g["temperature_2m"].count() >= 20
    for c in ["dew_mean", "cloud_mean", "cloud_day", "wind_max", "gust_max", "dpd_morning"]:
        src = {"dew_mean": "dew_point_2m", "cloud_mean": "cloud_cover", "cloud_day": "cloud_cover",
               "wind_max": "wind_speed_10m", "gust_max": "wind_gusts_10m", "dpd_morning": "dew_point_2m"}[c]
        out.loc[g[src].count() < 20, c] = np.nan
    out = out[complete]
    out.index.name = "date"
    return out


def _save(frames, key, path) -> pd.DataFrame:
    new = pd.concat(frames, ignore_index=True).drop_duplicates(key, keep="last")
    new = new.sort_values(key).reset_index(drop=True)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def fetch_extra(city: str, start: dt.date, end: dt.date, kind: str) -> pd.DataFrame:
    """kind='fc': archived forecasts (lead 1..7) -> fcx_<city>; kind='obs': reanalysis -> obsx_<city>.
    Cached incrementally like the main caches."""
    lat, lon, _ = CITIES[city]
    key = ["date", "lead"] if kind == "fc" else ["date"]
    path = WEATHER_DIR / f"{kind}x_{city}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=key)
    have_from = old["date"].min().date() if len(old) else None
    have_until = old["date"].max().date() if len(old) else None
    back = 3 if kind == "fc" else 10
    fetch_from = start if have_until is None else max(start, have_until - dt.timedelta(days=back))
    ranges = [(fetch_from, end)]
    if have_from is not None and have_from > start:
        ranges.insert(0, (start, have_from - dt.timedelta(days=1)))
    frames = [old] if len(old) else []
    size = 92 if kind == "fc" else 400
    for s, e in [c for a, b in ranges if a <= b for c in _chunks(a, b, size)]:
        if kind == "fc":
            hourly = ",".join(f"{v}_previous_day{l}" for v in _HOURLY_X for l in range(1, MAX_LEAD + 1))
            js = _get(PREVIOUS, {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
                                 "hourly": hourly, "timezone": "Asia/Kolkata"}, pause=8.0)
            h = js["hourly"]
            t = pd.to_datetime(h["time"])
            for lead in range(1, MAX_LEAD + 1):
                f = pd.DataFrame({v: h.get(f"{v}_previous_day{lead}") for v in _HOURLY_X}, index=t)
                frames.append(_aggregate_extra(f).reset_index().assign(lead=lead))
        else:
            js = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
                                "hourly": ",".join(_HOURLY_X), "timezone": "Asia/Kolkata"}, pause=2.0)
            h = js["hourly"]
            f = pd.DataFrame({v: h.get(v) for v in _HOURLY_X}, index=pd.to_datetime(h["time"]))
            frames.append(_aggregate_extra(f).reset_index())
        log.info("  %s extra %s %s..%s", city, kind, s, e)
        _save(frames, key, path)          # after every chunk: an interrupted run resumes here
    new = _save(frames, key, path) if frames else old
    return new[(new["date"] >= pd.Timestamp(start)) & (new["date"] <= pd.Timestamp(end))].assign(city=city)


# ----------------------------------------------------------------------------- live fc
def fetch_live_forecast(city: str, days: int = 8, past_days: int = 3) -> pd.DataFrame:
    """Daily forecast for today-past_days .. today+days-1 (issued now). Past days matter because
    the forecast origin (last published report) is usually 1-2 days before today."""
    lat, lon, _ = CITIES[city]
    js = _get(FORECAST, {"latitude": lat, "longitude": lon, "daily": _DAILY, "timezone": "Asia/Kolkata",
                         "hourly": ",".join(_HOURLY_X), "forecast_days": days, "past_days": past_days},
              pause=0.3)
    h = js["hourly"]
    f = pd.DataFrame({v: h.get(v) for v in _HOURLY_X}, index=pd.to_datetime(h["time"]))
    x = _aggregate_extra(f).reset_index()          # same daily aggregation as the archive
    return _daily_frame(js).merge(x, on="date", how="left").assign(city=city)


# ----------------------------------------------------------------------------- all cities
def update_all(start: dt.date, end: dt.date, forecasts: bool = True) -> None:
    for city in CITIES:
        log.info("weather: %s", city)
        fetch_observed(city, start, end)
        fetch_extra(city, start, end, "obs")
        if forecasts:
            fetch_archived_forecasts(city, start, end)
            fetch_extra(city, start, end, "fc")


def load_observed() -> pd.DataFrame:
    frames = [pd.read_parquet(p).assign(city=p.stem[4:]) for p in sorted(WEATHER_DIR.glob("obs_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_archived_forecasts() -> pd.DataFrame:
    frames = [pd.read_parquet(p).assign(city=p.stem[3:]) for p in sorted(WEATHER_DIR.glob("fc_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_extra(kind: str) -> pd.DataFrame:
    """Extra variables (WX_EXTRA) for all cities; kind 'obs' or 'fc'."""
    frames = [pd.read_parquet(p).assign(city=p.stem[len(kind) + 2:])
              for p in sorted(WEATHER_DIR.glob(f"{kind}x_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


ALL_VARS = WX_VARS + WX_EXTRA


def regional(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """City rows -> one row per `keys`, columns <var>_<region> (mean over the region's cities).
    Every variable in ALL_VARS gets a column (NaN where the source has none)."""
    df = df.assign(region=df["city"].map(lambda c: CITIES[c][2]))
    df = df.reindex(columns=list(df.columns) + [v for v in ALL_VARS if v not in df.columns])
    g = df.groupby(keys + ["region"])[ALL_VARS].mean().unstack("region")
    g.columns = [f"{v}_{r}" for v, r in g.columns]
    regions = sorted({c[2] for c in CITIES.values()})
    g = g.reindex(columns=[f"{v}_{r}" for v in ALL_VARS for r in regions])  # all regions present
    return g.reset_index()


def city_wide(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """City rows -> one row per `keys`, columns <var>_<city>."""
    df = df.reindex(columns=list(df.columns) + [v for v in ALL_VARS if v not in df.columns])
    g = df.set_index(keys + ["city"])[ALL_VARS].unstack("city")
    g.columns = [f"{v}_{c}" for v, c in g.columns]
    return g.reset_index()


def load_observed_all() -> pd.DataFrame:
    """Observed base + extra variables, one row per (city, date)."""
    obs, x = load_observed(), load_extra("obs")
    if obs.empty or x.empty:
        return obs
    return obs.merge(x, on=["city", "date"], how="left")


def load_forecasts_all() -> pd.DataFrame:
    """Archived forecasts, base + extra variables, one row per (city, date, lead)."""
    fc, x = load_archived_forecasts(), load_extra("fc")
    if fc.empty or x.empty:
        return fc
    x["lead"] = x["lead"].astype(int)
    fc["lead"] = fc["lead"].astype(int)
    return fc.merge(x, on=["city", "date", "lead"], how="left")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    end = dt.date.today() - dt.timedelta(days=1)
    if "--extra-only" in sys.argv:  # one-off back-fill of the extra variables
        for c in CITIES:
            log.info("weather extra: %s", c)
            fetch_extra(c, dt.date(2023, 3, 1), end, "obs")
            fetch_extra(c, dt.date(2023, 3, 1), end, "fc")
        sys.exit(0)
    update_all(dt.date(2023, 3, 1), end, forecasts="--no-fc" not in sys.argv)
    obs, fc = load_observed(), load_archived_forecasts()
    print("observed", obs.shape, obs.date.min(), obs.date.max())
    print("archived forecasts", fc.shape, fc.date.min(), fc.date.max())
    print("fc coverage by lead:", fc.dropna(subset=["tmax"]).groupby("lead").size().to_dict())
    _ = np
