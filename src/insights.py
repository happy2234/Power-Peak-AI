"""Unique features built on top of the forecasts:

  a) AI vs Grid-India scorecard
  b) record-peak probability (correlated Monte Carlo through P10/P50/P90) + calibration
  c) forecast explanations (LightGBM SHAP -> grouped GW drivers -> sentence; TFT importances)
  d) what-if simulator (temperature changes per region, extra holidays)
  e) evening-peak stress (gap evening - solar, top-5 % alerts)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED, REGIONS, TARGETS

TGT = list(TARGETS)
Z90 = 1.2815515655446004  # standard normal 90th percentile
REGION_NAMES = {"nr": "North", "wr": "West", "sr": "South", "er": "East", "ner": "North-East",
                "all": "all India"}


# ============================================================================ distributions
def split_normal_sigmas(p10, p50, p90):
    lo = np.maximum((np.asarray(p50) - np.asarray(p10)) / Z90, 1e-9)
    hi = np.maximum((np.asarray(p90) - np.asarray(p50)) / Z90, 1e-9)
    return lo, hi


def to_z(actual, p10, p50, p90):
    lo, hi = split_normal_sigmas(p10, p50, p90)
    d = np.asarray(actual) - np.asarray(p50)
    return np.where(d >= 0, d / hi, d / lo)


def residual_correlation(bt: pd.DataFrame, model: str, shrink: float = 0.2) -> np.ndarray:
    """21x21 correlation of standardized residuals (7 days x 3 targets) across origins,
    shrunk toward identity and projected to positive semi-definite."""
    b = bt[(bt["model"] == model)].dropna(subset=["actual", "p10", "p90"]).copy()
    b["z"] = to_z(b["actual"], b["p10"], b["p50"], b["p90"])
    b["slot"] = (b["h"] - 1) * len(TGT) + b["target"].map({t: i for i, t in enumerate(TGT)})
    w = b.pivot_table(index="origin", columns="slot", values="z").dropna()
    if len(w) < 30:
        return np.eye(HORIZON * len(TGT))
    c = np.corrcoef(w.to_numpy().T)
    c = (1 - shrink) * c + shrink * np.eye(len(c))
    vals, vecs = np.linalg.eigh(c)
    c = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def load_corr() -> np.ndarray:
    p = PROCESSED / "mc_corr.npy"
    return np.load(p) if p.exists() else np.eye(HORIZON * len(TGT))


def sample_paths(pred: pd.DataFrame, n: int = 20000, corr: np.ndarray | None = None,
                 seed: int = 0) -> dict[str, np.ndarray]:
    """pred: one origin, rows (h, target, p10, p50, p90). Returns target -> [n, 7] samples."""
    corr = load_corr() if corr is None else corr
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr + 1e-9 * np.eye(len(corr)))
    z = rng.standard_normal((n, len(corr))) @ L.T
    out = {}
    for i, t in enumerate(TGT):
        p = pred[pred["target"] == t].set_index("h").reindex(range(1, HORIZON + 1))
        lo, hi = split_normal_sigmas(p["p10"], p["p50"], p["p90"])
        zt = z[:, [(h - 1) * len(TGT) + i for h in range(1, HORIZON + 1)]]
        out[t] = p["p50"].to_numpy() + np.where(zt >= 0, zt * hi, zt * lo)
    return out


# ============================================================================ (b) record peak
def records_at(daily: pd.DataFrame, origin) -> dict[str, float]:
    """All-time (since Apr 2023, the start of our data) and calendar-year-to-date record of the
    daily peak = max(solar-hour peak, evening peak), known at the origin."""
    origin = pd.Timestamp(origin)
    peak = daily[["solar_peak", "evening_peak"]].max(axis=1)
    hist = peak[peak.index <= origin]
    year = hist[hist.index >= pd.Timestamp(origin.year, 1, 1)]
    return {"all_time": float(hist.max()), "all_time_date": hist.idxmax(),
            "year": float(year.max()) if len(year) else float("nan"),
            "year_date": year.idxmax() if len(year) else None}


def record_probability(pred: pd.DataFrame, records: dict, n: int = 20000, corr=None) -> dict:
    s = sample_paths(pred, n, corr)
    peak = np.maximum(s["solar_peak"], s["evening_peak"])        # [n, 7] daily peak
    mx = peak.max(1)
    out = {"p_all_time": float((mx > records["all_time"]).mean()),
           "p_year": float((mx > records["year"]).mean()) if not np.isnan(records["year"]) else float("nan"),
           "per_day_all_time": (peak > records["all_time"]).mean(0).tolist(),
           "per_day_year": (peak > records["year"]).mean(0).tolist(),
           "peak_p50": float(np.median(mx)), "peak_p90": float(np.quantile(mx, 0.9))}
    return out


def record_calibration(bt: pd.DataFrame, daily: pd.DataFrame, model: str, corr=None,
                       n: int = 4000) -> pd.DataFrame:
    """For every backtest origin: predicted P(any of next 7 days beats the record) vs outcome."""
    corr = load_corr() if corr is None else corr
    b = bt[bt["model"] == model]
    peak = daily[["solar_peak", "evening_peak"]].max(axis=1)
    rows = []
    for o, g in b.groupby("origin"):
        if g["h"].nunique() < HORIZON or g["actual"].isna().any():
            continue
        rec = records_at(daily, o)
        pr = record_probability(g, rec, n=n, corr=corr)
        fut = peak.reindex(pd.date_range(o + pd.Timedelta(days=1), periods=HORIZON)).max()
        rows.append({"origin": o, "p_all_time": pr["p_all_time"], "hit_all_time": fut > rec["all_time"],
                     "p_year": pr["p_year"], "hit_year": fut > rec["year"]})
    return pd.DataFrame(rows)


def reliability(p: np.ndarray, y: np.ndarray, bins=(0, .02, .1, .25, .5, .75, .9, 1.0001)) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.digitize(p, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        m = b == i
        if m.any():
            rows.append({"bin": f"{bins[i]:.2f}–{min(bins[i + 1], 1):.2f}", "n": int(m.sum()),
                         "mean_predicted": p[m].mean(), "observed_freq": y[m].mean()})
    return pd.DataFrame(rows)


def brier(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bs = float(np.mean((p - y) ** 2))
    clim = float(np.mean((y.mean() - y) ** 2))
    return {"brier": bs, "brier_climatology": clim, "skill": 1 - bs / clim if clim > 0 else float("nan"),
            "events": int(y.sum()), "n": len(y)}


# ============================================================================ (a) scorecard
def scorecard(bt: pd.DataFrame, model: str, target: str) -> pd.DataFrame:
    """Daily head-to-head (our forecast = mean of our 1..7-day-ahead forecasts for that day)."""
    from src.evaluate import head_to_head

    j = head_to_head(bt, model).query("target == @target").sort_values("date").reset_index(drop=True)
    j["running_win_rate"] = j["ours_wins"].expanding().mean()
    j["cum_ae_ours"] = j["ae_ours"].cumsum()
    j["cum_ae_gi"] = j["ae_gi"].cumsum()
    j["advantage"] = j["ae_gi"] - j["ae_ours"]   # > 0: we were closer
    return j


# ============================================================================ (c) explanations
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEST_LABEL = {"fest_diwali": "Diwali", "fest_holi": "Holi", "fest_eid_ul_fitr": "Eid al-Fitr",
              "fest_eid_ul_adha": "Eid al-Adha", "fest_dussehra": "Dussehra",
              "fest_durga_puja": "Durga Puja", "fest_pongal": "Pongal",
              "fest_ganesh_chaturthi": "Ganesh Chaturthi", "fest_christmas": "Christmas"}


WX_KIND = {  # weather variable -> driver kind
    "tmax": "heat", "tmean": "heat", "app_tmax": "heat", "cdd": "heat", "heat_index": "humidity",
    "tmin": "cold", "hdd": "cold", "fog": "fog", "precip": "rain", "rain_day": "rain", "rain3": "rain",
    "cloud_mean": "clouds", "cloud_day": "clouds", "rad": "clouds", "rh_mean": "humidity",
    "dew_mean": "humidity", "wind_max": "wind", "gust_max": "wind"}
KIND_LABEL = {"heat": "Heat", "cold": "Cold morning", "fog": "Fog", "rain": "Rain", "clouds": "Clouds",
              "humidity": "Humidity", "wind": "Wind/storm"}


def _weather_group(f: str) -> str | None:
    """'precip_sr_anom' -> 'wx_rain_sr'; 'd_tmax_ai' -> 'wx_heat_all'; None if not weather."""
    g = f[:-5] if f.endswith("_anom") else f
    if g.startswith(("d_", "ref_")) and g.split("_", 1)[1].rsplit("_", 1)[0] in WX_KIND:
        g = g.split("_", 1)[1]
    if "_" not in g:
        return None
    var, unit = g.rsplit("_", 1)
    if var not in WX_KIND or unit not in REGIONS + ["ai"]:
        return None
    return f"wx_{WX_KIND[var]}_{'all' if unit == 'ai' else unit}"


def feature_group(f: str) -> str:
    wx = _weather_group(f)
    if wx:
        return wx
    if f in ("dow", "is_saturday", "is_sunday", "is_weekend", "dow_sin", "dow_cos"):
        return "weekday"
    if f.startswith("fest_") or f in ("diwali_window", "is_festival", "is_holiday", "is_national_holiday",
                                      "days_to_festival", "days_from_festival", "is_bridge"):
        return "festival"
    if f in ("month", "doy_sin", "doy_cos"):
        return "season"
    if f.endswith(("_lag7", "_lag14")):
        return "last weeks' pattern"
    if f.endswith("_lag364"):
        return "same day last year"
    if f.endswith(("_last", "_roll30", "_std7", "_std30")):
        return "recent trend"
    if f == "ref_holidays":
        return "holidays in the last week"
    return "other"


def _group_label(group: str, row: pd.Series) -> str:
    if group.startswith("wx_"):
        _, kind, unit = group.split("_", 2)
        where = "all India" if unit == "all" else f"{REGION_NAMES[unit]} India"
        if kind == "heat" and unit != "all":
            return f"Heat in {where}"
        return f"{KIND_LABEL[kind]} ({where})" if unit == "all" else f"{KIND_LABEL[kind]} in {where}"
    if group == "weekday":
        return DAY_NAMES[int(row["dow"])]
    if group == "festival":
        names = [FEST_LABEL[f] for f in FEST_LABEL if row.get(f, 0) == 1]
        if names:
            return " / ".join(names)
        if row.get("diwali_window", 0) == 1:
            return "Diwali week"
        if row.get("is_holiday", 0) == 1:
            return "holiday"
        if row.get("days_to_festival", 99) <= 3:
            return "run-up to a festival"
        return "festival calendar"
    return group


def weather_drivers(contrib: dict, unit: str, top: int = 3, min_abs: float = 0.05) -> list[str]:
    """{'Rain in South India': -2.1, ...} -> ['Rain in South India: -2.1 GW', ...] (weather only)."""
    kinds = tuple(KIND_LABEL.values())
    items = [(k, v) for k, v in contrib.items() if k.startswith(kinds) and abs(v) >= min_abs]
    items.sort(key=lambda kv: -abs(kv[1]))
    return [f"{k}: {'−' if v < 0 else '+'}{abs(v):.1f} {unit}" for k, v in items[:top]]


def explain_lgbm(model, tab: pd.DataFrame, target: str, top: int = 3) -> pd.DataFrame:
    """Per forecast row: grouped signed contributions (GW for peaks, MU for energy) + sentence."""
    sv = model.shap_contributions(tab, target)
    unit_div, unit = (1000, "GW") if target != "energy" else (1, "MU")
    groups = pd.Series({f: feature_group(f) for f in model.features})
    g = sv[model.features].T.groupby(groups).sum().T / unit_div       # rows x groups
    out = []
    for i, (idx, row) in enumerate(tab.iterrows()):
        contrib = g.loc[idx].drop(labels=["other"], errors="ignore")
        order = contrib.abs().sort_values(ascending=False).index[:top]
        drivers = [(_group_label(k, row), float(contrib[k])) for k in order]
        ref = row[f"ref_{target}"] / unit_div
        parts = ", ".join(f"{v:+.1f} {unit} {lab}" for lab, v in drivers)
        labelled: dict[str, float] = {}
        for k, v in contrib.items():
            lab = _group_label(k, row)
            labelled[lab] = labelled.get(lab, 0.0) + float(v)
        out.append({"date": row["date"], "target": target, "ref": ref,
                    "drivers": drivers, "contrib": labelled,
                    "sentence": f"{pd.Timestamp(row['date']):%a %d %b}: vs the recent level "
                                f"({ref:,.1f} {unit}) — {parts}."})
    return pd.DataFrame(out)


def tft_importance(weights: dict, enc_feats: list[str], dec_feats: list[str]) -> pd.DataFrame:
    """Mean variable-selection weights (unsigned importances) from a TFT forward pass."""
    rows = []
    for key, feats in (("enc_vars", enc_feats), ("dec_vars", dec_feats)):
        if key in weights:
            w = weights[key].reshape(-1, weights[key].shape[-1]).mean(0)
            rows += [{"part": "past" if key == "enc_vars" else "future", "feature": f,
                      "group": feature_group(f.replace("_r", "_last") if f.endswith("_r") else f),
                      "weight": float(v)} for f, v in zip(feats, w)]
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


# ============================================================================ (d) what-if
@dataclass
class Scenario:
    temp_delta: dict | None = None      # {"all": +2} or {"nr": +3, "sr": -1}  (degC)
    holidays: list | None = None        # dates to treat as holidays


def _region_weights() -> dict:
    return json.loads((PROCESSED / "region_weights.json").read_text())


def _normals() -> dict:
    from src.features.build import load_normals

    return load_normals()


def scenario_weather(raw: pd.DataFrame, temp_delta: dict | None, dates=None) -> pd.DataFrame:
    """Apply temperature deltas to raw regional weather and re-derive model features.
    `dates`: the calendar day each row describes (needed for the anomaly features; raw frames
    are indexed by forecast origin). Dew point is shifted with temperature, so humidity stays
    the same in relative terms and the heat index moves with the temperature."""
    from src.features.build import derive_weather

    r = raw.copy()
    for key, dT in (temp_delta or {}).items():
        regs = REGIONS if key == "all" else [key]
        for reg in regs:
            for v in ("tmax", "tmean", "app_tmax", "tmin", "dew_mean"):
                if f"{v}_{reg}" in r:
                    r[f"{v}_{reg}"] = r[f"{v}_{reg}"] + dT
    return derive_weather(r, _region_weights(), dates=dates, normals=_normals())


def base_raw_weather(daily, fc, origin) -> dict[int, pd.DataFrame]:
    """Raw regional weather for origin+h as known at the origin (archived forecasts)."""
    from src.features.build import WX_RAW, weather_known_at

    o = pd.DatetimeIndex([pd.Timestamp(origin)])
    return {h: weather_known_at(fc, daily, o, h, cols=WX_RAW)[0] for h in range(1, HORIZON + 1)}


def what_if(model, daily, fc, origin, scenario: Scenario, raw_weather: dict | None = None) -> pd.DataFrame:
    """Base vs scenario forecast (LightGBM) for the 7 days after `origin`.
    raw_weather: {h: DataFrame (1 row, index=origin, WX_RAW)} - e.g. the live forecast."""
    from src.features.build import make_tabular
    from src.models.common import tab_to_long

    raw = raw_weather or base_raw_weather(daily, fc, origin)
    o = pd.Timestamp(origin)
    day = lambda h: [o + pd.Timedelta(days=h)]  # noqa: E731
    base_wx = {h: scenario_weather(r, None, day(h)) for h, r in raw.items()}
    scen_wx = {h: scenario_weather(r, scenario.temp_delta, day(h)) for h, r in raw.items()}
    tb = make_tabular(daily, fc, [origin], wx_override=base_wx)
    ts = make_tabular(daily, fc, [origin], wx_override=scen_wx, extra_holidays=scenario.holidays)
    b = tab_to_long(tb, model.predict(tb), "base")
    s = tab_to_long(ts, model.predict(ts), "scenario")
    m = b.merge(s, on=["date", "h", "target"], suffixes=("_base", "_scen"))
    m["delta_p50"] = m["p50_scen"] - m["p50_base"]
    return m[["date", "h", "target", "p10_base", "p50_base", "p90_base", "p10_scen", "p50_scen",
              "p90_scen", "delta_p50"]]


# ============================================================================ (e) evening stress
def evening_stress(pred: pd.DataFrame, daily: pd.DataFrame, origin, n: int = 20000, corr=None,
                   pct: float = 0.95) -> pd.DataFrame:
    """Per forecast day: solar & evening P50, gap distribution, top-5 % evening alert."""
    origin = pd.Timestamp(origin)
    hist = daily.loc[daily.index <= origin, "evening_peak"].dropna()
    thr = float(hist.quantile(pct))
    s = sample_paths(pred, n, corr)
    gap = s["evening_peak"] - s["solar_peak"]
    p = pred.pivot_table(index="h", columns="target", values="p50")
    out = pd.DataFrame({
        "date": [origin + pd.Timedelta(days=h) for h in range(1, HORIZON + 1)],
        "solar_p50": p["solar_peak"].to_numpy(), "evening_p50": p["evening_peak"].to_numpy(),
        "gap_p10": np.quantile(gap, 0.1, 0), "gap_p50": np.quantile(gap, 0.5, 0),
        "gap_p90": np.quantile(gap, 0.9, 0),
        "p_evening_above_solar": (gap > 0).mean(0),
        "p_evening_top5": (s["evening_peak"] > thr).mean(0)})
    out["threshold_top5"] = thr
    out["alert"] = (out["evening_p50"] > thr) | (out["p_evening_top5"] >= 0.5)
    return out


# ============================================================================ (b2) recalibration
RECORD_SKILL_MIN = 0.05   # below this Brier skill the gauges are labelled experimental


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(p, y) -> tuple[float, float] | None:
    """Logistic regression of the outcome on logit(p). None if there is nothing to learn from."""
    y = np.asarray(y, float)
    if len(y) < 30 or y.sum() < 2 or y.sum() > len(y) - 2:
        return None
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1.0).fit(_logit(p)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def apply_platt(p, ab):
    if ab is None:
        return np.asarray(p, float)
    a, b = ab
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def fit_isotonic(p, y):
    y = np.asarray(y, float)
    if len(y) < 30 or y.sum() < 2:
        return None
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p, float), y)


def rolling_recalibration(cal: pd.DataFrame, kind: str) -> pd.DataFrame:
    """For each month, fit Platt and isotonic maps on origins whose 7-day outcome was known before
    the month started (origin + 7 < month start) and apply them to that month's origins."""
    c = cal.sort_values("origin").copy()
    c["month"] = c["origin"].dt.to_period("M").dt.to_timestamp()
    c[f"p_{kind}_platt"] = np.nan
    c[f"p_{kind}_iso"] = np.nan
    for ms, idx in c.groupby("month").groups.items():
        past = c[c["origin"] + pd.Timedelta(days=HORIZON) < ms]
        p, y = past[f"p_{kind}"].to_numpy(), past[f"hit_{kind}"].astype(float).to_numpy()
        cur = c.loc[idx, f"p_{kind}"].to_numpy()
        c.loc[idx, f"p_{kind}_platt"] = apply_platt(cur, fit_platt(p, y))
        iso = fit_isotonic(p, y)
        c.loc[idx, f"p_{kind}_iso"] = iso.predict(cur) if iso is not None else cur
    return c.drop(columns="month")


def recalibration_live(cal: pd.DataFrame) -> dict:
    """Platt parameters on all backtest origins (for the live gauges)."""
    out = {}
    for kind in ("all_time", "year"):
        ab = fit_platt(cal[f"p_{kind}"], cal[f"hit_{kind}"].astype(float))
        out[kind] = list(ab) if ab else None
    return out


# ============================================================================ (d) product summaries
# Template-based, deterministic sentences for the Overview and the 7-day forecast page: no model
# calls, no external services, only the saved forecast, explanations and backtest.
CALENDAR = set(DAY_NAMES) | set(FEST_LABEL.values()) | {"Diwali week", "holiday", "run-up to a festival",
                                                         "festival calendar", "holidays in the last week"}
GENERIC = {"recent trend", "last weeks' pattern", "same day last year", "season", "other"}
TARGET_NOUN = {"solar_peak": "the solar-hour peak", "evening_peak": "the evening peak", "energy": "energy demand"}


def _unit(target: str) -> tuple[float, str, int]:
    return (1.0, "MU", 0) if target == "energy" else (1000.0, "GW", 1)


def recent_accuracy(bt: pd.DataFrame, model: str, days: int = 30) -> dict:
    """Our MAPE (all targets, 1–7 days ahead) and win rate against Grid-India over the last `days`
    days with actuals, each with the previous `days` for comparison."""
    from src.evaluate import head_to_head

    ours = bt[(bt["model"] == model)].dropna(subset=["actual"])
    if ours.empty:
        return {}
    end = ours["date"].max()
    cut1, cut0 = end - pd.Timedelta(days=days), end - pd.Timedelta(days=2 * days)

    def mape(lo, hi):
        g = ours[(ours["date"] > lo) & (ours["date"] <= hi)]
        return float(100 * ((g["p50"] - g["actual"]).abs() / g["actual"]).mean()) if len(g) else np.nan

    h = head_to_head(bt, model) if "grid_india" in set(bt["model"]) else pd.DataFrame()

    def win(lo, hi):
        if h.empty:
            return np.nan
        g = h[(h["date"] > lo) & (h["date"] <= hi)]
        return float(100 * g["ours_wins"].mean()) if len(g) else np.nan

    return {"end": end, "days": days, "mape": mape(cut1, end), "mape_prev": mape(cut0, cut1),
            "win": win(cut1, end), "win_prev": win(cut0, cut1),
            "n_days": int(ours.loc[ours["date"] > cut1, "date"].nunique())}


def driver_sentences(contrib: dict, target: str, top: int = 4, min_abs: float = 0.3) -> list[tuple[str, float]]:
    """Signed effects -> plain-English rows: [('Diwali week lowers the evening peak', -10.9), ...].
    Generic statistical inputs (recent trend, last weeks' pattern, ...) are left out."""
    noun = TARGET_NOUN.get(target, target)
    items = [(k, v) for k, v in contrib.items() if k not in GENERIC and abs(v) >= min_abs]
    items.sort(key=lambda kv: -abs(kv[1]))
    out = []
    for k, v in items[:top]:
        verb = "raises" if v > 0 else "lowers"
        subject = f"Day of the week ({k})" if k in DAY_NAMES else k[0].upper() + k[1:]
        out.append((f"{subject} {verb} {noun}", float(v)))
    return out


def week_insights(fc: pd.DataFrame, ex: pd.DataFrame | None, model: str, target: str = "evening_peak") -> list[tuple[str, str]]:
    """2–3 short findings about the week ahead for one target: the biggest driver, the day that
    stands out, and the largest weekend or festival effect. [(label, sentence)]."""
    div, unit, dec = _unit(target)
    noun = TARGET_NOUN.get(target, target)
    out: list[tuple[str, str]] = []
    lead = None
    e = ex[ex["target"] == target] if ex is not None and len(ex) else None
    if e is not None and len(e):
        tot: dict[str, float] = {}
        for c in e["contrib"]:
            for k, v in c.items():
                if k not in GENERIC:
                    tot[k] = tot.get(k, 0.0) + v / len(e)
        if tot:
            k, v = max(tot.items(), key=lambda kv: abs(kv[1]))
            if abs(v) >= 0.1:
                lead = k
                out.append(("Biggest driver", f"{k[0].upper() + k[1:]} {'raises' if v > 0 else 'lowers'} {noun} "
                                              f"by {abs(v):.{dec}f} {unit} a day on average this week."))
    f = fc[(fc["model"] == model) & (fc["target"] == target)].sort_values("date")
    if len(f) >= 3:
        mean = f["p50"].mean()
        dev = f["p50"] - mean
        i = dev.abs().idxmax()
        day, d = pd.Timestamp(f.loc[i, "date"]), dev[i] / div
        if abs(dev[i]) / mean >= 0.01:
            out.append(("Day to watch", f"{day:%a %d %b}: {f.loc[i, 'p50'] / div:,.{dec}f} {unit}, "
                                        f"{abs(d):,.{dec}f} {unit} {'above' if d > 0 else 'below'} the week's average."))
    if e is not None and len(e):
        floor = -0.3 * (1 if unit == "GW" else 10)
        cands = sorted(((pd.Timestamp(r["date"]), k, v) for _, r in e.iterrows() for k, v in r["contrib"].items()
                        if k in CALENDAR and k != "holidays in the last week" and v <= floor), key=lambda c: c[2])
        other = [c for c in cands if c[1] != lead]            # prefer an effect the first bullet did not name
        best = (other or cands or [None])[0]
        if best is not None:
            day, k, v = best
            what = f"{k} lowers" if k not in DAY_NAMES else f"the {'weekend' if k in ('Saturday', 'Sunday') else k} lowers"
            out.append(("Calendar effect", f"{day:%a %d %b}: {what[0].upper() + what[1:]} {noun} by "
                                           f"{abs(v):.{dec}f} {unit}."))
    return out[:3]


def week_headline(fc: pd.DataFrame, model: str, records: dict | None) -> str | None:
    """'Peak demand expected between *207 and 222 GW* this week. No record risk.' Record risk
    compares the week's P90 and P50 with the all-time peak (*...* marks the accent phrase)."""
    f = fc[fc["model"] == model]
    p = f.pivot_table(index="date", columns="target", values=["p50", "p90"])
    if not {("p50", "solar_peak"), ("p50", "evening_peak")} <= set(p.columns):
        return None
    p50 = p["p50"][["solar_peak", "evening_peak"]].max(axis=1) / 1000
    p90 = p["p90"][["solar_peak", "evening_peak"]].max(axis=1) / 1000 if "p90" in p else p50
    lo, hi = p50.min(), p50.max()
    rec = (records or {}).get("all_time")
    if not rec:
        tail = ""
    elif p50.max() >= rec / 1000:
        tail = ". A new all-time record is likely"
    elif p90.max() >= rec / 1000:
        tail = f". A new record is possible (upper range {p90.max():,.0f} GW vs {rec / 1000:,.0f} GW)"
    else:
        tail = ". No record risk"
    return f"Peak demand expected between *{lo:,.0f} and {hi:,.0f} GW* this week{tail}."
