"""Produce the live 7-day forecast and everything the app shows about it.

Outputs (data/processed/):
    latest_forecast.parquet       all models, long format (+ grid_india rows if published)
    latest_explanations.parquet   per day x target drivers + sentence (LightGBM SHAP)
    latest_stress.parquet         evening-peak stress table
    latest_meta.json              origin, best model, records, record probabilities, weather source
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED, TARGETS

log = logging.getLogger(__name__)
TGT = list(TARGETS)


def live_raw_weather(origin: pd.Timestamp) -> dict[int, pd.DataFrame] | None:
    """Raw regional weather for origin+1..+7 from the Open-Meteo forecast API (issued now)."""
    from src.data.weather import CITIES, fetch_live_forecast, regional
    from src.features.build import WX_RAW

    try:
        days = (origin + pd.Timedelta(days=HORIZON) - pd.Timestamp(dt.date.today())).days + 2
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:          # one request per city, in parallel
            frames = list(pool.map(lambda c: fetch_live_forecast(c, days=max(days, 8)), CITIES))
    except Exception as e:
        log.warning("live weather forecast unavailable (%s); using archived/persistence", e)
        return None
    reg = regional(pd.concat(frames), ["date"]).set_index("date")
    out = {}
    for h in range(1, HORIZON + 1):
        d = origin + pd.Timedelta(days=h)
        if d not in reg.index or reg.loc[d, [f"tmax_{r}" for r in ("nr", "wr", "sr", "er", "ner")]].isna().any():
            log.warning("live forecast misses %s; falling back", d.date())
            return None
        out[h] = pd.DataFrame([reg.loc[d, WX_RAW].to_numpy()], index=[origin], columns=WX_RAW)
    return out


RAW_WEATHER = PROCESSED / "latest_raw_weather.parquet"


def save_raw_weather(raw: dict[int, pd.DataFrame], origin: pd.Timestamp) -> None:
    pd.concat([r.assign(h=h) for h, r in raw.items()]).assign(origin=pd.Timestamp(origin)).to_parquet(
        RAW_WEATHER, index=False)


def load_raw_weather(origin: pd.Timestamp) -> dict[int, pd.DataFrame] | None:
    """The raw regional weather a forecast from `origin` was made with (None if not saved)."""
    if not RAW_WEATHER.exists():
        return None
    t = pd.read_parquet(RAW_WEATHER)
    if t.empty or pd.Timestamp(t["origin"].iloc[0]) != pd.Timestamp(origin):
        return None
    o = pd.Timestamp(origin)
    return {int(h): g.drop(columns=["h", "origin"]).set_axis([o]) for h, g in t.groupby("h")}


def live_name(backtest_name: str) -> str:
    """Backtest variant name -> live model name ('lstm_w@compact' is saved and loaded as 'lstm')."""
    from src.evaluate import parse_variant

    return parse_variant(backtest_name)[0]


def postprocess_live(fcst: pd.DataFrame) -> pd.DataFrame:
    """Add the ensemble and apply the per-horizon conformal correction learned in the backtest
    (data/processed/postprocess.json). No-op if the backtest has not produced it yet."""
    from src.postprocess import apply_conformal, apply_ensemble, load_live

    pp = load_live()
    if not pp:
        return fcst
    members = tuple(live_name(m) for m in pp["ensemble_members"])
    if pp.get("ensemble") and set(members) <= set(fcst["model"]):
        ens = apply_ensemble(fcst, pp["ensemble"], members)
        fcst = pd.concat([fcst, ens], ignore_index=True)
    # the live lgbm/lstm are the variants chosen in the backtest (= the ensemble members)
    chosen = list(pp["ensemble_members"]) + ["tft", "ensemble"]
    cal = {live_name(m): pp["conformal"][m] for m in chosen if m in pp["conformal"]}
    return apply_conformal(fcst, cal)


def recalibrate_record(rp: dict) -> tuple[dict, dict]:
    """Platt-recalibrated record probabilities + the reliability status from the backtest."""
    from src.insights import apply_platt
    from src.postprocess import load_live

    pp = load_live() or {}
    st = pp.get("record") or {}
    out = dict(rp)
    for kind in ("all_time", "year"):
        ab = (st.get("platt") or {}).get(kind)
        if ab:
            out[f"p_{kind}_raw"] = rp[f"p_{kind}"]
            out[f"p_{kind}"] = float(apply_platt([rp[f"p_{kind}"]], ab)[0])
    status = {"experimental": bool(st.get("experimental", True)), "skill": st.get("skill"),
              "threshold": st.get("threshold"), "method": st.get("method", "none"),
              "events": st.get("events")}
    return out, status


def current_recipe() -> str:
    """The model recipe in production ('baseline' or an accuracy-round tag such as 'nov26'), so the
    live scorecard can separate forecasts made before and after a switch."""
    p = PROCESSED / "model_selection.json"
    tags = {t for t in ((json.loads(p.read_text()).get("recipe") or {}).values() if p.exists() else []) if t}
    return "+".join(sorted(tags)) or "baseline"


def best_model_name() -> str:
    p = PROCESSED / "best_model.txt"
    return p.read_text().strip() if p.exists() else "lgbm"


def run_predict(origin=None, use_live_weather: bool = True) -> dict:
    from src.features.build import load_daily, make_tabular
    from src.insights import (evening_stress, explain_lgbm, record_probability, records_at,
                              scenario_weather)
    from src.models import registry
    from src.models.baselines import grid_india_forecasts

    daily, fc = load_daily()
    have = daily.dropna(subset=TGT, how="all")
    origin = pd.Timestamp(origin) if origin else have.index.max()

    raw = live_raw_weather(origin) if use_live_weather else None
    wx_source = "open-meteo live forecast" if raw else "archived forecast / persistence"
    if raw:   # the exact inputs, so the app's what-if simulator starts from the published forecast
        save_raw_weather(raw, origin)
    wx_override = ({h: scenario_weather(r, None, [origin + pd.Timedelta(days=h)]) for h, r in raw.items()}
                   if raw else None)

    preds = []
    tab = make_tabular(daily, fc, [origin], wx_override=wx_override)
    # the weather the forecast was made with, for the app (all-India + regional, °C / mm / %)
    from src.features.build import RAIN3_COLS, WX_FEATS

    tab[["date", "h", "wx_source"] + WX_FEATS + RAIN3_COLS].assign(weather_source=wx_source).to_parquet(
        PROCESSED / "latest_weather.parquet", index=False)
    models = {k: registry.load(k) for k in registry.KINDS}
    for kind, m in models.items():
        if m is None:
            continue
        if kind == "lgbm":
            preds.append(m.predict_long(tab))
        else:
            preds.append(m.predict_long(daily, fc, [origin], wx_override=wx_override))
    if not preds:
        raise RuntimeError("no trained models found; run scripts/train.py first")
    fcst = postprocess_live(pd.concat(preds, ignore_index=True))
    best = live_name(best_model_name())
    best = best if best in fcst["model"].unique() else fcst["model"].iloc[0]
    fcst["is_best"] = fcst["model"] == best
    fcst["issued_at"] = pd.Timestamp.now().floor("s")
    fcst["recipe"] = current_recipe()
    fcst["weather_source"] = wx_source

    gi = grid_india_forecasts()
    if len(gi):
        g = gi[(gi["date"] > origin) & (gi["date"] <= origin + pd.Timedelta(days=HORIZON))]
        if len(g):
            fcst = pd.concat([fcst, pd.DataFrame({
                "model": "grid_india", "origin": origin, "date": g["date"],
                "h": (g["date"] - origin).dt.days, "target": g["target"], "p10": np.nan,
                "p50": g["p50"], "p90": np.nan, "actual": np.nan, "is_best": False})], ignore_index=True)
    fcst.to_parquet(PROCESSED / "latest_forecast.parquet", index=False)

    bp = fcst[fcst["model"] == best]
    rec = records_at(daily, origin)
    rp = record_probability(bp, rec)
    rp, status = recalibrate_record(rp)
    stress = evening_stress(bp, daily, origin)
    stress.to_parquet(PROCESSED / "latest_stress.parquet", index=False)

    expl = []
    if models.get("lgbm") is not None:
        for t in TGT:
            expl.append(explain_lgbm(models["lgbm"], tab, t))
    if expl:
        e = pd.concat(expl, ignore_index=True)
        e["drivers"] = e["drivers"].map(json.dumps)
        e["contrib"] = e["contrib"].map(json.dumps)
        e.to_parquet(PROCESSED / "latest_explanations.parquet", index=False)

    meta = {"origin": str(origin.date()), "best_model": best, "weather_source": wx_source,
            "issued_at": str(pd.Timestamp.now().floor("s")),
            "records": {"all_time": rec["all_time"], "all_time_date": str(rec["all_time_date"].date()),
                        "year": rec["year"], "year_date": str(rec["year_date"].date()) if rec["year_date"] is not None else None},
            "record_probability": rp, "record_status": status,
            "models": sorted(fcst["model"].unique().tolist())}
    (PROCESSED / "latest_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
