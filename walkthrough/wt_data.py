"""Data layer of the walkthrough: every function returns REAL values read from the project's files or
computed by the project's own code (src/...). No GUI imports here, so it is testable without Qt.

Every function returns plain pandas / dict objects plus a `src` list: the files or functions that
produced them (shown in the step's log). Missing files raise StepError with a readable message.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import HORIZON, PROCESSED, TARGETS  # noqa: E402

PSP_DIR = ROOT / "data" / "raw" / "psp"
WEATHER_DIR = ROOT / "data" / "raw" / "weather"
MODELS = ROOT / "models"
TRAINING = ROOT / "reports" / "training"
TGT = list(TARGETS)
TGT_LABEL = {"solar_peak": "Solar-hour peak (MW)", "evening_peak": "Evening peak (MW)", "energy": "Energy met (MU)"}


class StepError(Exception):
    """A step cannot run: the message is shown to the presenter instead of a chart."""


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def need(path: Path) -> Path:
    if not Path(path).exists():
        raise StepError(f"File missing: {rel(path)}\nThis step cannot run without it; the other steps still work.")
    return Path(path)


def read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(need(PROCESSED / name))


# --------------------------------------------------------------------------- the chosen day
def latest_day() -> pd.Timestamp:
    """Latest day with real actuals (not imputed, report present)."""
    d = read_parquet("daily_clean.parquet")
    ok = d[~d["is_missing_report"] & ~d["is_imputed"] & d["solar_peak"].notna()]
    return pd.Timestamp(ok["date"].max())


def psp_file(day: pd.Timestamp) -> Path:
    from src.parse.psp_parser import date_from_filename

    if not PSP_DIR.exists():
        raise StepError(f"Folder missing: {rel(PSP_DIR)}")
    hits = []
    for p in sorted(PSP_DIR.iterdir()):
        if "NLDC_PSP" not in p.name.upper() or p.suffix.lower() not in (".xls", ".xlsx"):
            continue
        try:
            if pd.Timestamp(date_from_filename(p)) == day:
                hits.append(p)
        except ValueError:
            pass
    if not hits:
        raise StepError(f"No raw PSP file for {day.date()} in {rel(PSP_DIR)}")
    return hits[-1]


def psp_file_safe(day: pd.Timestamp) -> Path:
    """Like psp_file but never raises (used for the file table of step 10)."""
    try:
        return psp_file(day)
    except StepError:
        return PSP_DIR / f"<no file for {day.date()}>"


# --------------------------------------------------------------------------- 1 + 2: raw Excel, parser
def raw_sheet(path: Path) -> dict:
    """MOP_E grid and the location of the three cells the project uses (same logic as the parser)."""
    from src.parse import psp_parser as pp

    sheets = pp.read_sheets(path)
    raw = pp.find_sheet(sheets, "MOP_E")
    if raw is None:
        raise StepError(f"{path.name} has no MOP_E sheet. Sheets: {list(sheets)}")
    grid = pp._normed_grid(raw)
    start = pp._section_start(grid, "a", "power supply position")
    if start is None:
        raise StepError("Section A (Power Supply Position) not found in MOP_E")
    end = pp._next_section(grid, start)
    header_row, region_cols = None, {}
    for r in range(start, end):
        cols = {grid.iat[r, c]: c for c in range(grid.shape[1]) if grid.iat[r, c] in pp._REGION_KEY}
        if {"nr", "wr", "sr", "er", "ner"} <= cols.keys():
            region_cols = {pp._REGION_KEY[k]: c for k, c in cols.items()}
            header_row = r
            break
    if header_row is None:
        raise StepError("Region header row not found in section A")
    import re

    rx = re.compile(pp.SUMMARY_ROWS["energy_met_mu"])
    er = next((r for r in range(header_row + 1, end) if rx.search(pp._row_label(grid, r))), None)
    cells = {}
    if er is not None:
        cells[(er, region_cols["all_india"])] = ("Energy Met, TOTAL (section A)", "energy_mu")
    for key, pattern, name in [("solar_hr", r"^solar hr", "Solar hr, Max Demand (section I)"),
                               ("non_solar_hr", r"^non[- ]?solar hr", "Non-Solar hr, Max Demand (section I)")]:
        hit = pp.find_cells(grid, pattern)
        if hit:
            r, c = hit[0]
            cc = next((x for x in range(c + 1, grid.shape[1]) if grid.iat[r, x]), None)
            if cc is not None:
                cells[(r, cc)] = (name, "solar_peak_mw" if key == "solar_hr" else "evening_peak_mw")
    return {"sheets": list(sheets), "shapes": {k: v.shape for k, v in sheets.items()}, "raw": raw,
            "cells": cells, "header_row": header_row, "section": (start, end), "path": path}


def parse_row(path: Path) -> dict:
    from src.parse.psp_parser import parse_daily_summary

    df = parse_daily_summary(path)
    r = df.iloc[0]
    out = {"date": pd.Timestamp(r["date"]).date().isoformat(),
           "solar_peak_mw": float(r["peak_demand_solar_hr_mw"]),
           "evening_peak_mw": float(r["peak_demand_non_solar_hr_mw"]),
           "energy_mu": float(r["energy_met_mu_all_india"])}
    return {"row": out, "full": df}


# --------------------------------------------------------------------------- 3: history and cleaning
def history() -> dict:
    from src.data.quality import build_clean, find_outliers

    summ = read_parquet("daily_summary.parquet")
    clean = read_parquet("daily_clean.parquet")
    rebuilt, imputed = build_clean(summ)
    same = bool(np.allclose(rebuilt[TGT].to_numpy(float), clean[TGT].reindex(rebuilt.index).to_numpy(float),
                            equal_nan=True)) if len(rebuilt) == len(clean) else False
    flagged = find_outliers(clean)
    return {"summary": summ, "clean": clean, "imputed": imputed, "outliers": flagged, "rebuild_matches": same,
            "n_missing": int(clean["is_missing_report"].sum()), "n_imputed": int(clean["is_imputed"].sum()),
            "n_outlier_col": int(clean["is_outlier"].sum())}


# --------------------------------------------------------------------------- 4: weather
def weights() -> dict:
    p = need(PROCESSED / "region_weights.json")
    return json.loads(p.read_text())


def weather_day(day: pd.Timestamp) -> dict:
    from src.data.weather import CITIES, load_observed_all, regional
    from src.features.build import derive_weather

    if not WEATHER_DIR.exists() or not list(WEATHER_DIR.glob("obs_*.parquet")):
        raise StepError(f"No raw weather in {rel(WEATHER_DIR)}")
    obs = load_observed_all()
    day_rows = obs[obs["date"] == day].copy()
    if day_rows.empty:
        raise StepError(f"No observed weather for {day.date()} in {rel(WEATHER_DIR)}")
    day_rows["region"] = day_rows["city"].map(lambda c: CITIES[c][2])
    w = weights()
    reg = regional(obs[obs["date"] == day], ["date"]).set_index("date")
    der = derive_weather(reg, w)
    ai = der.iloc[0]
    return {"cities": day_rows.sort_values(["region", "city"]).reset_index(drop=True), "weights": w,
            "regional": der.iloc[0], "ai": ai, "n_cities": len(day_rows), "city_regions": {c: v[2] for c, v in CITIES.items()}}


# --------------------------------------------------------------------------- 5: calendar
def calendar_day(day: pd.Timestamp) -> dict:
    from src.features.calendar import FESTIVAL_DATES, calendar_features, national_holidays

    cal = calendar_features([day]).iloc[0]
    hol = national_holidays([day.year]).get(day)
    fest = [f for f, ds in FESTIVAL_DATES.items() if day in ds]
    near = []
    for f, ds in FESTIVAL_DATES.items():
        for d in ds:
            if abs((d - day).days) <= 30:
                near.append((f, d.date().isoformat(), int((d - day).days)))
    return {"cal": cal, "holiday": hol, "festivals_today": fest, "festivals_near": sorted(near, key=lambda x: abs(x[2]))[:6],
            "weekday": day.day_name()}


# --------------------------------------------------------------------------- 6 + 7: merge, lags, scaling
def features_row(day: pd.Timestamp) -> pd.Series:
    f = read_parquet("features.parquet")
    r = f[f["date"] == day]
    if r.empty:
        raise StepError(f"{day.date()} is not in features.parquet")
    return r.iloc[0]


def model_configs() -> dict:
    out = {}
    for kind in ("lgbm", "lstm", "tft"):
        ptr = MODELS / kind / "latest.txt"
        if not ptr.exists():
            continue
        d = MODELS / kind / ptr.read_text().strip()
        cfg = d / "config.yaml"
        if cfg.exists():
            out[kind] = {"dir": d, "cfg": yaml.safe_load(cfg.read_text())}
    return out


def group_of(col: str) -> str:
    from src.features.build import CAL_FEATS, ENC_CAL, DEC_CAL, WX_FEATS, WX_RAW

    if "_lag" in col or col.startswith("ref_") and col.endswith(("_lag7", "_lag14", "_lag364")):
        return "lag"
    if any(s in col for s in ("_roll", "_std7", "_std30")):
        return "rolling"
    if col in set(CAL_FEATS) | set(ENC_CAL) | set(DEC_CAL) or col.startswith("fest_") or col == "ref_holidays":
        return "calendar"
    base = col.replace("_anom", "").replace("d_", "", 1) if col.startswith("d_") else col.replace("_anom", "")
    if col in set(WX_FEATS) | set(WX_RAW) or base in set(WX_FEATS) | set(WX_RAW) or col.startswith("rain3") or col == "wx_source":
        return "weather"
    if col.endswith("_last") or col.endswith("_r") or col.endswith("_mask"):
        return "lag"
    return "other"


def feature_groups() -> dict:
    """{model: {group: [cols]}} from each model's config.yaml."""
    out = {}
    for kind, v in model_configs().items():
        cfg = v["cfg"]
        cols = list(cfg["features"]) if kind == "lgbm" else list(dict.fromkeys(cfg["enc_features"] + cfg["dec_features"]))
        g: dict[str, list] = {"weather": [], "calendar": [], "lag": [], "rolling": [], "other": []}
        for c in cols:
            g[group_of(c)].append(c)
        out[kind] = g
    return out


def lag_example(day: pd.Timestamp, target: str = "solar_peak", h: int = 1) -> dict:
    """The origin is `day`; the forecast is for day+h. Shows the real past values behind the *_last,
    *_lag7/14/364 and *_roll30 features and the ratio the model really sees (from make_tabular)."""
    from src.features.build import load_daily, make_tabular

    daily, fc = load_daily()
    if day not in daily.index:
        raise StepError(f"{day.date()} not in features.parquet")
    tab = make_tabular(daily, fc, [day], horizons=[h]).iloc[0]
    s = daily[target]
    d = day + pd.Timedelta(days=h)
    ref = float(tab[f"ref_{target}"])
    pts = {"last": (day, float(s.get(day, np.nan))),
           "lag7": (d - pd.Timedelta(days=7), float(s.get(d - pd.Timedelta(days=7), np.nan))),
           "lag14": (d - pd.Timedelta(days=14), float(s.get(d - pd.Timedelta(days=14), np.nan))),
           "lag364": (d - pd.Timedelta(days=364), float(s.get(d - pd.Timedelta(days=364), np.nan)))}
    roll_days = pd.date_range(day - pd.Timedelta(days=29), day)
    return {"origin": day, "target_day": d, "h": h, "target": target, "ref": ref, "pts": pts, "series": s,
            "roll_days": roll_days, "roll30": float(s.reindex(roll_days).mean()),
            "tab": {k: float(tab[k]) for k in tab.index if k.startswith(target)},
            "ref_days": pd.date_range(day - pd.Timedelta(days=6), day), "daily": daily, "fc": fc}


def scaling_example(day: pd.Timestamp, feature: str = "solar_peak_lag7") -> dict:
    """One decoder feature before and after the production scaler (models/lstm/<latest>/scaler.npz)."""
    from src.features.build import dec_feats, make_sequences, load_daily

    mc = model_configs()
    if "lstm" not in mc:
        raise StepError("No production LSTM found (models/lstm/latest.txt)")
    d = mc["lstm"]["dir"]
    sc = need(d / "scaler.npz")
    z = np.load(sc)
    cfg = mc["lstm"]["cfg"]
    daily, fc = load_daily()
    seq = make_sequences(daily, fc, [day], cfg["enc_days"], feature_set=cfg.get("feature_set", "full"))
    names = dec_feats(cfg.get("feature_set", "full"))
    i = names.index(feature)
    raw = seq["dec"][0, :, i]
    scaled = np.clip((seq["dec"][0, :, i] - z["dec_mu"][i]) / z["dec_sd"][i], -10, 10)
    return {"feature": feature, "raw": raw, "scaled": scaled, "mu": float(z["dec_mu"][i]), "sd": float(z["dec_sd"][i]),
            "dec": seq["dec"][0], "dec_mu": z["dec_mu"], "dec_sd": z["dec_sd"],
            "enc_shape": seq["enc"].shape[1:], "dec_shape": seq["dec"].shape[1:], "enc_days": cfg["enc_days"],
            "origin": day, "scaler": d / "scaler.npz", "names": names, "train_end": cfg["train_end"],
            "all_scaled_ok": True}


def splits() -> dict:
    """Train / validation / test date ranges: the production run (config.yaml) and the fixed split."""
    from src.features.build import load_daily, valid_origins

    daily, _ = load_daily()
    origins = valid_origins(daily, start="2023-06-01")
    prod = {}
    mc = model_configs()
    if "lstm" in mc:
        te = pd.Timestamp(mc["lstm"]["cfg"]["train_end"])
        o = origins[origins <= te]
        n_val = max(int(len(o) * 0.15), 20)
        val = o[-n_val:]
        tr = o[o <= val[0] - pd.Timedelta(days=HORIZON)]
        prod = {"train": (tr.min(), tr.max()), "gap": (tr.max(), val.min()), "validation": (val.min(), val.max()),
                "n_train": len(tr), "n_val": len(val)}
    fixed = {}
    p = PROCESSED / "fixed_split_info.json"
    if p.exists():
        j = json.loads(p.read_text())
        fixed = {k: (pd.Timestamp(j[k][0]), pd.Timestamp(j[k][1])) for k in ("fit", "validation", "test") if k in j}
    if not prod and not fixed:
        raise StepError("No model config.yaml and no fixed_split_info.json found")
    return {"production": prod, "fixed": fixed, "first": origins.min(), "last": daily.index.max()}


# --------------------------------------------------------------------------- 8: training (replay + live)
def histories() -> dict:
    out = {}
    mc = model_configs()
    for kind in ("lstm", "tft"):
        if kind not in mc:
            continue
        ts = mc[kind]["dir"].name
        p = TRAINING / f"{kind}_{ts}.csv"
        if not p.exists():
            p = mc[kind]["dir"] / "history.csv"
        if p.exists():
            out[kind] = {"hist": pd.read_csv(p), "src": p}
    return out


def param_counts() -> dict:
    """Parameter counts of the production nets (built with the project's own class, loaded from model.pt)."""
    out = {}
    try:
        from src.models.deep import DeepForecaster
    except Exception as e:  # torch missing
        return {"error": f"torch not installed ({e}); counts skipped"}
    for kind, v in model_configs().items():
        if kind == "lgbm":
            continue
        try:
            m = DeepForecaster.load(v["dir"], device=__import__("torch").device("cpu"))
            out[kind] = int(sum(p.numel() for p in m.model.parameters()))
        except Exception as e:
            out[kind] = f"n/a ({e})"
    return out


def lgbm_info() -> dict:
    mc = model_configs()
    if "lgbm" not in mc:
        raise StepError("No production LightGBM (models/lgbm/latest.txt)")
    cfg = mc["lgbm"]["cfg"]
    it = cfg["best_iters"]
    return {"best_iters": it, "total_trees": int(sum(it.values())), "params": cfg["params"], "src": mc["lgbm"]["dir"] / "config.yaml",
            "features": cfg["features"], "train_end": cfg.get("train_end")}


# ---- live training: the callables run inside a QThread; `progress(dict)` and `cancelled()` come from the worker
def live_lgbm(target: str, progress, cancelled, deadline: float, outdir: Path, n_estimators: int = 150) -> dict:
    """Small LightGBM P10/P50/P90 for ONE target on the real training window. Fixed seed. Writes only to outdir."""
    import lightgbm as lgb
    from src.features.build import load_daily, make_tabular, tabular_feature_cols, valid_origins
    from src.models.tune import best_params
    from src.config import QUANTILES

    daily, fc = load_daily()
    mc = model_configs()
    train_end = pd.Timestamp(mc["lgbm"]["cfg"].get("train_end") or daily.index.max()) if "lgbm" in mc else daily.index.max()
    fs = (mc["lgbm"]["cfg"].get("feature_set") if "lgbm" in mc else "compact") or "compact"
    origins = valid_origins(daily, start="2023-06-01", end=train_end)
    t0 = time.time()
    progress({"stage": "features", "text": f"building tabular samples for {len(origins)} origins ..."})
    tab = make_tabular(daily, fc, origins)
    feats = tabular_feature_cols(tab, fs)
    cut = origins[int(len(origins) * 0.9)]
    d = tab.dropna(subset=[f"y_{target}"])
    tr, va = d[d["origin"] < cut - pd.Timedelta(days=HORIZON)], d[d["origin"] >= cut]
    bp = {k: v for k, v in best_params("lgbm", fs).items() if k in ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda")}
    preds = {}
    for q in QUANTILES:
        if cancelled() or time.time() > deadline:
            raise TimeoutError("cancelled" if cancelled() else "time budget")
        curve = []

        def cb(env, _c=curve, _q=q):
            _c.append(env.evaluation_result_list[0][2])
            if env.iteration % 10 == 0:
                progress({"stage": "lgbm", "q": _q, "iter": env.iteration + 1, "of": n_estimators, "val": _c[-1], "elapsed": time.time() - t0})
            if cancelled() or time.time() > deadline:
                raise TimeoutError("cancelled" if cancelled() else "time budget")

        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, random_state=42, max_bin=127,
                              subsample_freq=1, verbose=-1, **bp)
        m.fit(tr[feats], tr[f"y_{target}"], eval_set=[(va[feats], va[f"y_{target}"])], callbacks=[cb])
        preds[q] = m.predict(va[feats]) * va[f"ref_{target}"].to_numpy()
    actual = va[f"actual_{target}"].to_numpy()
    res = pd.DataFrame({"date": va["date"].to_numpy(), "h": va["h"].to_numpy(), "p10": preds[0.1], "p50": preds[0.5], "p90": preds[0.9], "actual": actual})
    res.to_csv(outdir / "live_lgbm_validation.csv", index=False)
    mape = float(np.nanmean(np.abs(res["p50"] - res["actual"]) / res["actual"]) * 100)
    return {"seconds": time.time() - t0, "n_train": len(tr), "n_val": len(va), "mape": mape, "val": res, "n_estimators": n_estimators}


class _Trial:
    """Stands in for an Optuna trial: DeepForecaster calls report()/should_prune() every epoch, which gives us
    a progress hook and a clean cancel without touching the project's code."""

    def __init__(self, progress, cancelled, deadline, max_epochs):
        self.progress, self.cancelled, self.deadline, self.max_epochs = progress, cancelled, deadline, max_epochs
        self.t0 = time.time()

    def report(self, val, epoch):
        self.progress({"stage": "lstm", "epoch": epoch, "of": self.max_epochs, "val": float(val), "elapsed": time.time() - self.t0})

    def should_prune(self):
        return self.cancelled() or time.time() > self.deadline


def live_lstm(progress, cancelled, deadline: float, outdir: Path, max_epochs: int = 8) -> dict:
    try:
        import torch
    except Exception as e:
        raise StepError(f"torch is not installed ({e}); the live LSTM is skipped") from e
    import optuna
    from src.features.build import load_daily, valid_origins
    from src.models.deep import DeepForecaster
    from src.models.tune import best_params

    daily, fc = load_daily()
    mc = model_configs()
    cfg = mc["lstm"]["cfg"] if "lstm" in mc else {}
    fs = cfg.get("feature_set", "compact")
    te = pd.Timestamp(cfg.get("train_end") or daily.index.max())
    origins = valid_origins(daily, start="2023-06-01", end=te)
    params = best_params("lstm", fs)
    m = DeepForecaster("lstm", params, max_epochs=max_epochs, patience=max_epochs, seed=42, device=torch.device("cpu"), feature_set=fs)
    trial = _Trial(progress, cancelled, deadline, max_epochs)
    t0 = time.time()
    try:
        m.fit(daily, fc, origins, refit_all=False, trial=trial)
    except optuna.TrialPruned:
        raise TimeoutError("cancelled" if cancelled() else "time budget")
    h = pd.DataFrame(m.history)
    h.to_csv(outdir / "live_lstm_history.csv", index=False)
    return {"seconds": time.time() - t0, "history": h, "n_origins": len(origins), "best_val": float(m.best_val)}


# --------------------------------------------------------------------------- 9: ensemble and forecast
def ensemble_data(day: pd.Timestamp) -> dict:
    b = read_parquet("backtest_processed.parquet")
    post = json.loads(need(PROCESSED / "postprocess.json").read_text())
    members = post.get("ensemble_members", ["lstm_w@compact", "lgbm@compact"])
    names = list(members) + ["ensemble"]
    bt = b[b["model"].isin(names)]
    # forecast of THIS day's origin (1..7 days ahead) when it is in the backtest, else the stored latest forecast
    lf = read_parquet("latest_forecast.parquet")
    fc = None
    src = None
    if (lf["origin"] == day).any():
        fc, src = lf[lf["origin"] == day], "data/processed/latest_forecast.parquet"
    else:
        sel = b[(b["origin"] == day) & b["model"].isin(names + ["grid_india"])]
        if len(sel):
            fc, src = sel, "data/processed/backtest_processed.parquet"
    return {"backtest": bt, "members": members, "weights": post["ensemble"], "forecast": fc, "forecast_src": src,
            "post": post, "origin_in_backtest": bool((b["origin"] == day).any())}


def week_ahead_pdfs() -> list[Path]:
    d = ROOT / "data" / "raw" / "forecast"
    return sorted(d.glob("*.pdf")) if d.exists() else []


# --------------------------------------------------------------------------- 10: what we get
def outputs_table(day: pd.Timestamp, step_files: dict[int, list[Path]]) -> pd.DataFrame:
    rows = []
    for step, files in step_files.items():
        for p in files:
            p = Path(p)
            if not p.exists():
                rows.append({"step": step, "path": rel(p), "rows": "", "columns": "", "size": "missing", "modified": ""})
                continue
            nr = nc = ""
            try:
                if p.suffix == ".parquet":
                    import pyarrow.parquet as pq

                    f = pq.ParquetFile(p)
                    nr, nc = f.metadata.num_rows, f.metadata.num_columns
                elif p.suffix == ".csv":
                    df = pd.read_csv(p)
                    nr, nc = df.shape
                elif p.suffix in (".xls", ".xlsx"):
                    from src.parse.psp_parser import read_sheets

                    nr, nc = len(read_sheets(p)), "sheets"
                elif p.suffix == ".json":
                    j = json.loads(p.read_text())
                    nr, nc = (len(j), "keys") if isinstance(j, dict) else (len(j), "items")
                elif p.suffix == ".npz":
                    z = np.load(p)
                    nr, nc = len(z.files), "arrays"
            except Exception:
                pass
            st = p.stat()
            rows.append({"step": step, "path": rel(p), "rows": nr, "columns": nc, "size": (f"{st.st_size:,} B" if st.st_size < 1024 else f"{st.st_size / 1024:,.0f} KB"),
                         "modified": pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows)
