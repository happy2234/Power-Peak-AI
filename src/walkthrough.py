"""Data layer of the "Raw to processed" page: ONE real day, from the raw Grid-India Excel file to the model input.

No Streamlit import: pure functions that read files and return plain pandas / dict objects, so they are testable
without the app. Every value comes from a file or from the project's own parser (src/parse/psp_parser.py);
nothing is typed in by hand. Every loader returns None when its file is missing.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import ENCODER_DAYS, HORIZON, MODELS_DIR, PROCESSED, QUANTILES, RAW, REGIONS, TARGETS
from src.data.weather import CITIES, WX_VARS
from src.parse import psp_parser as P

RAW_LATEST = RAW / "psp_latest"
CLIP = 10.0                                   # src/models/deep.py: np.clip((x - mu) / sd, -10, 10)
MODEL_KINDS = ("lstm", "tft", "lgbm")

# the label rules of P._parse_solar_nonsolar_peaks (its patterns are local to that function);
# tests/test_walkthrough.py checks that both give the same values
_PEAK_ROWS = {"solar_peak": r"^solar hr", "evening_peak": r"^non[- ]?solar hr"}
TARGET_NAMES = {"solar_peak": "Solar-hour peak", "evening_peak": "Evening (non-solar) peak", "energy": "Energy met"}


# ============================================================================ helpers
def col_letter(c: int) -> str:
    """0 -> 'A', 7 -> 'H', 26 -> 'AA'."""
    s = ""
    c += 1
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def cell_ref(r: int, c: int) -> str:
    """Zero-based (row, col) -> Excel reference: (8, 7) -> 'H9'."""
    return f"{col_letter(c)}{r + 1}"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _read_parquet(path: Path, **kw) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, **kw) if path.exists() else None
    except Exception:
        return None


def _row_for(df: pd.DataFrame | None, date) -> pd.Series | None:
    if df is None or "date" not in df:
        return None
    hit = df[pd.to_datetime(df["date"]) == pd.Timestamp(date).normalize()]
    return None if hit.empty else hit.iloc[0]


# ============================================================================ the raw file
def latest_raw_file() -> Path | None:
    """The newest DD.MM.YY_NLDC_PSP*.xls(x) in data/raw/psp_latest (by the date in its name), else None."""
    if not RAW_LATEST.is_dir():
        return None
    files = []
    for p in RAW_LATEST.iterdir():
        try:
            if p.is_file() and p.suffix.lower() in (".xls", ".xlsx"):
                files.append((P.date_from_filename(p), p.name, p))
        except ValueError:
            continue
    return max(files)[2] if files else None


def raw_overview(path: str | Path) -> dict:
    """File name, size, data date (from the file name), report date and the shape of every sheet."""
    path = Path(path)
    sheets = P.read_sheets(path)
    return {"name": path.name, "size_bytes": path.stat().st_size, "data_date": P.date_from_filename(path),
            "report_date": P.report_date(sheets),
            "sheets": [{"sheet": n, "rows": int(d.shape[0]), "columns": int(d.shape[1])} for n, d in sheets.items()]}


def _section_a(grid: pd.DataFrame) -> tuple[int, int, int, dict[str, int]]:
    """(start, end, header_row, {region: column}) of section A, found as P.parse_daily_summary finds them."""
    start = P._section_start(grid, "a", "power supply position")
    if start is None:
        raise ValueError("section A not found")
    end = P._next_section(grid, start)
    for r in range(start, end):
        cols = {grid.iat[r, c]: c for c in range(grid.shape[1]) if grid.iat[r, c] in P._REGION_KEY}
        if {"nr", "wr", "sr", "er", "ner"} <= cols.keys():
            return start, end, r, {P._REGION_KEY[k]: c for k, c in cols.items()}
    raise ValueError("region header not found in section A")


def target_cells(path: str | Path) -> list[dict]:
    """For each of the three targets: its label text, sheet, Excel cell, value and the daily_summary column.
    Found with the parser's own rules (SUMMARY_ROWS for energy, the 'Solar hr' / 'Non-Solar hr' rows of
    section I for the two peaks), never from fixed row numbers."""
    sheets = P.read_sheets(path)
    raw = P.find_sheet(sheets, "MOP_E")
    sheet_name = next(n for n, d in sheets.items() if d is raw)
    grid = P._normed_grid(raw)
    out = []

    start, end, header_row, region_cols = _section_a(grid)
    used: set[int] = set()
    energy_row = None
    for metric, pattern in P.SUMMARY_ROWS.items():          # same loop as the parser (a row is used once)
        rx = re.compile(pattern)
        m = next((r for r in range(header_row + 1, end) if r not in used and rx.search(P._row_label(grid, r))), None)
        if m is not None:
            used.add(m)
        if metric == "energy_met_mu":
            energy_row = m
    r, c = energy_row, region_cols["all_india"]
    out.append({"target": "energy", "name": TARGET_NAMES["energy"], "label": str(raw.iat[r, 0]).strip() or P._row_label(grid, r),
                "sheet": sheet_name, "cell": cell_ref(r, c), "row": r, "col": c, "value": P.to_num(raw.iat[r, c]),
                "column": TARGETS["energy"], "unit": "MU"})

    for key, pattern in _PEAK_ROWS.items():
        hits = P.find_cells(grid, pattern)
        if not hits:
            continue
        r, c = hits[0]
        vc = next((cc for cc in range(c + 1, grid.shape[1]) if grid.iat[r, cc]), None)
        out.append({"target": key, "name": TARGET_NAMES[key], "label": str(raw.iat[r, c]).strip(), "sheet": sheet_name,
                    "cell": cell_ref(r, vc), "row": r, "col": vc, "value": P.to_num(raw.iat[r, vc]),
                    "column": TARGETS[key], "unit": "MW"})
    order = list(TARGETS)
    return sorted(out, key=lambda d: order.index(d["target"]))


def mop_e_sections(path: str | Path) -> list[dict]:
    """MOP_E section A (all-India and regional supply position) and section I (peak at solar / non-solar hour)
    as raw cell text, with the target cells flagged. Each section: {"title", "rows": [{"excel_row", "cells",
    "targets": {col: target}}]}; cells are the raw values as text, trimmed to the columns in use."""
    sheets = P.read_sheets(path)
    raw = P.find_sheet(sheets, "MOP_E")
    grid = P._normed_grid(raw)
    cells = target_cells(path)
    start, end, _, _ = _section_a(grid)
    spans = [("A. Power supply position at all-India and regional level", start, end)]
    title = P.find_cells(grid, r"^i\s*[.)]\s*all india peak demand")
    peak_rows = [d["row"] for d in cells if d["target"] != "energy"]
    if title and peak_rows:
        spans.append(("I. All-India peak demand and shortage at solar and non-solar hour", title[0][0], max(peak_rows) + 1))
    out = []
    for name, a, b in spans:
        block = raw.iloc[a:b]
        used = [c for c in range(block.shape[1]) if any(P.norm(v) for v in block.iloc[:, c])]
        last = (max(used) + 1) if used else 0
        rows = []
        for r in range(a, b):
            texts = ["" if not P.norm(raw.iat[r, c]) else re.sub(r"\s+", " ", str(raw.iat[r, c])).strip()
                     for c in range(last)]
            if not any(texts):
                continue
            rows.append({"excel_row": r + 1, "cells": texts,
                         "targets": {d["col"]: d["target"] for d in cells if d["row"] == r}})
        out.append({"title": name, "rows": rows, "n_cols": last})
    return out


def reparsed_row(path: str | Path) -> pd.DataFrame:
    """One daily_summary row, parsed again from the raw file by the project's own parser."""
    return P.parse_daily_summary(path)


def compare_with_stored(path: str | Path) -> dict | None:
    """Re-parse the raw file and compare with the stored daily_summary row of the same date.
    Returns {"date", "rows": [(column, parsed, stored, match)], "n_columns", "n_match", "match"} or None."""
    parsed = reparsed_row(path)
    stored = _row_for(daily_summary(), parsed["date"].iloc[0])
    if stored is None:
        return None
    rows = []
    for c in parsed.columns:
        a, b = parsed[c].iloc[0], stored.get(c, np.nan)
        if isinstance(a, (float, int, np.floating, np.integer)) and isinstance(b, (float, int, np.floating, np.integer)):
            same = bool(np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True))
        else:
            same = bool((pd.isna(a) and pd.isna(b)) or a == b)
        rows.append((c, a, b, same))
    n = sum(r[3] for r in rows)
    return {"date": pd.Timestamp(parsed["date"].iloc[0]), "rows": rows, "n_columns": len(rows), "n_match": n,
            "match": n == len(rows)}


# ============================================================================ processed tables
def daily_summary() -> pd.DataFrame | None:
    return _read_parquet(PROCESSED / "daily_summary.parquet")


def daily_summary_row(date) -> pd.Series | None:
    return _row_for(daily_summary(), date)


def daily_clean() -> pd.DataFrame | None:
    return _read_parquet(PROCESSED / "daily_clean.parquet")


def daily_clean_row(date) -> pd.Series | None:
    """The cleaned row of that day, including the is_imputed / is_outlier / *_imputed flags."""
    return _row_for(daily_clean(), date)


def latest_clean_date() -> pd.Timestamp | None:
    d = daily_clean()
    return None if d is None or d.empty else pd.Timestamp(d["date"].max())


def clean_counts() -> dict | None:
    d = daily_clean()
    if d is None:
        return None
    return {"days": int(len(d)), "imputed": int(d["is_imputed"].sum()), "outliers": int(d["is_outlier"].sum()),
            "missing_reports": int(d["is_missing_report"].sum()) if "is_missing_report" in d else 0,
            "first": pd.Timestamp(d["date"].min()), "last": pd.Timestamp(d["date"].max())}


def recent_clean(date, days: int = 120) -> pd.DataFrame | None:
    """The last `days` rows of daily_clean up to and including `date`."""
    d = daily_clean()
    if d is None:
        return None
    d = d.assign(date=pd.to_datetime(d["date"]))
    return d[d["date"] <= pd.Timestamp(date)].sort_values("date").tail(days).reset_index(drop=True)


def city_weather() -> pd.DataFrame | None:
    return _read_parquet(PROCESSED / "weather_city.parquet")


def city_weather_table(date) -> pd.DataFrame | None:
    """The 11 cities for one day: one row per city (region, then the six base variables of weather.WX_VARS)."""
    row = _row_for(city_weather(), date)
    if row is None:
        return None
    recs = []
    for city, (_, _, region) in CITIES.items():
        rec = {"city": city, "region": region.upper()}
        rec.update({v: row.get(f"{v}_{city}", np.nan) for v in WX_VARS})
        recs.append(rec)
    return pd.DataFrame(recs)


def region_weights() -> dict[str, float] | None:
    return _read_json(PROCESSED / "region_weights.json")


def weather_check(date, var: str = "tmax") -> dict | None:
    """Rebuild one all-India value from the cities: mean of each region's cities, then the weighted average
    over the regions with region_weights.json. Compared with the stored <var>_ai in features.parquet."""
    cities, w, feat = _row_for(city_weather(), date), region_weights(), features_row(date)
    if cities is None or w is None or feat is None or f"{var}_ai" not in feat:
        return None
    reg = {r: float(np.mean([cities[f"{var}_{c}"] for c, v in CITIES.items() if v[2] == r])) for r in REGIONS}
    ai = sum(w[r] * reg[r] for r in REGIONS)
    return {"var": var, "regional": reg, "weights": {r: w[r] for r in REGIONS}, "rebuilt": float(ai),
            "stored": float(feat[f"{var}_ai"]), "match": bool(np.isclose(ai, feat[f"{var}_ai"], atol=1e-6))}


def features() -> pd.DataFrame | None:
    return _read_parquet(PROCESSED / "features.parquet")


def features_row(date) -> pd.Series | None:
    return _row_for(features(), date)


# ============================================================================ the model input
def _latest_dir(kind: str) -> Path | None:
    try:
        name = (MODELS_DIR / kind / "latest.txt").read_text().strip()
    except OSError:
        return None
    d = MODELS_DIR / kind / name
    return d if d.is_dir() else None


def model_config(kind: str) -> dict | None:
    d = _latest_dir(kind)
    try:
        return yaml.safe_load((d / "config.yaml").read_text()) if d and (d / "config.yaml").exists() else None
    except Exception:
        return None


def model_scaler(kind: str = "lstm") -> dict[str, np.ndarray] | None:
    d = _latest_dir(kind)
    try:
        if d is None or not (d / "scaler.npz").exists():
            return None
        z = np.load(d / "scaler.npz")
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def model_summary() -> list[dict]:
    """One row per model kind found: encoder / decoder / distinct feature counts, run and train_end."""
    out = []
    for kind in MODEL_KINDS:
        c = model_config(kind)
        if c is None:
            continue
        if "enc_features" in c:
            enc, dec = list(c["enc_features"]), list(c["dec_features"])
            out.append({"model": kind, "run": _latest_dir(kind).name, "encoder": len(enc), "decoder": len(dec),
                        "distinct": len(set(enc) | set(dec)), "feature_set": c.get("feature_set"),
                        "train_end": c.get("train_end")})
        else:
            f = c.get("features")
            n = len(f) if isinstance(f, list) else f
            out.append({"model": kind, "run": _latest_dir(kind).name, "encoder": None, "decoder": None,
                        "distinct": n, "feature_set": c.get("feature_set"), "train_end": c.get("train_end")})
    return out


def window_info() -> dict:
    """The look-back / forecast window and the output shape (targets x horizon days x quantiles)."""
    cfg = model_config("lstm") or {}
    return {"encoder_days": cfg.get("enc_days", ENCODER_DAYS), "horizon": cfg.get("horizon", HORIZON),
            "targets": list(TARGETS), "quantiles": list(cfg.get("quantiles") or QUANTILES),
            "train_end": cfg.get("train_end"),
            "outputs": len(TARGETS) * HORIZON * len(QUANTILES)}


_WEATHER_RX = re.compile(r"^(tmax|tmin|tmean|app_tmax|heat_index|rh_mean|dew_mean|precip|rain_day|rain3|cloud_mean|"
                         r"cloud_day|rad|wind_max|gust_max|cdd|hdd|fog|dpd_morning)_")
GROUPS = ("Weather", "Calendar", "Lag", "Rolling", "Other")


def feature_group(name: str) -> str:
    if "lag" in name:
        return "Lag"
    if "roll" in name or name.endswith(("_ma7", "_ma30")):
        return "Rolling"
    if _WEATHER_RX.match(name):
        return "Weather"
    if name in ("dow", "month", "is_weekend", "days_to_festival", "days_from_festival", "is_bridge", "diwali_window",
                "is_festival", "is_holiday", "is_national_holiday", "is_saturday", "is_sunday") \
            or name.startswith(("dow_", "doy_", "fest_")):
        return "Calendar"
    return "Other"


def feature_groups(date) -> dict[str, pd.DataFrame] | None:
    """The features.parquet row of that day, grouped. Lag and rolling features are not stored in
    features.parquet: they are built when the model sequences are made, so those groups are normally empty."""
    row = features_row(date)
    if row is None:
        return None
    groups = {g: [] for g in GROUPS}
    for name, v in row.items():
        if name != "date":
            groups[feature_group(name)].append((name, v))
    return {g: pd.DataFrame(v, columns=["feature", "value"]) for g, v in groups.items()}


def sequence_only_features(kind: str = "lstm") -> list[str]:
    """Model inputs that are NOT columns of features.parquet (ratios, masks, lags, forecast-weather flags)."""
    cfg, f = model_config(kind), features()
    if cfg is None or f is None:
        return []
    names = list(dict.fromkeys(list(cfg.get("enc_features", [])) + list(cfg.get("dec_features", []))))
    return [n for n in names if n not in f.columns]


def scaled_examples(date, model: str = "lstm") -> pd.DataFrame | None:
    """Raw -> scaled for the encoder features that are columns of features.parquet:
    scaled = clip((raw - enc_mu) / enc_sd, -10, 10), with enc_mu / enc_sd from the model's scaler.npz."""
    cfg, sc, row = model_config(model), model_scaler(model), features_row(date)
    if cfg is None or sc is None or row is None or "enc_features" not in cfg:
        return None
    recs = []
    for i, name in enumerate(cfg["enc_features"]):
        if name not in row.index:
            continue
        raw = float(row[name])
        mu, sd = float(sc["enc_mu"][i]), float(sc["enc_sd"][i])
        z = (raw - mu) / sd
        recs.append({"feature": name, "raw": raw, "enc_mu": mu, "enc_sd": sd, "scaled": float(np.clip(z, -CLIP, CLIP)),
                     "clipped": bool(abs(z) > CLIP)})
    return pd.DataFrame(recs)


# ============================================================================ the chain of files
def file_chain() -> list[dict]:
    """Every file in the chain: raw -> parsed -> cleaned -> weather -> features -> model, with rows, columns,
    size and modified time (None where a count does not apply or the file is missing)."""
    raw = latest_raw_file()
    items: list[tuple[str, str, Path | None]] = [
        ("1 Raw", raw.name if raw else "Latest Grid-India report (missing)", raw),
        ("2 Parse", "daily_summary.parquet", PROCESSED / "daily_summary.parquet"),
        ("3 Clean", "daily_clean.parquet", PROCESSED / "daily_clean.parquet"),
        ("4 Weather", "weather_city.parquet", PROCESSED / "weather_city.parquet"),
        ("4 Weather", "region_weights.json", PROCESSED / "region_weights.json"),
        ("5 Features", "features.parquet", PROCESSED / "features.parquet")]
    for kind in MODEL_KINDS:
        d = _latest_dir(kind)
        if d is not None:
            items.append(("5 Model input", f"{kind}/{d.name}/config.yaml", d / "config.yaml"))
            if (d / "scaler.npz").exists():
                items.append(("5 Model input", f"{kind}/{d.name}/scaler.npz", d / "scaler.npz"))
    items.append(("6 Output", "latest_forecast.parquet", PROCESSED / "latest_forecast.parquet"))
    out = []
    for step, name, p in items:
        rec = {"step": step, "file": name, "detail": "",
               "rows": None, "columns": None, "size_kb": None, "modified": None, "exists": bool(p and p.exists())}
        if p is not None and p.exists():
            st = p.stat()
            rec["size_kb"] = st.st_size / 1024
            rec["modified"] = pd.Timestamp(dt.datetime.fromtimestamp(st.st_mtime))
            if p.suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    md = pq.ParquetFile(p).metadata
                    rec["rows"], rec["columns"] = md.num_rows, md.num_columns
                except Exception:
                    pass
            elif p.suffix in (".xls", ".xlsx"):
                rec["detail"] = f"{len(P.read_sheets(p))} sheets"
        out.append(rec)
    return out
