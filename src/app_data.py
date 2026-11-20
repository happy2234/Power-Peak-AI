"""Read-only loaders for the Streamlit app. Every function returns None (never raises) when
its data has not been produced yet, so pages can show 'not available yet'."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED, REPORTS, TARGETS, readonly

log = logging.getLogger(__name__)


def _read(path: Path):
    try:
        if not path.exists():
            return None
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        return path.read_text()
    except Exception as e:  # corrupt/partial file: degrade gracefully
        log.warning("could not read %s: %s", path, e)
        return None


def daily() -> pd.DataFrame | None:
    d = _read(PROCESSED / "features.parquet")
    return None if d is None else d.set_index("date")


def summary() -> pd.DataFrame | None:
    return _read(PROCESSED / "daily_summary.parquet")


def states() -> pd.DataFrame | None:
    return _read(PROCESSED / "state_demand.parquet")


def latest_forecast() -> pd.DataFrame | None:
    return _read(PROCESSED / "latest_forecast.parquet")


def latest_meta() -> dict | None:
    return _read(PROCESSED / "latest_meta.json")


def latest_explanations() -> pd.DataFrame | None:
    e = _read(PROCESSED / "latest_explanations.parquet")
    if e is not None:
        e["drivers"] = e["drivers"].map(json.loads)
        e["contrib"] = e["contrib"].map(json.loads)
    return e


def latest_stress() -> pd.DataFrame | None:
    return _read(PROCESSED / "latest_stress.parquet")


def backtest() -> pd.DataFrame | None:
    """Backtest rows for the app: the post-processed table (ensemble + calibrated intervals) when
    it exists, limited to the reported window; else the raw predictions."""
    p = _read(PROCESSED / "backtest_processed.parquet")
    if p is None:
        return _read(PROCESSED / "backtest_predictions.parquet")
    from src.evaluate import eval_window

    return eval_window(p).reset_index(drop=True)


def eval_weather() -> pd.DataFrame | None:
    """MAPE by weather regime / season (and before/after) written by scripts/backtest.py."""
    return _read(PROCESSED / "eval_weather.parquet")


def eval_coverage() -> pd.DataFrame | None:
    return _read(PROCESSED / "eval_coverage.parquet")


def latest_weather() -> pd.DataFrame | None:
    return _read(PROCESSED / "latest_weather.parquet")


def fairness_audit() -> dict | None:
    return _read(PROCESSED / "fairness_audit.json")


def model_selection() -> dict | None:
    return _read(PROCESSED / "model_selection.json")


def fixed_split_tables() -> pd.DataFrame | None:
    """Evaluation tables of the fixed train/test split (scripts/fixed_split.py)."""
    return _read(PROCESSED / "fixed_split_tables.parquet")


def evaluation_comparison() -> pd.DataFrame | None:
    return _read(PROCESSED / "evaluation_comparison.parquet")


def live_summary() -> dict | None:
    return _read(PROCESSED / "live_summary.json")


def fixed_split_info() -> dict | None:
    return _read(PROCESSED / "fixed_split_info.json")


def model_changes() -> list | None:
    """Production recipe switches: [{"date", "from", "to", "note"}] (scripts/deploy_recipe.py)."""
    return _read(PROCESSED / "model_changes.json")


def postprocess_params() -> dict | None:
    return _read(PROCESSED / "postprocess.json")


def live_scorecard() -> pd.DataFrame | None:
    """Live forecasts scored against actuals by the daily pipeline (src/pipeline.py)."""
    return _read(PROCESSED / "live_scorecard.parquet")


def best_model() -> str | None:
    t = _read(PROCESSED / "best_model.txt")
    return t.strip() if t else None


def record_calibration() -> pd.DataFrame | None:
    return _read(PROCESSED / "record_calibration.parquet")


def results_md() -> str | None:
    return _read(REPORTS / "results.md")


def quality_md() -> str | None:
    latest = REPORTS / "latest" / "data_quality.md"   # daily copy from the pipeline
    return _read(latest if latest.exists() else REPORTS / "data_quality.md")


def training_curves() -> dict[str, pd.DataFrame]:
    out = {}
    d = REPORTS / "training"
    if d.exists():
        for p in sorted(d.glob("*.csv")):
            try:
                out[p.stem] = pd.read_csv(p)
            except Exception:
                pass
    return out


def optuna_trials(kind: str) -> pd.DataFrame | None:
    p = MODELS_DIR / kind / "optuna_trials.csv"
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def training_log() -> pd.DataFrame | None:
    p = PROCESSED / "training_log.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows) if rows else None


def grid_india() -> pd.DataFrame | None:
    try:
        from src.models.baselines import grid_india_forecasts

        g = grid_india_forecasts()
        return g if len(g) else None
    except Exception:
        return None


def snapshot_time() -> pd.Timestamp | None:
    """When the data was last produced, from the forecast's own issue time (latest_meta.json). Unlike
    a file's modification time this survives a git checkout, so a deployed copy shows the snapshot's
    date, not the day it was cloned."""
    try:
        return pd.Timestamp((latest_meta() or {})["issued_at"])
    except Exception:
        return None


def last_updated() -> pd.Timestamp | None:
    """When the data shown in the app last changed (newest of the actuals and forecast files); in a
    read-only deployment, the snapshot time."""
    if readonly() and snapshot_time() is not None:
        return snapshot_time()
    times = [p.stat().st_mtime for p in (PROCESSED / "daily_summary.parquet", PROCESSED / "features.parquet",
                                          PROCESSED / "latest_forecast.parquet") if p.exists()]
    return pd.Timestamp.fromtimestamp(max(times)).floor("s") if times else None


# ----------------------------------------------------------------------------- KPIs
def kpis(d: pd.DataFrame) -> dict:
    """Headline numbers from the daily table."""
    last = d.dropna(subset=["energy"]).index.max()
    peak = d[["solar_peak", "evening_peak"]].max(axis=1)
    e30 = d.loc[last - pd.Timedelta(days=29):last, "energy"].sum()
    prev = d.loc[last - pd.Timedelta(days=29 + 364):last - pd.Timedelta(days=364), "energy"].sum()
    rec_date = peak.idxmax()
    return {"last_date": last, "latest_peak": float(peak.loc[last]), "latest_energy": float(d.loc[last, "energy"]),
            "yoy_30d": float(e30 / prev - 1) if prev > 0 else np.nan,
            "record_peak": float(peak.max()), "record_date": rec_date}


TARGET_LIST = list(TARGETS)
