"""Cached data access for the Streamlit app (display only). Page chrome lives in ui.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import app_data  # noqa: E402


@st.cache_data(ttl=600)
def daily():
    return app_data.daily()


@st.cache_data(ttl=600)
def summary():
    return app_data.summary()


@st.cache_data(ttl=600)
def states():
    return app_data.states()


@st.cache_data(ttl=300)
def latest_forecast():
    return app_data.latest_forecast()


@st.cache_data(ttl=300)
def latest_meta():
    return app_data.latest_meta()


@st.cache_data(ttl=300)
def latest_explanations():
    return app_data.latest_explanations()


@st.cache_data(ttl=300)
def latest_stress():
    return app_data.latest_stress()


@st.cache_data(ttl=600, show_spinner="Loading 12 months of backtest forecasts…")
def backtest():
    return app_data.backtest()


@st.cache_data(ttl=600)
def grid_india():
    return app_data.grid_india()


def _cached(name: str, ttl: int = 600, spinner: str | bool = False):
    """A cached copy of an app_data reader (one parquet/JSON/markdown read per TTL, not per rerun)."""
    fn = getattr(app_data, name)

    def reader():
        return fn()

    # st.cache_data keys a function by its qualified name and source: give each reader its own
    # name before decorating, or every reader would share one cache entry
    reader.__name__ = reader.__qualname__ = f"read_{name}"
    return st.cache_data(ttl=ttl, show_spinner=spinner)(reader)


eval_weather = _cached("eval_weather")
eval_coverage = _cached("eval_coverage")
latest_weather = _cached("latest_weather", ttl=300)
fairness_audit = _cached("fairness_audit")
model_selection = _cached("model_selection")
fixed_split_tables = _cached("fixed_split_tables")
evaluation_comparison = _cached("evaluation_comparison")
fixed_split_info = _cached("fixed_split_info")
live_summary = _cached("live_summary", ttl=300)
live_scorecard = _cached("live_scorecard", ttl=300)
model_changes = _cached("model_changes", ttl=300)
record_calibration = _cached("record_calibration")
quality_md = _cached("quality_md", ttl=300)
training_curves = _cached("training_curves")
training_log = _cached("training_log")


@st.cache_data(ttl=600, show_spinner=False)
def leaderboard():
    from src.evaluate import leaderboard as lb

    b = backtest()
    return None if b is None else lb(b)


@st.cache_data(ttl=600, show_spinner=False)
def scorecard(model: str, target: str, corrected: bool = False):
    """Daily head-to-head with Grid-India (optionally with Grid-India's past-only bias removed)."""
    from src.insights import scorecard as sc

    b = backtest()
    if corrected:
        bias = grid_india_bias()
        b = b.merge(bias, on=["date", "target"], how="left")
        gi = b["model"] == "grid_india"
        b.loc[gi, "p50"] = b.loc[gi, "p50"] - b.loc[gi, "bias_past"].fillna(0)
    return sc(b, model, target)


@st.cache_data(ttl=3600, show_spinner="Estimating Grid-India's bias from earlier weeks…")
def grid_india_bias():
    """Grid-India's constant bias per row, estimated only from earlier days whose actuals were
    published before that week's forecast was prepared (src/audit.py)."""
    from src.audit import grid_india_with_issue, past_bias
    from src.evaluate import actuals_long

    gi = grid_india_with_issue(actuals_long(daily()))
    gi["bias_past"] = past_bias(gi)
    return gi[["date", "target", "bias_past"]]


@st.cache_data(ttl=600, show_spinner=False)
def recent_accuracy(model: str, days: int = 30) -> dict:
    from src.insights import recent_accuracy as ra

    b = backtest()
    return {} if b is None else ra(b, model, days)


def best_model():
    m = latest_meta()
    return (m or {}).get("best_model") or app_data.best_model() or "lgbm"


def model_label(m: str | None) -> str:
    from src.viz import MODEL_LABELS

    return MODEL_LABELS.get(m, m or "–")


def backtest_best():
    """Best model under its backtest name (e.g. 'lstm_w'; the live forecast calls it 'lstm')."""
    return app_data.best_model() or best_model()
