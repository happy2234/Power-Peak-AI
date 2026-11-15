import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON, REGIONS
from src.features.build import CAL_FEATS, TGT, WX_BASE, WX_FEATS, WX_RAW, derive_weather
from src.features.calendar import calendar_features

# app pages do `import common` / `import ui`; make app/ importable regardless of which page runs first
_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

@pytest.fixture(autouse=True)
def _full_app_mode(monkeypatch):
    """The app defaults to presentation mode; the page tests cover every page, so they run in full
    mode. Tests of presentation mode set POWER_PEAK_MODE themselves."""
    monkeypatch.setenv("POWER_PEAK_MODE", "full")


WEIGHTS = {"nr": 0.3, "wr": 0.3, "sr": 0.25, "er": 0.13, "ner": 0.02}


def _raw_weather(n, rng, base=None):
    df = pd.DataFrame(index=range(n))
    season = 4 * np.sin(np.arange(n) / 58) if base is None else base
    for r in REGIONS:
        t = 31 + season + 2 * rng.standard_normal(n)
        df[f"tmax_{r}"] = t
        df[f"tmean_{r}"] = t - 6
        df[f"app_tmax_{r}"] = t + 2
        df[f"rh_mean_{r}"] = 60 + 5 * rng.standard_normal(n)
        df[f"precip_{r}"] = rng.exponential(2, n)
        df[f"rad_{r}"] = 20 + rng.standard_normal(n)
    return df


@pytest.fixture(scope="session")
def demand_world():
    """Synthetic daily table + archived forecasts where demand rises with temperature and
    falls on Sundays/holidays, so model behaviour has a known direction."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-04-01", "2024-12-31", name="date")
    n = len(idx)
    raw = _raw_weather(n, rng)
    wx = derive_weather(raw, WEIGHTS)
    cal = calendar_features(idx)
    cols = {}
    temp_effect = 0.012 * (wx["tmax_ai"].to_numpy() - 31)
    sunday = -0.04 * cal["is_sunday"].to_numpy() - 0.06 * cal["is_holiday"].to_numpy()
    trend = 1 + 0.05 * np.arange(n) / 365
    for k, lvl in zip(TGT, (210_000, 200_000, 4500)):
        cols[k] = lvl * trend * (1 + temp_effect + sunday + 0.005 * rng.standard_normal(n))
        cols[f"{k}_imputed"] = False
    cols["is_imputed"] = False
    cols["is_outlier"] = False
    for c in dict.fromkeys(WX_FEATS + WX_RAW):
        cols[c] = wx[c].to_numpy()
    for c in CAL_FEATS:
        cols[c] = cal[c].to_numpy()
    daily = pd.DataFrame(cols, index=idx)
    # archived forecasts = truth + lead-dependent noise
    rows = []
    for lead in range(1, HORIZON + 1):
        noisy = raw.copy()
        for r in REGIONS:
            e = 0.3 * lead * rng.standard_normal(n)
            for v in ("tmax", "tmean", "app_tmax"):
                noisy[f"{v}_{r}"] += e
        f = derive_weather(noisy, WEIGHTS)
        f["date"], f["lead"] = idx, lead
        rows.append(f)
    fc = pd.concat(rows, ignore_index=True)[["date", "lead"] + list(dict.fromkeys(WX_FEATS + WX_RAW))]
    return daily, fc


@pytest.fixture(autouse=True)
def _weights(monkeypatch):
    import src.insights as ins

    monkeypatch.setattr(ins, "_region_weights", lambda: WEIGHTS)
    monkeypatch.setattr(ins, "_normals", lambda: {})
    _ = WX_BASE
