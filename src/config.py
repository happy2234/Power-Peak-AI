"""Project-wide paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
WEATHER_DIR = RAW / "weather"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"



def readonly() -> bool:
    """POWER_PEAK_READONLY=1: a public, read-only deployment of a data snapshot (no Update data
    button, no pipeline runs from the app). Unset on the maintainer's machine."""
    v = os.environ.get("POWER_PEAK_READONLY")
    if v is None:       # fail closed on Streamlit Community Cloud, which runs apps from /mount/src/<repo>
        return str(ROOT).startswith("/mount/src/")
    return v.strip().lower() in ("1", "true", "yes", "on")


START_DATE = "2023-04-01"
HORIZON = 7                      # forecast days t+1 .. t+7
QUANTILES = (0.1, 0.5, 0.9)
ENCODER_DAYS = 60                # DL look-back window

# target name -> column in daily_summary.parquet
TARGETS = {
    "solar_peak": "peak_demand_solar_hr_mw",       # MW, solar-hour (daytime) peak
    "evening_peak": "peak_demand_non_solar_hr_mw",  # MW, non-solar-hour (evening) peak
    "energy": "energy_met_mu_all_india",            # MU (GWh) per day
}
TARGET_UNITS = {"solar_peak": "MW", "evening_peak": "MW", "energy": "MU"}
TARGET_LABELS = {"solar_peak": "Solar-hour peak", "evening_peak": "Evening (non-solar) peak",
                 "energy": "Energy met"}

REGIONS = ["nr", "wr", "sr", "er", "ner"]
