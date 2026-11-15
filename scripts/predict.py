#!/usr/bin/env python
"""Write the live 7-day forecast (data/processed/latest_forecast.parquet and friends).

    python scripts/predict.py                 # origin = last day with data, live weather
    python scripts/predict.py --origin 2026-10-25 --no-live-weather
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.predict import run_predict

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", help="forecast origin (last known day), default: latest")
    ap.add_argument("--no-live-weather", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    meta = run_predict(args.origin, use_live_weather=not args.no_live_weather)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
