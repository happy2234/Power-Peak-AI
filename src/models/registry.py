"""Locate and load the latest saved model of each kind (None if not trained yet)."""
from __future__ import annotations

import logging

from src.config import MODELS_DIR

log = logging.getLogger(__name__)
KINDS = ("lgbm", "lstm", "tft")


def latest_dir(kind: str):
    ptr = MODELS_DIR / kind / "latest.txt"
    if not ptr.exists():
        return None
    d = MODELS_DIR / kind / ptr.read_text().strip()
    return d if d.exists() else None


def load(kind: str):
    d = latest_dir(kind)
    if d is None:
        return None
    try:
        if kind == "lgbm":
            from src.models.baselines import load_lgbm

            return load_lgbm(d)
        from src.models.deep import DeepForecaster

        return DeepForecaster.load(d)
    except Exception as e:  # incompatible / corrupt model: report, don't crash callers
        log.warning("could not load %s from %s: %s", kind, d, e)
        return None
