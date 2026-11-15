#!/usr/bin/env python
"""Train forecasting models on all available data and save them to models/<name>/<timestamp>/.

    python scripts/train.py --model all            # lgbm + lstm + tft with saved/best params
    python scripts/train.py --model tft --tune     # Optuna search first (30 trials), then train
    python scripts/train.py --model lgbm --tune --trials 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED  # noqa: E402
from src.utils.device import describe_device, set_seed  # noqa: E402

log = logging.getLogger("train")


def chosen_variant(kind: str) -> tuple[float, str]:
    """(regime_boost, feature_set) chosen by the backtest (data/processed/model_selection.json)."""
    p = PROCESSED / "model_selection.json"
    if p.exists():
        sel = json.loads(p.read_text())
        return float(sel.get("regime_boost", {}).get(kind, 0.0)), sel.get("feature_set", {}).get(kind, "full")
    return 0.0, "full"


def chosen_recipe(kind: str) -> tuple[str | None, dict]:
    """(tag, options) of the recipe chosen for this family (src/evaluate.py RECIPES)."""
    from src.evaluate import RECIPES

    p = PROCESSED / "model_selection.json"
    tag = (json.loads(p.read_text()).get("recipe") or {}).get(kind) if p.exists() else None
    return tag, dict(RECIPES[tag].get(kind, {})) if tag else {}


def train(kind: str, tune: bool, trials: int, boost: float | None = None, feature_set: str | None = None) -> dict:
    from src.features.build import load_daily, make_tabular, valid_origins
    from src.models.tune import best_params, tune as run_tune

    info = {"model": kind}
    c_boost, c_fs = chosen_variant(kind)
    boost = c_boost if boost is None else boost
    fs = c_fs if feature_set is None else feature_set
    info["feature_set"] = fs
    if tune:
        t0 = time.time()
        res = run_tune(kind, n_trials=trials, feature_set=fs)
        info["tune_minutes"] = round((time.time() - t0) / 60, 1)
        info["best_params"] = res["best_params"]
        log.info("%s tuned: %s", kind, res)
    daily, fc = load_daily()
    origins = valid_origins(daily, start="2023-06-01")
    info["regime_boost"] = boost
    tag, opts = chosen_recipe(kind)
    info["recipe"] = tag
    t0 = time.time()
    if kind == "lgbm":
        from src.models.baselines import LGBMQuantile, save_lgbm

        m = LGBMQuantile(best_params("lgbm", fs), regime_boost=boost, feature_set=fs, **opts).fit(
            make_tabular(daily, fc, origins))
        out = save_lgbm(m, str(origins.max().date()))
    else:
        from src.models.deep import DeepForecaster

        m = DeepForecaster(kind, best_params(kind, fs), regime_boost=boost, feature_set=fs, **opts).fit(
            daily, fc, origins, verbose=True)
        out = m.save()
        info["epochs"] = len(m.history)
        info["best_val_pinball_std"] = round(m.best_val, 4)
    info["train_seconds"] = round(time.time() - t0, 1)
    info["saved_to"] = str(out.relative_to(ROOT))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["lgbm", "lstm", "tft", "all"], default="all")
    ap.add_argument("--tune", action="store_true", help="run Optuna before training")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--regime-boost", type=float, default=None,
                    help="up-weight rare weather regimes (default: the variant chosen by the backtest)")
    ap.add_argument("--feature-set", choices=["v1", "compact", "full"], default=None,
                    help="weather inputs (default: the set chosen by the backtest)")
    ap.add_argument("--tune-only", action="store_true", help="tune and save parameters, do not train")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    set_seed(42)
    log.info("device: %s", describe_device())
    kinds = ["lgbm", "lstm", "tft"] if args.model == "all" else [args.model]
    if args.tune_only:
        from src.models.tune import tune as run_tune

        for k in kinds:
            log.info("%s", run_tune(k, n_trials=args.trials, feature_set=args.feature_set or "full"))
        return
    results = [train(k, args.tune, args.trials, args.regime_boost, args.feature_set) for k in kinds]
    for r in results:
        log.info("%s", r)
    log_path = PROCESSED / "training_log.jsonl"
    with open(log_path, "a") as fh:
        for r in results:
            fh.write(json.dumps({**r, "device": describe_device(), "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


if __name__ == "__main__":
    main()
