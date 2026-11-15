#!/usr/bin/env python
"""Fixed train/test evaluation (train once on FY 2023-24 + FY 2024-25, test from 2025-04-01).

    python scripts/fixed_split.py --tune     # Optuna on the train/validation split, then train and test
    python scripts/fixed_split.py            # reuse models/<kind>/best_params_fixed_split.yaml
    python scripts/fixed_split.py --report   # rebuild the report from the saved predictions

Writes reports/fixed_split_results.md and data/processed/fixed_split_*.parquet,
evaluation_comparison.parquet, live_summary.json.
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

import pandas as pd  # noqa: E402

from src.config import PROCESSED  # noqa: E402
from src.utils.device import describe_device, set_seed  # noqa: E402

log = logging.getLogger("fixed_split")


def main():
    from src import fixed_split as fx
    from src.evaluate import BACKTEST_START

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log.info("device: %s", describe_device())
    set_seed(42)
    t0 = time.time()
    daily, fc = fx.features()
    fit_o, val_o, test_o = fx.split_origins(daily)
    log.info("origins: fit %s..%s (%d), validation %s..%s (%d), test %s..%s (%d)",
             fit_o.min().date(), fit_o.max().date(), len(fit_o), val_o.min().date(), val_o.max().date(), len(val_o),
             test_o.min().date(), test_o.max().date(), len(test_o))
    pred_p, info_p = PROCESSED / "fixed_split_predictions.parquet", PROCESSED / "fixed_split_info.json"
    if a.report and pred_p.exists():
        pred = pd.read_parquet(pred_p)
        info = json.loads(info_p.read_text())
    else:
        if a.tune:
            for kind in ("lgbm", "lstm", "tft"):
                fx.tune(kind, daily, fc, fit_o, val_o, n_trials=a.trials)
        models = fx.train(daily, fc, fit_o, val_o)
        val_pred = fx.predict(models, daily, fc, val_o)
        test_pred = fx.predict(models, daily, fc, test_o)
        pred, post = fx.postprocess(val_pred, test_pred)
        pred.to_parquet(pred_p, index=False)
        info = {"fit": [str(fit_o.min().date()), str(fit_o.max().date())],
                "validation": [str(val_o.min().date()), str(val_o.max().date())],
                "test": [str(test_o.min().date()), str(test_o.max().date())], "post": post,
                "device": describe_device(), "minutes": round((time.time() - t0) / 60, 1)}
        info_p.write_text(json.dumps(info, indent=1, default=str))
    gi = fx.grid_india_rows(daily)
    t = fx.tables(pred, daily, gi)
    t["_pred_rolling_window"] = pred[(pred["origin"] >= pd.Timestamp(BACKTEST_START))].dropna(subset=["actual"])
    pd.concat([v.assign(kind=k) for k, v in t.items() if not k.startswith("_") and k != "grid_india"],
              ignore_index=True).to_parquet(PROCESSED / "fixed_split_tables.parquet", index=False)
    if "grid_india" in t:
        t["grid_india"].to_parquet(PROCESSED / "fixed_split_grid_india.parquet", index=False)
    rp = PROCESSED / "backtest_processed.parquet"
    rolling = pd.read_parquet(rp) if rp.exists() else None
    live = fx.live_summary()
    info["post"] = {k: {kk: {int(h): v for h, v in vv.items()} for kk, vv in val.items()} if k == "ensemble" else val
                    for k, val in info["post"].items()}
    text = fx.write_report(t, info, rolling, live)
    print(text[:4000])
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
