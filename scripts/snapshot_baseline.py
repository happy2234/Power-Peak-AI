#!/usr/bin/env python
"""Snapshot the production models and the baseline's test-period metrics before the accuracy round.

    python scripts/snapshot_baseline.py          # writes models/baseline_snapshot/
    bash models/baseline_snapshot/restore.sh     # roll production back to this snapshot

The baseline's test metrics come from the fixed-split predictions already saved by
scripts/fixed_split.py (frozen models, test from 2025-04-01); they are summarised here, not
recomputed, and the summary is logged as test-period access #1.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import MODELS_DIR, PROCESSED, REPORTS  # noqa: E402

SNAP = MODELS_DIR / "baseline_snapshot"
PROCESSED_FILES = ["fixed_split_predictions.parquet", "fixed_split_info.json", "fixed_split_tables.parquet",
                   "fixed_split_grid_india.parquet", "evaluation_comparison.parquet", "model_selection.json",
                   "postprocess.json", "best_model.txt", "latest_meta.json", "backtest_processed.parquet",
                   "eval_weather.parquet", "eval_coverage.parquet", "record_calibration.parquet", "mc_corr.npy",
                   "latest_forecast.parquet", "live_scorecard.parquet"]


def main():
    from src import fixed_split as fx
    from src import test_access
    from src.models.common import metrics

    if SNAP.exists() and any(SNAP.iterdir()):
        sys.exit(f"{SNAP} already exists; remove it first if you really want a new baseline")
    (SNAP / "production").mkdir(parents=True)
    (SNAP / "processed").mkdir()
    manifest = {"created": str(pd.Timestamp.now().floor("s")), "production": {}, "processed": []}
    for kind in ("lgbm", "lstm", "tft"):
        src = MODELS_DIR / kind
        dst = SNAP / "production" / kind
        dst.mkdir()
        latest = (src / "latest.txt").read_text().strip()
        shutil.copytree(src / latest, dst / latest)
        for f in src.glob("*.yaml"):
            shutil.copy2(f, dst / f.name)
        shutil.copy2(src / "latest.txt", dst / "latest.txt")
        manifest["production"][kind] = latest
    for f in PROCESSED_FILES:
        if (PROCESSED / f).exists():
            shutil.copy2(PROCESSED / f, SNAP / "processed" / f)
            manifest["processed"].append(f)

    # ---- baseline test metrics (from the saved predictions; test access #1)
    pred = pd.read_parquet(PROCESSED / "fixed_split_predictions.parquet")
    test = pred[pred["date"] >= pd.Timestamp(fx.TEST_START)].dropna(subset=["actual"])
    rows = []
    for (m, t), g in test.groupby(["model", "target"]):
        rows.append({"model": m, "target": t, **metrics(g), "days": int(g["date"].nunique())})
    tab = pd.DataFrame(rows)
    tab.to_csv(SNAP / "baseline_test_metrics.csv", index=False)
    test_access.record("baseline", "fixed-split test metrics of the production recipe (saved predictions, "
                       "summarised, not recomputed)", len(test), note=f"predictions from {fx.TEST_START} to "
                       f"{test['date'].max().date()}")
    manifest["baseline_test_rows"] = len(test)
    (SNAP / "manifest.json").write_text(json.dumps(manifest, indent=1))

    restore = ["#!/bin/bash", "# Roll production back to the models of this snapshot.", "set -e",
               'cd "$(dirname "$0")/../.."']
    for kind, latest in manifest["production"].items():
        restore += [f"mkdir -p models/{kind}", f"[ -d models/{kind}/{latest} ] || cp -r "
                    f"models/baseline_snapshot/production/{kind}/{latest} models/{kind}/",
                    f"echo -n {latest} > models/{kind}/latest.txt"]
    for f in ("model_selection.json", "postprocess.json", "best_model.txt"):
        if f in manifest["processed"]:
            restore.append(f"cp models/baseline_snapshot/processed/{f} data/processed/{f}")
    restore.append('echo "Production restored to the baseline snapshot. Run: python scripts/predict.py"')
    (SNAP / "restore.sh").write_text("\n".join(restore) + "\n")
    (SNAP / "restore.sh").chmod(0o755)

    # ---- committed copy of the baseline numbers
    ens = tab[tab["model"] == "ensemble"].set_index("target")
    L = ["# Baseline before the accuracy round", "",
         f"Snapshot taken {manifest['created']} (models/baseline_snapshot/, not in git). Production models: "
         + ", ".join(f"{k} {v}" for k, v in manifest["production"].items()) + ".", "",
         f"Fixed-split test period: every origin from {test['origin'].min().date()}, labels "
         f"{test['date'].min().date()} .. {test['date'].max().date()}, models trained once on FY 2023-24 + "
         "FY 2024-25 (validation Jan–Mar 2025). These numbers are the first of the two allowed looks at the "
         "test period.", "", "| model | target | days | MAPE % | MAE | bias | P10–P90 coverage % |",
         "|---|---|---|---|---|---|---|"]
    for _, r in tab.sort_values(["model", "target"]).iterrows():
        L.append(f"| {r['model']} | {r['target']} | {r['days']} | {r['MAPE']:.2f} | {r['MAE']:,.0f} | "
                 f"{r['bias']:+,.0f} | {r.get('coverage80', float('nan')):.1f} |")
    L += ["", f"Ensemble, mean of the three targets: MAPE {ens['MAPE'].mean():.3f} %, coverage "
          f"{ens['coverage80'].mean():.1f} %.", "", "Roll back: `bash models/baseline_snapshot/restore.sh`.", ""]
    (REPORTS / "baseline_snapshot.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
