#!/usr/bin/env python
"""Helpers for the scheduled snapshot job (.github/workflows/daily-snapshot.yml).

    python scripts/ci_snapshot.py check     # after the pipeline: fail unless the run and its outputs are sound
    python scripts/ci_snapshot.py finish    # copy the run history and the data-quality report next to the data
    python scripts/ci_snapshot.py paths     # the only paths the job may commit, one per line

`check` is the safety net: the job commits nothing unless it exits 0. It fails when the pipeline run
did not end as "success" (a failed download counts: the snapshot must not look fresh when it is not),
when the forecast is incomplete or has missing values, or when the data went backwards.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
KEEP_RUNS = 60

# everything the job is allowed to change in the repository (git add uses exactly these)
COMMIT_PATHS = ["data/processed", "data/raw/weather", "reports/data_quality.md", "reports/data_quality_retries.json"]


def fail(msg: str) -> None:
    print(f"SNAPSHOT CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _last_run() -> dict:
    p = ROOT / "logs" / "pipeline_runs.jsonl"
    if not p.exists():
        fail("the pipeline wrote no run log")
    return json.loads(p.read_text().strip().splitlines()[-1])


def check(max_age_days: int = 3) -> None:
    run = _last_run()
    if run.get("status") != "success":
        bad = [f"{s['short']}: {s.get('error') or s.get('message')}" for s in run["steps"]
               if s["status"] in ("failed", "skipped")]
        fail(f"pipeline status is {run.get('status')!r} ({'; '.join(bad) or run.get('summary')})")

    meta = json.loads((PROCESSED / "latest_meta.json").read_text())
    issued = pd.Timestamp(meta["issued_at"])
    if (pd.Timestamp.now() - issued) > pd.Timedelta(hours=6):
        fail(f"latest_meta.json was not written by this run (issued_at {issued})")

    fc = pd.read_parquet(PROCESSED / "latest_forecast.parquet")
    best = fc[fc["model"] == meta["best_model"]]
    if best["date"].nunique() != 7 or best["p50"].isna().any() or not (best["p50"] > 0).all():
        fail(f"forecast of {meta['best_model']} is incomplete: {best['date'].nunique()} days, "
             f"{int(best['p50'].isna().sum())} missing values")

    ds = pd.read_parquet(PROCESSED / "daily_summary.parquet", columns=["date"])["date"]
    latest = pd.Timestamp(ds.max())
    if (pd.Timestamp.now().normalize() - latest).days > max_age_days:
        fail(f"latest actuals are {latest:%Y-%m-%d}, more than {max_age_days} days old: the download is stuck")
    if latest < pd.Timestamp(meta["origin"]) - pd.Timedelta(days=1):
        fail(f"forecast origin {meta['origin']} is later than the actuals ({latest:%Y-%m-%d})")
    print(f"ok: success run, forecast for {best['date'].min():%d %b}-{best['date'].max():%d %b}, "
          f"actuals through {latest:%d %b %Y}; {run.get('summary')}")


def finish() -> None:
    """Keep the last runs and the quality report next to the data, where the read-only app looks."""
    new = (ROOT / "logs" / "pipeline_runs.jsonl").read_text().splitlines()
    out = PROCESSED / "pipeline_runs.jsonl"
    old = out.read_text().splitlines() if out.exists() else []
    out.write_text("\n".join((old + new)[-KEEP_RUNS:]) + "\n")
    q = ROOT / "reports" / "latest" / "data_quality.md"
    if q.exists():
        shutil.copy(q, ROOT / "reports" / "data_quality.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "finish", "paths"])
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check()
    elif a.cmd == "finish":
        finish()
    else:
        print("\n".join(COMMIT_PATHS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
