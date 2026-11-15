#!/usr/bin/env python
"""Parse every raw file in data/raw and append (deduplicated) to data/processed/*.parquet.

    python scripts/build_dataset.py              # incremental: only dates not yet processed
    python scripts/build_dataset.py --rebuild    # re-parse everything from scratch

Outputs
    daily_summary.parquet      one row per date            key: date
    state_demand.parquet       one row per date x state    key: date, state
    timeseries_15min.parquet   96 rows per date            key: timestamp
    forecast_daily.parquet     Grid-India week-ahead peaks  key: forecast_start, date
    forecast_curve.parquet     Grid-India 15-min forecast   key: forecast_start, timestamp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parse.forecast_parser import parse_forecast_curve, parse_forecast_pdf  # noqa: E402
from src.parse.psp_parser import (  # noqa: E402
    date_from_filename,
    parse_daily_summary,
    parse_state_demand,
    parse_timeseries,
)

log = logging.getLogger("build_dataset")

KEYS = {
    "daily_summary": ["date"],
    "state_demand": ["date", "state"],
    "timeseries_15min": ["timestamp"],
    "forecast_daily": ["forecast_start", "date"],
    "forecast_curve": ["forecast_start", "timestamp"],
}


def load(out_dir: Path, name: str) -> pd.DataFrame:
    p = out_dir / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def merge_and_save(out_dir: Path, name: str, old: pd.DataFrame, new: list[pd.DataFrame]) -> pd.DataFrame:
    new = [d for d in new if not d.empty]
    if not new:
        return old
    df = pd.concat([old, *new], ignore_index=True) if not old.empty else pd.concat(new, ignore_index=True)
    # keep="last": newly parsed rows replace older ones for the same key
    df = df.drop_duplicates(subset=KEYS[name], keep="last").sort_values(KEYS[name]).reset_index(drop=True)
    out = out_dir / f"{name}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)  # atomic on POSIX
    return df


def psp_files(raw: Path) -> list[Path]:
    files = [p for p in (raw / "psp").glob("*") if p.suffix.lower() in {".xls", ".xlsx"}]
    good = []
    for p in files:
        try:
            date_from_filename(p)
            good.append(p)
        except ValueError:
            log.warning("skipping %s: no DD.MM.YY date in name", p.name)
    # oldest first, and for duplicate dates the newest file last (so it wins the dedupe)
    return sorted(good, key=lambda p: (date_from_filename(p), p.stat().st_mtime))


def build(raw: Path, out_dir: Path, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {n: (pd.DataFrame() if rebuild else load(out_dir, n)) for n in KEYS}
    failures: list[tuple[str, str]] = []

    # ---------------------------------------------------------------- PSP reports
    done = set(tables["daily_summary"]["date"].dt.date) if not tables["daily_summary"].empty else set()
    todo = [p for p in psp_files(raw) if rebuild or date_from_filename(p) not in done]
    log.info("PSP: %d files to parse (%d dates already processed)", len(todo), len(done))
    new = {"daily_summary": [], "state_demand": [], "timeseries_15min": []}
    for i, p in enumerate(todo, 1):
        try:
            new["daily_summary"].append(parse_daily_summary(p))
            new["state_demand"].append(parse_state_demand(p))
            new["timeseries_15min"].append(parse_timeseries(p))
        except Exception as e:  # keep going; report at the end
            failures.append((p.name, f"{type(e).__name__}: {e}"))
        if i % 100 == 0:
            log.info("  parsed %d/%d", i, len(todo))
    for name, frames in new.items():
        tables[name] = merge_and_save(out_dir, name, tables[name], frames)

    # ---------------------------------------------------------------- forecast PDFs
    seen = set(tables["forecast_daily"]["source_file"]) if not tables["forecast_daily"].empty else set()
    pdfs = sorted((raw / "forecast").glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    pdfs = [p for p in pdfs if rebuild or p.name not in seen]
    log.info("Forecast: %d PDFs to parse", len(pdfs))
    fd, fc = [], []
    for p in pdfs:
        try:
            fd.append(parse_forecast_pdf(p))
            fc.append(parse_forecast_curve(p))
        except Exception as e:
            failures.append((p.name, f"{type(e).__name__}: {e}"))
    tables["forecast_daily"] = merge_and_save(out_dir, "forecast_daily", tables["forecast_daily"], fd)
    tables["forecast_curve"] = merge_and_save(out_dir, "forecast_curve", tables["forecast_curve"], fc)

    # ---------------------------------------------------------------- report
    for name, df in tables.items():
        log.info("%-18s %7d rows  %s", name, len(df), _span(df))
    if failures:
        log.warning("%d file(s) failed:", len(failures))
        for f, err in failures:
            log.warning("  %s -> %s", f, err)
    return tables


def _span(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    col = "date" if "date" in df else "timestamp"
    return f"{df[col].min():%Y-%m-%d} .. {df[col].max():%Y-%m-%d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--rebuild", action="store_true", help="ignore existing outputs and re-parse all")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    build(args.raw, args.out, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
