#!/usr/bin/env python
"""Download Grid-India history: daily PSP reports (+ optionally week-ahead forecast PDFs).

    python scripts/backfill.py --start 2023-04-01                  # .. yesterday
    python scripts/backfill.py --start 2023-04-01 --end 2023-06-30 --forecasts
    python scripts/backfill.py --days 7 --forecasts --build        # daily cron job

Already-downloaded files are skipped, so the script is safe to re-run / resume.
One listing request per financial year is used to resolve real file names, then one
request per missing file, with a polite delay between requests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fetch.grid_india import GridIndiaClient  # noqa: E402

log = logging.getLogger("backfill")


def parse_args() -> argparse.Namespace:
    yesterday = dt.date.today() - dt.timedelta(days=1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat, help="first data date (YYYY-MM-DD)")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=yesterday,
                    help="last data date (default: yesterday)")
    ap.add_argument("--days", type=int, help="alternative to --start: the last N days up to --end")
    ap.add_argument("--forecasts", action="store_true", help="also fetch week-ahead forecast PDFs")
    ap.add_argument("--build", action="store_true", help="run build_dataset afterwards")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests (default 1.5)")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    if args.days:
        args.start = args.end - dt.timedelta(days=args.days - 1)
    if not args.start:
        ap.error("give --start or --days")
    if args.start > args.end:
        ap.error("--start is after --end")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    client = GridIndiaClient(cache_dir=args.raw / "_index", delay=args.delay)

    dates = [args.start + dt.timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    log.info("PSP backfill %s .. %s (%d days)", args.start, args.end, len(dates))
    ok, missing, failed = 0, [], []
    for i, d in enumerate(dates, 1):
        try:
            path = client.download_psp(d, args.raw / "psp")
            if path:
                ok += 1
            else:
                missing.append(d)
                log.warning("%s: not listed on Grid-India", d)
        except (requests.RequestException, ValueError, RuntimeError) as e:
            failed.append((d, str(e)))
            log.error("%s: %s", d, e)
        except KeyboardInterrupt:
            log.warning("interrupted at %s; re-run to resume", d)
            break
        if i % 50 == 0:
            log.info("progress %d/%d (ok=%d missing=%d failed=%d)", i, len(dates), ok, len(missing), len(failed))

    if args.forecasts:
        # a forecast published a few days before --start may still cover it
        paths = client.download_forecasts(args.start - dt.timedelta(days=7), args.end + dt.timedelta(days=7),
                                          args.raw / "forecast")
        log.info("forecast PDFs available locally for the period: %d", len(paths))

    log.info("PSP done: %d ok, %d not listed, %d failed", ok, len(missing), len(failed))
    if missing:
        log.info("not listed: %s", ", ".join(map(str, missing[:30])) + (" ..." if len(missing) > 30 else ""))
    for d, err in failed:
        log.info("failed: %s %s", d, err)

    if args.build:
        from scripts.build_dataset import build  # noqa: E402
        build(args.raw, ROOT / "data" / "processed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
