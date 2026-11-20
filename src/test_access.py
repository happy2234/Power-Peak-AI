"""Audit trail for the fixed split's test period.

The accuracy round may look at the test period exactly twice: once for the baseline, once for
the final recipe. Every evaluation that touches test rows calls record(); the log is append-only
(data/processed/test_access_log.jsonl) and the final report quotes it.
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import PROCESSED

LOG = PROCESSED / "test_access_log.jsonl"
TEST_START = "2025-04-01"
# the last validation origin is 2025-03-24 (its 7-day labels end 2025-03-31); anything later
# produces labels inside the test period
LAST_VALIDATION_ORIGIN = "2025-03-24"


def record(purpose: str, what: str, rows: int, note: str = "", path=None) -> dict:
    entry = {"at": str(pd.Timestamp.now().floor("s")), "purpose": purpose, "what": what, "rows": int(rows),
             "note": note}
    p = path or LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def entries(path=None) -> list[dict]:
    p = path or LOG
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def guard_origins(origins, allow_test: bool = False) -> None:
    """Raise if any origin would produce a label in the test period (unless allow_test)."""
    o = pd.DatetimeIndex(pd.to_datetime(origins))
    if not allow_test and len(o) and o.max() > pd.Timestamp(LAST_VALIDATION_ORIGIN):
        raise PermissionError(f"origin {o.max().date()} reaches the test period (labels from {TEST_START}); "
                              "only the final measurement may do that")
