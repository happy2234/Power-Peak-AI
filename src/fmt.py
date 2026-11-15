"""Display formatting shared by the app and the pipeline summaries.

Conventions: dates "09 Nov 2026", timestamps "09 Nov 2026 16:47", peaks in GW with one decimal,
energy in MU (million units = GWh) with no decimals.
"""
from __future__ import annotations

import pandas as pd

from src.config import TARGET_UNITS

DATE = "%d %b %Y"
DATETIME = "%d %b %Y %H:%M"
DATE_JS = "DD MMM YYYY"          # the same format for st.column_config.DateColumn


def date(d) -> str:
    return "–" if d is None or pd.isna(d) else f"{pd.Timestamp(d):{DATE}}"


def datetime(d) -> str:
    return "–" if d is None or pd.isna(d) else f"{pd.Timestamp(d):{DATETIME}}"


def date_range(a, b) -> str:
    """'02–08 Nov 2026', '28 Oct – 03 Nov 2026', '30 Dec 2025 – 05 Jan 2026'."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a.normalize() == b.normalize():
        return date(a)
    if (a.year, a.month) == (b.year, b.month):
        return f"{a:%d}–{b:{DATE}}"
    if a.year == b.year:
        return f"{a:%d %b} – {b:{DATE}}"
    return f"{date(a)} – {date(b)}"


def scale(target: str) -> tuple[float, str, int]:
    """(divisor, unit, decimals): peaks MW -> GW with 1 decimal, energy stays MU with 0."""
    return (1000.0, "GW", 1) if TARGET_UNITS[target] == "MW" else (1.0, "MU", 0)


def value(v, target: str, unit: bool = True, sign: bool = False) -> str:
    """A raw model value (MW or MU) formatted for display, e.g. '245.3 GW' or '4,512 MU'."""
    if v is None or pd.isna(v):
        return "–"
    div, u, dec = scale(target)
    s = f"{v / div:{'+' if sign else ''},.{dec}f}"
    return f"{s} {u}" if unit else s


def duration(s) -> str:
    """'0.4 s', '12 s', '2 min 05 s'."""
    if s is None or pd.isna(s):
        return "–"
    s = float(s)
    if s < 10:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m} min {sec:02d} s"
