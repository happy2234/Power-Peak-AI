"""Calendar features: weekday/season, Indian national holidays, major festivals.

All calendar features depend only on the date itself, so they are known in advance for any
forecast day (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Major festivals 2023-2027 (main day; regional/lunar dates can vary by +-1 day).
# Sources: Government of India holiday lists / standard Hindu & Islamic calendars.
FESTIVALS: dict[str, list[str]] = {
    # Diwali 2024: national holiday 31-Oct, Lakshmi Puja (and the demand dip) on 1-Nov
    "diwali": ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"],
    "holi": ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"],
    "eid_ul_fitr": ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21", "2027-03-10"],
    "eid_ul_adha": ["2023-06-29", "2024-06-17", "2025-06-07", "2026-05-27", "2027-05-17"],
    "dussehra": ["2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20", "2027-10-09"],
    # Durga Puja: Maha Saptami .. Maha Navami (3 days)
    "durga_puja": ["2023-10-21", "2023-10-22", "2023-10-23", "2024-10-10", "2024-10-11",
                   "2024-10-12", "2025-09-29", "2025-09-30", "2025-10-01", "2026-10-17",
                   "2026-10-18", "2026-10-19", "2027-10-06", "2027-10-07", "2027-10-08"],
    "pongal": ["2023-01-15", "2024-01-15", "2025-01-14", "2026-01-15", "2027-01-15"],
    "ganesh_chaturthi": ["2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14", "2027-09-04"],
    "christmas": [f"{y}-12-25" for y in range(2023, 2028)],
}
FESTIVAL_DATES = {f: pd.to_datetime(d) for f, d in FESTIVALS.items()}
ALL_FESTIVAL_DAYS = pd.DatetimeIndex(sorted({d for ds in FESTIVAL_DATES.values() for d in ds}))
DIWALI_WINDOW = (-2, 2)  # Dhanteras .. Bhai Dooj: demand dips for several days
CAP = 30                 # cap for days-to/from-festival features


def national_holidays(years) -> dict[pd.Timestamp, str]:
    """Indian national/gazetted holidays from the `holidays` package (country IN)."""
    import holidays

    hol = holidays.country_holidays("IN", years=list(years))
    return {pd.Timestamp(d): name for d, name in hol.items()}


def calendar_features(dates, extra_holidays=None) -> pd.DataFrame:
    """One row per date (any order/subset). `extra_holidays`: dates to treat as holidays
    (what-if scenarios)."""
    want = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    full = pd.date_range(want.min() - pd.Timedelta(days=1), want.max() + pd.Timedelta(days=1))
    return _calendar_contiguous(full, extra_holidays).loc[want]


def _calendar_contiguous(idx: pd.DatetimeIndex, extra_holidays=None) -> pd.DataFrame:
    years = range(idx.year.min() - 1, idx.year.max() + 2)
    nat = national_holidays(years)
    extra = set(pd.to_datetime(list(extra_holidays or [])).normalize())

    df = pd.DataFrame(index=idx)
    df.index.name = "date"
    dow = idx.day_of_week
    df["dow"] = dow
    df["is_saturday"] = (dow == 5).astype(np.int8)
    df["is_sunday"] = (dow == 6).astype(np.int8)
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    df["month"] = idx.month
    doy = idx.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["is_national_holiday"] = idx.isin(list(nat)).astype(np.int8)
    for f, ds in FESTIVAL_DATES.items():
        df[f"fest_{f}"] = idx.isin(ds).astype(np.int8)
    dw = FESTIVAL_DATES["diwali"]
    df["diwali_window"] = np.array([any(DIWALI_WINDOW[0] <= (d - x).days <= DIWALI_WINDOW[1] for x in dw)
                                    for d in idx], dtype=np.int8)
    df["is_festival"] = idx.isin(ALL_FESTIVAL_DAYS).astype(np.int8)
    if extra:
        mask = idx.isin(list(extra))
        df.loc[mask, "is_national_holiday"] = 1
        df.loc[mask, "is_festival"] = 1
    df["is_holiday"] = ((df["is_national_holiday"] + df["is_festival"]) > 0).astype(np.int8)

    # days to next / since previous major festival (capped)
    fd = ALL_FESTIVAL_DAYS.union(pd.DatetimeIndex(sorted(extra))) if extra else ALL_FESTIVAL_DAYS
    pos = fd.searchsorted(idx)
    nxt = np.array([(fd[p] - d).days if p < len(fd) else CAP for p, d in zip(pos, idx)])
    pos_prev = fd.searchsorted(idx, side="right") - 1
    prv = np.array([(d - fd[p]).days if p >= 0 else CAP for p, d in zip(pos_prev, idx)])
    df["days_to_festival"] = np.minimum(nxt, CAP)
    df["days_from_festival"] = np.minimum(prv, CAP)
    # bridge days: a working day between a holiday and a weekend
    hol = df["is_holiday"].astype(bool) | (dow == 6)
    prev_h = pd.Series(hol, index=idx).shift(1, fill_value=False).to_numpy()
    next_h = pd.Series(hol, index=idx).shift(-1, fill_value=False).to_numpy()
    df["is_bridge"] = (~hol & prev_h & next_h).astype(np.int8)
    return df
