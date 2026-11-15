"""Parsers for Grid-India (NLDC) daily Power Supply Position (PSP) reports (.xls / .xlsx).

Design rule: never hard-code row/column numbers. Every block is located by matching
label text (case-insensitive, whitespace/newline/footnote tolerant), because the layout
drifts between years (e.g. 2023 files have one row fewer at the top, slightly different
labels, and no TimeSeries sheet at all).

The *data date* is taken from the filename (DD.MM.YY_NLDC_PSP*.xls); the report itself is
published the next day ("Date of Reporting").
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REGIONS = ["NR", "WR", "SR", "ER", "NER"]
_FILENAME_DATE = re.compile(r"(\d{2})\.(\d{2})\.(\d{2,4})")
_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")

# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def norm(value) -> str:
    """Normalise a label: lowercase, footnote marks removed, whitespace collapsed."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰*#]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def to_num(value) -> float:
    """Convert a cell to float; '-', '', 'NA' and unparsable strings become NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().replace(",", "")
    if s in {"", "-", "--", "na", "NA", "N/A", "nan"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_hhmm(value) -> str | None:
    """Normalise a time-of-day cell (str '9:42', datetime.time, Excel day-fraction) to 'HH:MM'."""
    td = to_timedelta(value)
    if td is None:
        return None
    mins = int(round(td.total_seconds() / 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def to_timedelta(value) -> pd.Timedelta | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dt.datetime):
        value = value.time()
    if isinstance(value, dt.time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, dt.timedelta):
        return pd.Timedelta(value)
    if isinstance(value, (int, float, np.number)) and 0 <= float(value) <= 1:
        return pd.Timedelta(days=float(value)).round("min")  # Excel fraction of a day
    m = _HHMM.match(str(value))
    if m:
        h, mi, se = int(m[1]), int(m[2]), int(m[3] or 0)
        if h <= 24 and mi < 60:
            return pd.Timedelta(hours=h, minutes=mi, seconds=se)
    return None


def snake(label: str) -> str:
    s = norm(label)
    s = re.sub(r"\(?[+-]ve\b.*$", "", s)  # drop trailing sign notes like "+ve Imp,-ve Exp"
    s = s.replace("&", "_").replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def date_from_filename(path: str | Path) -> dt.date:
    """'01.11.26_NLDC_PSP_491.xls' -> 2026-11-01 (the data date)."""
    m = _FILENAME_DATE.search(Path(path).name)
    if not m:
        raise ValueError(f"Cannot find DD.MM.YY date in filename: {path}")
    d, mo, y = int(m[1]), int(m[2]), int(m[3])
    if y < 100:
        y += 2000
    return dt.date(y, mo, d)


@lru_cache(maxsize=8)
def _read_cached(path: str, mtime: float) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, header=None, engine="calamine")


def read_sheets(path: str | Path) -> dict[str, pd.DataFrame]:
    """All sheets as raw object grids (header=None). Cached so the 3 parsers read once."""
    p = Path(path)
    return _read_cached(str(p.resolve()), p.stat().st_mtime)


def find_sheet(sheets: dict[str, pd.DataFrame], name: str) -> pd.DataFrame | None:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for sname, df in sheets.items():
        if re.sub(r"[^a-z0-9]", "", sname.lower()) == key:
            return df
    return None


def _normed_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(norm))


def find_cells(grid: pd.DataFrame, pattern: str, rows: range | None = None) -> list[tuple[int, int]]:
    """(row, col) of normalised cells matching regex `pattern` (re.search)."""
    rx = re.compile(pattern)
    out = []
    for r in rows if rows is not None else range(len(grid)):
        for c in range(grid.shape[1]):
            if rx.search(grid.iat[r, c]):
                out.append((r, c))
    return out


def _row_label(grid: pd.DataFrame, r: int) -> str:
    """First non-empty normalised cell in a row (labels may sit in col 0 or a merged col)."""
    for c in range(grid.shape[1]):
        if grid.iat[r, c]:
            return grid.iat[r, c]
    return ""


def _section_start(grid: pd.DataFrame, letter: str, keyword: str) -> int | None:
    """Row whose leading label looks like '<letter>. ... <keyword> ...'."""
    rx = re.compile(rf"^{letter}\s*[.)]\s*.*{keyword}")
    for r in range(len(grid)):
        if rx.search(_row_label(grid, r)):
            return r
    return None


def _next_section(grid: pd.DataFrame, start: int) -> int:
    rx = re.compile(r"^[a-z]\s*[.)]\s+\S")
    for r in range(start + 1, len(grid)):
        lab = _row_label(grid, r)
        if rx.match(lab) and len(lab) > 6:
            return r
    return len(grid)


def report_date(sheets: dict[str, pd.DataFrame]) -> dt.date | None:
    """'Date of Reporting' value from MOP_E (None in older files that lack it)."""
    raw = find_sheet(sheets, "MOP_E")
    if raw is None:
        return None
    grid = _normed_grid(raw)
    for r, c in find_cells(grid, r"date of reporting", range(min(10, len(grid)))):
        for cc in range(c + 1, grid.shape[1]):
            v = raw.iat[r, cc]
            if isinstance(v, (dt.date, dt.datetime)):
                return v.date() if isinstance(v, dt.datetime) else v
            if norm(v):
                try:
                    return pd.to_datetime(str(v), dayfirst=True).date()
                except (ValueError, TypeError):
                    continue
    return None


# --------------------------------------------------------------------------------------
# 1. Daily summary (MOP_E section A + I)
# --------------------------------------------------------------------------------------

# metric name -> regex on the normalised row label. Order matters (first match wins).
SUMMARY_ROWS: dict[str, str] = {
    "evening_peak_demand_mw": r"^demand met.*evening peak",
    "evening_peak_shortage_mw": r"^(sho\w*ge.*evening peak|peak shortage)",
    "energy_met_mu": r"^energy met",
    "hydro_gen_mu": r"^hydro gen",
    "wind_gen_mu": r"^wind gen",
    "solar_gen_mu": r"^solar gen",
    "energy_shortage_mu": r"^energy shortage",
    "max_demand_mw": r"^maximum demand met during the day",
    "max_demand_time": r"^time of maximum demand",
}
_REGION_KEY = {"nr": "nr", "wr": "wr", "sr": "sr", "er": "er", "ner": "ner", "total": "all_india",
               "all india": "all_india"}


def parse_daily_summary(path: str | Path) -> pd.DataFrame:
    """One row per file: all-India + regional energy met, peak demand & time, evening peak,
    hydro/wind/solar generation, shortages; plus solar-hour / non-solar-hour all-India peaks
    (section I) when present.

    Columns are `<metric>_<region>` with region in {nr, wr, sr, er, ner, all_india}.
    """
    sheets = read_sheets(path)
    raw = find_sheet(sheets, "MOP_E")
    if raw is None:
        raise ValueError(f"{path}: no MOP_E sheet")
    grid = _normed_grid(raw)
    data_date = date_from_filename(path)

    start = _section_start(grid, "a", "power supply position")
    if start is None:
        raise ValueError(f"{path}: section A not found")
    end = _next_section(grid, start)

    # Header row with region names, e.g. ['', '', 'nr', 'wr', 'sr', 'er', 'ner', 'total']
    region_cols: dict[str, int] = {}
    header_row = None
    for r in range(start, end):
        cols = {grid.iat[r, c]: c for c in range(grid.shape[1]) if grid.iat[r, c] in _REGION_KEY}
        if {"nr", "wr", "sr", "er", "ner"} <= cols.keys():
            region_cols = {_REGION_KEY[k]: c for k, c in cols.items()}
            header_row = r
            break
    if header_row is None:
        raise ValueError(f"{path}: region header (NR WR SR ER NER TOTAL) not found in section A")

    row: dict[str, object] = {"date": pd.Timestamp(data_date)}
    rd = report_date(sheets)
    row["report_date"] = pd.Timestamp(rd) if rd else pd.NaT
    if rd and (rd - data_date).days != 1:
        log.warning("%s: report date %s is not data date %s + 1", Path(path).name, rd, data_date)

    used_rows: set[int] = set()
    for metric, pattern in SUMMARY_ROWS.items():
        rx = re.compile(pattern)
        match = next((r for r in range(header_row + 1, end)
                      if r not in used_rows and rx.search(_row_label(grid, r))), None)
        if match is None:
            log.warning("%s: summary row %r not found", Path(path).name, metric)
        else:
            used_rows.add(match)
        for region, c in region_cols.items():
            v = raw.iat[match, c] if match is not None else None
            row[f"{metric}_{region}"] = to_hhmm(v) if metric.endswith("_time") else to_num(v)

    row.update(_parse_solar_nonsolar_peaks(raw, grid))
    return pd.DataFrame([row])


def _parse_solar_nonsolar_peaks(raw: pd.DataFrame, grid: pd.DataFrame) -> dict[str, object]:
    """Section I: 'All India Peak Demand and shortage at Solar and Non-Solar Hour'.
    Row label 'Solar hr' / 'Non-Solar hr' followed by Max Demand (MW), Time, Shortage (MW)."""
    out: dict[str, object] = {}
    for key, pattern in [("solar_hr", r"^solar hr"), ("non_solar_hr", r"^non[- ]?solar hr")]:
        cells = find_cells(grid, pattern)
        vals = []
        if cells:
            r, c = cells[0]
            vals = [raw.iat[r, cc] for cc in range(c + 1, grid.shape[1]) if grid.iat[r, cc]]
        out[f"peak_demand_{key}_mw"] = to_num(vals[0]) if len(vals) > 0 else np.nan
        out[f"peak_demand_{key}_time"] = to_hhmm(vals[1]) if len(vals) > 1 else None
        out[f"peak_shortage_{key}_mw"] = to_num(vals[2]) if len(vals) > 2 else np.nan
    return out


# --------------------------------------------------------------------------------------
# 2. State-level demand (MOP_E section C)
# --------------------------------------------------------------------------------------

# Canonical state -> region map (key = lowercase alphanumerics only). The sheet only prints
# the region label once per group, on the *middle* row, so this map is the primary source;
# in-sheet labels are a cross-check and nearest-label is the fallback for unknown names.
_STATE_REGION_RAW = {
    "NR": ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttar Pradesh", "Uttarakhand", "HP",
           "Himachal Pradesh", "J&K(UT) & Ladakh(UT)", "J&K", "Ladakh", "Chandigarh"],
    "WR": ["Chhattisgarh", "Gujarat", "MP", "Madhya Pradesh", "Maharashtra", "Goa", "DNHDDPDCL",
           "DD", "DNH", "Daman & Diu", "Dadra & Nagar Haveli", "AMNSIL", "BALCO", "RIL Jamnagar"],
    "SR": ["Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu", "Puducherry"],
    "ER": ["Bihar", "DVC", "Jharkhand", "Odisha", "West Bengal", "Sikkim",
           "Andaman & Nicobar", "Andaman- Nicobar"],
    "NER": ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura"],
}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


STATE_REGION = {_key(s): reg for reg, states in _STATE_REGION_RAW.items() for s in states}

STATE_COLUMNS: dict[str, str] = {  # output column -> regex on concatenated header text
    "max_demand_mw": r"max\.? ?demand met",
    "shortage_at_max_demand_mw": r"shortage (at|during) max",
    "energy_met_mu": r"energy met",
    "drawal_schedule_mu": r"drawal",
    "od_ud_mu": r"^od.*\(mu\)",
    "energy_shortage_mu": r"energy shortage",
}


def parse_state_demand(path: str | Path) -> pd.DataFrame:
    """Long table: date, region, state, max_demand_mw, energy_met_mu, shortages, drawal..."""
    sheets = read_sheets(path)
    raw = find_sheet(sheets, "MOP_E")
    if raw is None:
        raise ValueError(f"{path}: no MOP_E sheet")
    grid = _normed_grid(raw)
    data_date = date_from_filename(path)

    start = _section_start(grid, "c", "power supply position in states")
    if start is None:
        raise ValueError(f"{path}: section C (states) not found")
    end = _next_section(grid, start)

    # Header: the row containing 'states'; header text may span that row and the one above.
    hdr = next((r for r in range(start, end) if any(grid.iat[r, c] in {"states", "state"}
                                                     for c in range(grid.shape[1]))), None)
    if hdr is None:
        raise ValueError(f"{path}: 'States' header not found in section C")
    state_col = next(c for c in range(grid.shape[1]) if grid.iat[hdr, c] in {"states", "state"})
    region_col = next((c for c in range(state_col) if grid.iat[hdr, c].startswith("region")), 0)
    header_text = {c: " ".join(t for t in (grid.iat[hdr - 1, c], grid.iat[hdr, c]) if t)
                   for c in range(grid.shape[1])}

    col_idx: dict[str, int] = {}
    for name, pattern in STATE_COLUMNS.items():
        rx = re.compile(pattern)
        c = next((c for c, t in header_text.items() if rx.search(t) and c not in col_idx.values()),
                 None)
        if c is None:
            log.warning("%s: state column %r not found", Path(path).name, name)
        else:
            col_idx[name] = c

    records = []
    for r in range(hdr + 1, end):
        state = raw.iat[r, state_col]
        if not norm(state):
            continue
        if "max_demand_mw" in col_idx and np.isnan(to_num(raw.iat[r, col_idx["max_demand_mw"]])) \
                and np.isnan(to_num(raw.iat[r, col_idx.get("energy_met_mu", state_col)])):
            continue  # not a data row (e.g. stray note)
        rec = {"row": r, "state": re.sub(r"\s+", " ", str(state)).strip(),
               "sheet_region": grid.iat[r, region_col].upper() or None}
        for name, c in col_idx.items():
            rec[name] = to_num(raw.iat[r, c])
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"{path}: no state rows parsed")
    df["region"] = _assign_regions(df, Path(path).name)
    df.insert(0, "date", pd.Timestamp(data_date))
    cols = ["date", "region", "state", *[c for c in STATE_COLUMNS if c in df.columns]]
    return df[cols].reset_index(drop=True)


def _assign_regions(df: pd.DataFrame, fname: str) -> pd.Series:
    labelled = df[df["sheet_region"].isin(REGIONS)]
    regions = []
    for _, rec in df.iterrows():
        k = _key(rec["state"])
        bulk = re.search(r"_(nr|wr|sr|er|ner)\b", rec["state"].lower())
        if rec["sheet_region"] in REGIONS:
            reg = rec["sheet_region"]
            if k in STATE_REGION and STATE_REGION[k] != reg:
                log.warning("%s: %s labelled %s in sheet but mapped to %s; using sheet label",
                            fname, rec["state"], reg, STATE_REGION[k])
        elif k in STATE_REGION:
            reg = STATE_REGION[k]
        elif bulk:
            reg = bulk[1].upper()
        elif not labelled.empty:
            nearest = (labelled["row"] - rec["row"]).abs().idxmin()
            reg = labelled.at[nearest, "sheet_region"]
            log.warning("%s: unknown state %r, assigned nearest region label %s",
                        fname, rec["state"], reg)
        else:
            reg = None
        regions.append(reg)
    return pd.Series(regions, index=df.index)


# --------------------------------------------------------------------------------------
# 3. 15-minute all-India time series (TimeSeries sheet; absent before ~2024)
# --------------------------------------------------------------------------------------


def parse_timeseries(path: str | Path) -> pd.DataFrame:
    """96 rows (00:00..23:45) with a full `timestamp` and every generation/demand column.

    Returns an empty DataFrame if the file has no TimeSeries sheet (older reports).
    """
    sheets = read_sheets(path)
    raw = find_sheet(sheets, "TimeSeries")
    if raw is None:
        log.info("%s: no TimeSeries sheet", Path(path).name)
        return pd.DataFrame()
    grid = _normed_grid(raw)
    data_date = pd.Timestamp(date_from_filename(path))

    hdr = next((r for r in range(len(grid)) if grid.iat[r, 0] == "time"), None)
    if hdr is None:
        raise ValueError(f"{path}: TIME header not found in TimeSeries")

    names: dict[int, str] = {}
    for c in range(1, grid.shape[1]):
        label = raw.iat[hdr, c]
        if not norm(label):
            continue
        name = snake(label)
        while name in names.values():  # guard against duplicate headers
            name += "_dup"
        names[c] = name

    records = []
    for r in range(hdr + 1, len(raw)):
        td = to_timedelta(raw.iat[r, 0])
        if td is None:
            continue  # unit row "(A) (B) ..." or footnotes
        rec = {"timestamp": data_date + td}
        rec.update({n: to_num(raw.iat[r, c]) for c, n in names.items()})
        records.append(rec)

    df = pd.DataFrame(records)
    if len(df) != 96:
        log.warning("%s: TimeSeries has %d rows (expected 96)", Path(path).name, len(df))
    df.insert(0, "date", data_date)
    return df.reset_index(drop=True)
