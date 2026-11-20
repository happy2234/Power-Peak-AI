"""Data-quality checks: coverage, failed downloads, gaps, sanity ranges, outliers.

Produces
    data/processed/daily_clean.parquet   full calendar, targets + context, is_imputed flags
    reports/latest/data_quality.md       human-readable summary, rewritten every run (git-ignored)
    reports/data_quality.md              the same, committed — only via `python -m src.data.quality --full`
    reports/data_quality_retries.json    outcome of the last retry per missing date (kept between runs)

Rule: gaps are never filled silently. Gaps of <= MAX_IMPUTE_GAP consecutive days are linearly
interpolated and flagged (`is_imputed`, per-target `<target>_imputed`); longer gaps stay NaN.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import PROCESSED, RAW, REGIONS, REPORTS, ROOT, START_DATE, TARGETS

log = logging.getLogger(__name__)

MAX_IMPUTE_GAP = 2
RANGES = {  # plausible ranges (sanity checks)
    "solar_peak": (150_000, 300_000),
    "evening_peak": (130_000, 300_000),
    "energy": (3000, 6000),
}
CONTEXT_COLS = (
    ["max_demand_mw_all_india", "evening_peak_demand_mw_all_india", "energy_shortage_mu_all_india",
     "solar_gen_mu_all_india", "wind_gen_mu_all_india", "hydro_gen_mu_all_india",
     "peak_demand_solar_hr_time", "peak_demand_non_solar_hr_time", "max_demand_time_all_india"]
    + [f"energy_met_mu_{r}" for r in REGIONS] + [f"max_demand_mw_{r}" for r in REGIONS]
)


@dataclass
class QualityReport:
    start: pd.Timestamp
    end: pd.Timestamp
    n_expected: int = 0
    n_present: int = 0
    missing_days: list = field(default_factory=list)
    log_failures: list = field(default_factory=list)       # (date/file, reason)
    retried: dict = field(default_factory=dict)            # date -> outcome
    target_missing: dict = field(default_factory=dict)     # target -> list of dates (after impute)
    imputed: dict = field(default_factory=dict)            # target -> list of dates
    out_of_range: list = field(default_factory=list)       # (date, target, value)
    outliers: list = field(default_factory=list)           # (date, target, value, robust z)


# ----------------------------------------------------------------------------- log parsing
_LOG_PSP = re.compile(r"(ERROR|WARNING)\s+(\d{4}-\d{2}-\d{2}): (.+)")
_LOG_PARSE = re.compile(r"WARNING\s+ (\S.*?\.(?:xls|xlsx|pdf)) -> (.+)")


def parse_backfill_log(path: Path) -> list[tuple[str, str, str]]:
    """[(kind, item, reason)] for PSP download failures/not-listed and parse failures."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if (m := _LOG_PSP.search(line)) and "forecast" not in line:
            kind = "not_listed" if "not listed" in m[3] else "download_error"
            out.append((kind, m[2], m[3].strip()))
        elif (m := _LOG_PARSE.search(line)):
            out.append(("parse_error", m[1].strip(), m[2].strip()))
    # keep last occurrence per item
    return list({(k, i): (k, i, r) for k, i, r in out}.values())


# ----------------------------------------------------------------------------- retries
RETRIES_FILE = REPORTS / "data_quality_retries.json"


def merge_retries(saved: dict[str, str], new: dict[str, str], missing: list[str]) -> dict[str, str]:
    """Retry outcomes to report: the latest outcome per date, only for dates that are still missing.
    A run without retries (the daily pipeline) keeps the earlier outcomes instead of dropping them."""
    merged = {**saved, **new}
    return {d: merged[d] for d in sorted(merged) if d in set(missing)}


def retry_missing(dates: list[pd.Timestamp]) -> dict[str, str]:
    """Try each missing date once more (fresh listing); rebuild the dataset if anything arrived."""
    if not dates:
        return {}
    from src.fetch.grid_india import PSP_TYPE, GridIndiaClient, financial_year

    client = GridIndiaClient(cache_dir=RAW / "_index")
    for fy in {financial_year(d.date()) for d in dates}:
        try:
            client.list_files(PSP_TYPE, fy, use_cache=False)
        except Exception as e:  # network down etc.
            log.warning("listing refresh failed for %s: %s", fy, e)
    outcome, got = {}, False
    for d in dates:
        try:
            p = client.download_psp(d.date(), RAW / "psp")
            outcome[str(d.date())] = f"downloaded {p.name}" if p else "still not listed"
            got |= p is not None
        except Exception as e:
            outcome[str(d.date())] = f"failed again: {str(e)[:120]}"
    if got:
        from scripts.build_dataset import build

        build(RAW, PROCESSED)
    return outcome


# ----------------------------------------------------------------------------- core
def build_clean(summary: pd.DataFrame, end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Full-calendar daily table with targets, context columns and imputation flags."""
    s = summary.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.drop_duplicates("date", keep="last").set_index("date").sort_index()
    end = end or s.index.max()
    idx = pd.date_range(START_DATE, end, freq="D", name="date")
    cols = list(TARGETS.values()) + [c for c in CONTEXT_COLS if c in s.columns]
    df = s.reindex(idx)[cols].rename(columns={v: k for k, v in TARGETS.items()})
    df["is_missing_report"] = ~idx.isin(s.index)

    imputed: dict[str, list] = {}
    numeric = [c for c in df.columns if df[c].dtype.kind == "f"]
    df["is_imputed"] = False
    for c in numeric:
        isna = df[c].isna()
        if not isna.any():
            continue
        # length of each NaN run
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        # only interior gaps (bounded on both sides by data) of length <= MAX_IMPUTE_GAP
        first_valid, last_valid = df[c].first_valid_index(), df[c].last_valid_index()
        interior = (df.index > first_valid) & (df.index < last_valid)
        fill = isna & (run_len <= MAX_IMPUTE_GAP) & interior
        if fill.any():
            df.loc[fill, c] = df[c].interpolate("time")[fill]
            if c in TARGETS:
                df[f"{c}_imputed"] = fill
                imputed[c] = [str(d.date()) for d in df.index[fill]]
            df.loc[fill, "is_imputed"] = True
    for t in TARGETS:
        if f"{t}_imputed" not in df:
            df[f"{t}_imputed"] = False
    return df.reset_index(), imputed


def find_outliers(df: pd.DataFrame, z_thresh: float = 6.0) -> list[tuple]:
    """Robust check: residual vs a centred 15-day rolling median, scaled by rolling MAD."""
    out = []
    for t in TARGETS:
        x = df.set_index("date")[t]
        med = x.rolling(15, center=True, min_periods=7).median()
        resid = x - med
        mad = resid.abs().rolling(61, center=True, min_periods=20).median() * 1.4826
        z = resid / mad
        for d in z[z.abs() > z_thresh].index:
            out.append((str(d.date()), t, float(x[d]), round(float(z[d]), 1)))
    return out


LATEST_REPORT = REPORTS / "latest" / "data_quality.md"   # rewritten by every pipeline run (git-ignored)
FULL_REPORT = REPORTS / "data_quality.md"                # committed; written only on request


def run_quality(retry: bool = True, log_path: Path | None = None,
                report_path: Path | None = None) -> tuple[pd.DataFrame, QualityReport]:
    """report_path: where the Markdown summary goes (default: the git-ignored daily copy in
    reports/latest/; `python -m src.data.quality --full` writes the committed report)."""
    log_path = log_path or ROOT / "backfill.log"
    summary = pd.read_parquet(PROCESSED / "daily_summary.parquet")
    end = pd.Timestamp(summary["date"].max())
    expected = pd.date_range(START_DATE, end, freq="D")
    present = set(pd.to_datetime(summary["date"]))
    missing = [d for d in expected if d not in present]

    rep = QualityReport(start=expected[0], end=end, n_expected=len(expected))
    rep.log_failures = parse_backfill_log(log_path)
    new_retries: dict[str, str] = {}
    if retry and missing:
        new_retries = retry_missing(missing)
        summary = pd.read_parquet(PROCESSED / "daily_summary.parquet")
        present = set(pd.to_datetime(summary["date"]))
        missing = [d for d in expected if d not in present]
    rep.n_present = len(expected) - len(missing)
    rep.missing_days = [str(d.date()) for d in missing]
    import json

    try:
        saved = json.loads(RETRIES_FILE.read_text()) if RETRIES_FILE.exists() else {}
    except (OSError, ValueError):
        saved = {}
    rep.retried = merge_retries(saved, new_retries, rep.missing_days)
    if new_retries or rep.retried != saved:
        RETRIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        RETRIES_FILE.write_text(json.dumps(rep.retried, indent=2) + "\n")

    df, rep.imputed = build_clean(summary, end)
    for t in TARGETS:
        rep.target_missing[t] = [str(d.date()) for d in df.loc[df[t].isna(), "date"]]
        lo, hi = RANGES[t]
        bad = df[(df[t] < lo) | (df[t] > hi)]
        rep.out_of_range += [(str(r.date.date()), t, float(r[t])) for _, r in bad.iterrows()]
    rep.outliers = find_outliers(df)
    df["is_outlier"] = df["date"].dt.strftime("%Y-%m-%d").isin({o[0] for o in rep.outliers})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "daily_clean.parquet", index=False)
    write_report(rep, df, report_path or LATEST_REPORT)
    return df, rep


def _ranges(dates: list[str]) -> str:
    """Compress consecutive dates: 2023-04-01..2023-04-29 (29 d)."""
    if not dates:
        return "none"
    ds = pd.to_datetime(sorted(dates))
    parts, start, prev = [], ds[0], ds[0]
    for d in list(ds[1:]) + [None]:
        if d is not None and (d - prev).days == 1:
            prev = d
            continue
        n = (prev - start).days + 1
        parts.append(f"{start.date()}" if n == 1 else f"{start.date()}..{prev.date()} ({n} d)")
        if d is not None:
            start = prev = d
    return ", ".join(parts)


def write_report(rep: QualityReport, df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or LATEST_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# Data quality report\n",
         f"Period **{rep.start.date()} .. {rep.end.date()}** — {rep.n_present}/{rep.n_expected} daily "
         f"reports present ({rep.n_present / rep.n_expected:.1%}).\n",
         "## Missing daily reports", f"{_ranges(rep.missing_days)}\n"]
    if rep.retried:
        L += ["### Retry (once)", *[f"- {d}: {o}" for d, o in rep.retried.items()], ""]
    fails = [f for f in rep.log_failures if f[0] != "parse_error"]
    parse = [f for f in rep.log_failures if f[0] == "parse_error"]
    L += ["## Failures recorded in backfill.log",
          f"Download problems: {len(fails)}"] + [f"- `{i}` — {k}: {r}" for k, i, r in fails]
    L += [f"\nParse failures: {len(parse)}"] + [f"- `{i}` — {r}" for _, i, r in parse] + [""]
    L += ["## Targets", "| target | available | still missing (not imputed) | imputed (gap ≤ 2 d) | min | median | max |",
          "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        x = df[t]
        L.append(f"| {t} | {x.notna().sum()} | {_ranges(rep.target_missing[t])} | "
                 f"{_ranges(rep.imputed.get(t, []))} | {x.min():,.0f} | {x.median():,.0f} | {x.max():,.0f} |")
    L += ["", "Missing solar/evening peaks in April 2023 are a source limitation (those reports have no "
          "section I). They are left as NaN; models mask them.", ""]
    L += ["## Sanity ranges", *[f"- {t}: {lo:,}–{hi:,}" for t, (lo, hi) in RANGES.items()],
          f"\nOut-of-range values: {len(rep.out_of_range)}"]
    L += [f"- {d} {t} = {v:,.0f}" for d, t, v in rep.out_of_range]
    L += ["", "## Outliers (|robust z| > 6 vs 15-day rolling median)", f"{len(rep.outliers)} flagged "
          "(kept in the data, flagged `is_outlier`; most are genuine holidays/storms):"]
    L += [f"- {d} {t} = {v:,.0f} (z={z})" for d, t, v, z in rep.outliers]
    path.write_text("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    out = FULL_REPORT if "--full" in sys.argv else LATEST_REPORT
    df, rep = run_quality(report_path=out)
    print(out.read_text())
