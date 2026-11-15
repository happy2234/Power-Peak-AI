"""The daily pipeline: one implementation shared by scripts/daily_run.sh (systemd timer / cron)
and the app's "Fetch latest data and update forecast" button.

    python -m src.pipeline                      # trigger=manual
    python -m src.pipeline --trigger timer      # what scripts/daily_run.sh runs

Steps (each reports pending / running / done / no new data / failed / skipped and its duration):
    1 psp          Download new Grid-India PSP reports
    2 gi_forecast  Download the latest Grid-India week-ahead forecast
    3 weather      Update weather forecasts (Open-Meteo)
    4 datasets     Parse files and update the datasets
    5 features     Rebuild features
    6 forecast     Generate the new 7-day forecast
    7 scorecard    Update the scorecard (newly scored days: actual vs our model vs Grid-India)

Download steps (1–3) are best-effort: if one fails, the run continues with the data already on disk.
A failure in steps 4–7 stops the run; the remaining steps are marked skipped. Every step writes
its outputs atomically as it finishes, so the results of earlier steps stay saved.

A lock file (logs/pipeline.lock) prevents two runs from overlapping; every run is appended to
logs/pipeline_runs.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src import fmt
from src.config import PROCESSED, RAW, ROOT, TARGETS

log = logging.getLogger("pipeline")

PENDING, RUNNING, DONE, NO_DATA, FAILED, SKIPPED = "pending", "running", "done", "no new data", "failed", "skipped"
COMPLETED = (DONE, NO_DATA)

STEPS: list[tuple[str, str, str]] = [   # key, label, short label for the flow diagram
    ("psp", "Download new Grid-India PSP reports", "PSP reports"),
    ("gi_forecast", "Download the latest Grid-India week-ahead forecast", "Week-ahead forecast"),
    ("weather", "Update weather forecasts (Open-Meteo)", "Weather"),
    ("datasets", "Parse files and update the datasets", "Datasets"),
    ("features", "Rebuild features", "Features"),
    ("forecast", "Generate the new 7-day forecast", "7-day forecast"),
    ("scorecard", "Update the scorecard", "Scorecard"),
]
SOFT_STEPS = {"psp", "gi_forecast", "weather"}   # failure does not stop the run
PSP_LOOKBACK_DAYS = 7          # always re-check the last week (late uploads)
PSP_MAX_CATCHUP_DAYS = 45      # after a long pause, fetch at most this many days per run
WEATHER_REFRESH_DAYS = 14
LOCK_MAX_AGE = dt.timedelta(hours=3)
HISTORY_FILE = "forecast_history.parquet"
SCORECARD_FILE = "live_scorecard.parquet"


# ============================================================================ data classes
@dataclass
class Paths:
    raw: Path = RAW
    processed: Path = PROCESSED
    logs: Path = ROOT / "logs"

    @property
    def lock(self) -> Path:
        return self.logs / "pipeline.lock"

    @property
    def runs_log(self) -> Path:
        """Run history. A read-only deployment has no logs/ folder; the scheduled job copies the
        recent runs next to the data (data/processed/pipeline_runs.jsonl) for it to show."""
        from src.config import readonly

        return self.processed / "pipeline_runs.jsonl" if readonly() else self.logs / "pipeline_runs.jsonl"

    @property
    def progress(self) -> Path:
        """Live state of the run in progress (rewritten after every step change); read by the app."""
        return self.logs / "pipeline_progress.json"


@dataclass
class StepResult:
    key: str
    label: str
    short: str
    status: str = PENDING
    started: str | None = None
    duration_s: float | None = None
    message: str = ""
    details: dict = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


@dataclass
class RunResult:
    run_id: str
    trigger: str
    started: str
    steps: list[StepResult]
    finished: str | None = None
    status: str = "running"          # running | success | partial | failed
    summary: str = ""
    duration_s: float | None = None
    changes: dict = field(default_factory=dict)   # what the run changed: see what_changed()

    def step(self, key: str) -> StepResult:
        return next(s for s in self.steps if s.key == key)

    @property
    def failed_step(self) -> StepResult | None:
        return next((s for s in self.steps if s.status == FAILED), None)

    def to_dict(self) -> dict:
        return asdict(self)


class PipelineLocked(RuntimeError):
    def __init__(self, info: dict):
        self.info = info
        super().__init__(f"An update is already running (started {self.started_hhmm})")

    @property
    def started_hhmm(self) -> str:
        try:
            return f"{pd.Timestamp(self.info['started']):%H:%M}"
        except Exception:
            return "unknown time"


@dataclass
class Context:
    """What the steps share: paths, the Grid-India client (created lazily), today's date, and
    facts earlier steps found out (state) for the summary."""
    paths: Paths
    today: dt.date
    live_weather: bool = True
    client_factory: Callable | None = None
    state: dict = field(default_factory=dict)
    _client: object = None

    @property
    def client(self):
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from src.fetch.grid_india import GridIndiaClient

                self._client = GridIndiaClient(cache_dir=self.paths.raw / "_index", delay=1.0)
        return self._client

    @property
    def yesterday(self) -> dt.date:
        return self.today - dt.timedelta(days=1)


# ============================================================================ lock
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(paths: Paths) -> dict | None:
    """Info about the run holding the lock, or None. A stale lock (dead process on this host,
    or older than LOCK_MAX_AGE) is removed."""
    try:
        info = json.loads(paths.lock.read_text())
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        info = {}
    if not info:                    # unreadable: a lock written by an older version mid-write counts as held for a moment
        try:
            if time.time() - paths.lock.stat().st_mtime < 10:
                return {"started": pd.Timestamp.now().isoformat(timespec="seconds")}
        except OSError:
            return None
    started = pd.Timestamp(info["started"]) if info.get("started") else None
    dead = info.get("host") == socket.gethostname() and info.get("pid") and not _pid_alive(int(info["pid"]))
    too_old = started is None or pd.Timestamp.now() - started > LOCK_MAX_AGE
    if dead or too_old:
        log.warning("removing stale pipeline lock %s", info)
        paths.lock.unlink(missing_ok=True)
        return None
    return info


@contextmanager
def pipeline_lock(paths: Paths, trigger: str, run_id: str | None = None):
    paths.logs.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "host": socket.gethostname(), "trigger": trigger, "run_id": run_id,
            "started": pd.Timestamp.now().isoformat(timespec="seconds")}
    # Create the lock with its content already in it: os.link is atomic and fails if the lock exists, so
    # another process can never read a half-written (empty) lock and mistake it for a stale one.
    tmp = paths.lock.with_name(f"{paths.lock.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(info))
    try:
        for attempt in range(2):
            try:
                os.link(tmp, paths.lock)
                break
            except FileExistsError:
                held = read_lock(paths)
                if held is not None or attempt == 1:
                    raise PipelineLocked(held or {}) from None
    finally:
        tmp.unlink(missing_ok=True)
    try:
        yield info
    finally:
        paths.lock.unlink(missing_ok=True)


# ============================================================================ progress and changes
def write_progress(paths: Paths, run: RunResult) -> None:
    """Save the live state of `run` (atomic) so any page load, including one after a browser
    refresh, can show it. Never raises: progress is a convenience, the run must not depend on it."""
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        tmp = paths.progress.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(run.to_dict(), default=str))
        tmp.replace(paths.progress)
    except Exception:
        log.debug("could not write pipeline progress", exc_info=True)


def read_progress(paths: Paths | None = None) -> dict | None:
    try:
        return json.loads((paths or Paths()).progress.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _snapshot(paths: Paths) -> dict:
    """The facts 'What changed' compares before and after a run."""
    snap: dict = {"latest_data": None, "forecast_origin": None, "scorecard_dates": []}
    try:
        d = _last_processed_date(paths)
        snap["latest_data"] = str(d) if d else None
    except Exception:
        pass
    try:
        snap["forecast_origin"] = str(pd.Timestamp(json.loads((paths.processed / "latest_meta.json")
                                                              .read_text())["origin"]).date())
    except Exception:
        pass
    try:
        p = paths.processed / SCORECARD_FILE
        if p.exists():
            snap["scorecard_dates"] = sorted({str(pd.Timestamp(x).date())
                                              for x in pd.read_parquet(p, columns=["date"])["date"]})
    except Exception:
        pass
    return snap


def what_changed(before: dict, after: dict) -> dict:
    """Latest data date before/after, forecast days that are new, and days added to the live scorecard."""
    def days(origin):
        return {str((pd.Timestamp(origin) + pd.Timedelta(days=i)).date()) for i in range(1, 8)} if origin else set()

    new_fc = sorted(days(after["forecast_origin"]) - days(before["forecast_origin"]))
    new_sc = sorted(set(after["scorecard_dates"]) - set(before["scorecard_dates"]))
    return {"data_before": before["latest_data"], "data_after": after["latest_data"],
            "forecast_before": before["forecast_origin"], "forecast_after": after["forecast_origin"],
            "forecast_new_dates": new_fc, "scorecard_days_added": len(new_sc), "scorecard_new_dates": new_sc}


# ============================================================================ helpers
def _dates_text(dates: list[dt.date]) -> str:
    """'02–08 Nov 2026' for a contiguous run, otherwise a comma-separated list."""
    if not dates:
        return ""
    ds = sorted(dates)
    if (ds[-1] - ds[0]).days == len(ds) - 1:
        return fmt.date_range(ds[0], ds[-1])
    return ", ".join(fmt.date(d) for d in ds)


def _rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows if path.exists() else 0
    except Exception:
        return len(pd.read_parquet(path)) if path.exists() else 0


def _psp_exists(psp_dir: Path, d: dt.date) -> bool:
    return any(p.suffix.lower() in (".xls", ".xlsx") and p.stat().st_size > 0
               for p in psp_dir.glob(f"{d:%d.%m.%y}_NLDC_PSP*"))


def _last_processed_date(paths: Paths) -> dt.date | None:
    p = paths.processed / "daily_summary.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["date"])["date"]
    return d.max().date() if len(d) else None


def _processed_dates(paths: Paths) -> set[dt.date]:
    """Days already parsed into daily_summary.parquet. A checkout without the raw downloads (the
    scheduled GitHub Action) still knows these days, so it does not fetch or report them again."""
    p = paths.processed / "daily_summary.parquet"
    if not p.exists():
        return set()
    return {pd.Timestamp(x).date() for x in pd.read_parquet(p, columns=["date"])["date"]}


def _forecast_period(name: str) -> tuple[dt.date, dt.date] | None:
    from src.fetch.grid_india import _PERIOD

    m = _PERIOD.search(name)
    if not m:
        return None
    return tuple(dt.datetime.strptime(x, "%d-%b-%Y").date() for x in (m[1], m[2]))  # type: ignore[return-value]


# ============================================================================ steps
# Each step returns (status, message, details). Raising marks the step failed.
def step_psp(ctx: Context):
    psp_dir = ctx.paths.raw / "psp"
    psp_dir.mkdir(parents=True, exist_ok=True)
    end = ctx.yesterday
    last = _last_processed_date(ctx.paths)
    start = end - dt.timedelta(days=PSP_LOOKBACK_DAYS - 1)
    if last is not None:
        start = min(start, last + dt.timedelta(days=1))
    start = max(start, end - dt.timedelta(days=PSP_MAX_CATCHUP_DAYS - 1))
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    new, missing, errors = [], [], []
    processed = _processed_dates(ctx.paths)
    for d in dates:
        had = _psp_exists(psp_dir, d)
        if not had and d in processed:      # parsed earlier, raw file not kept on this machine
            continue
        try:
            path = ctx.client.download_psp(d, psp_dir)
        except Exception as e:  # one bad date must not hide the others
            errors.append((d, f"{type(e).__name__}: {e}"))
            continue
        if path is None:
            missing.append(d)
        elif not had:
            new.append(d)
    if errors and len(errors) == len(dates):
        raise RuntimeError(f"Grid-India could not be reached: {errors[-1][1]}")

    ctx.state["new_psp_dates"] = new
    yesterday_ok = _psp_exists(psp_dir, end) or end in processed
    parts = [f"{len(new)} new report{'s' if len(new) != 1 else ''}: {_dates_text(new)}." if new
             else "No new reports."]
    if not yesterday_ok:
        parts.append(f"The report for {fmt.date(end)} is not published yet.")
    if errors:
        parts.append(f"{len(errors)} date(s) could not be downloaded: {_dates_text([d for d, _ in errors])}.")
    details = {"checked": f"{start}..{end}", "new_dates": [str(d) for d in new],
               "not_published": [str(d) for d in missing], "errors": [f"{d}: {e}" for d, e in errors]}
    return (DONE if new else NO_DATA), " ".join(parts), details


def step_gi_forecast(ctx: Context):
    fc_dir = ctx.paths.raw / "forecast"
    fc_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in fc_dir.glob("*.pdf")}
    fd = ctx.paths.processed / "forecast_daily.parquet"      # already parsed, PDF not kept on this machine
    if fd.exists():
        before |= {Path(n).name for n in pd.read_parquet(fd, columns=["source_file"])["source_file"].dropna()}
    paths = ctx.client.download_forecasts(ctx.today - dt.timedelta(days=14), ctx.today + dt.timedelta(days=7), fc_dir)
    new = sorted({Path(p).name for p in paths} - before)
    periods = [p for p in (_forecast_period(n) for n in [*before, *new]) if p]
    latest = max(periods, default=None)
    ctx.state["new_gi_forecasts"] = new
    cover = f"latest covers {fmt.date_range(*latest)}" if latest else "none available locally"
    if new:
        new_periods = [p for p in (_forecast_period(n) for n in new) if p]
        what = ", ".join(fmt.date_range(*p) for p in sorted(new_periods)) or ", ".join(new)
        return DONE, f"New week-ahead forecast for {what}.", {"new_files": new}
    return NO_DATA, f"No new week-ahead forecast ({cover}).", {"new_files": []}


def step_weather(ctx: Context):
    from src.config import WEATHER_DIR
    from src.data.weather import CITIES, fetch_archived_forecasts, fetch_extra, fetch_observed

    start, end = ctx.today - dt.timedelta(days=WEATHER_REFRESH_DAYS), ctx.yesterday
    obs_before = sum(_rows(WEATHER_DIR / f"obs_{c}.parquet") for c in CITIES)
    fc_before = sum(_rows(WEATHER_DIR / f"fc_{c}.parquet") for c in CITIES)
    errors = []
    for city in CITIES:
        try:
            fetch_observed(city, start, end)
            fetch_archived_forecasts(city, start, end)
            fetch_extra(city, start, end, "obs")    # tmin, dew point, clouds, wind, fog proxy
            fetch_extra(city, start, end, "fc")
        except Exception as e:
            errors.append(f"{city}: {type(e).__name__}: {e}")
    if len(errors) == len(CITIES):
        raise RuntimeError(f"Open-Meteo could not be reached: {errors[-1]}")
    obs_added = sum(_rows(WEATHER_DIR / f"obs_{c}.parquet") for c in CITIES) - obs_before
    fc_added = sum(_rows(WEATHER_DIR / f"fc_{c}.parquet") for c in CITIES) - fc_before
    msg = (f"{len(CITIES) - len(errors)} of {len(CITIES)} cities updated: {obs_added:+,} observed rows, "
           f"{fc_added:+,} archived-forecast rows. The live 7-day weather forecast is fetched in step 6.")
    if errors:
        msg += f" Failed: {', '.join(e.split(':')[0] for e in errors)}."
    if not errors and not obs_added and not fc_added:
        return NO_DATA, f"No new data: all {len(CITIES)} cities were already up to date.", \
            {"observed_rows_added": 0, "forecast_rows_added": 0, "errors": []}
    return DONE, msg, {"observed_rows_added": obs_added, "forecast_rows_added": fc_added, "errors": errors}


def step_datasets(ctx: Context):
    from scripts.build_dataset import KEYS, build
    from src.data.quality import run_quality

    before = {n: _rows(ctx.paths.processed / f"{n}.parquet") for n in KEYS}
    tables = build(ctx.paths.raw, ctx.paths.processed)
    added = {n: len(tables[n]) - before[n] for n in KEYS}
    clean, rep = run_quality(retry=False)        # daily_clean.parquet (gap imputation, flags)
    last = pd.Timestamp(clean["date"].max())
    ctx.state["rows_added"] = added
    ctx.state["actuals_through"] = last
    msg = "Rows added: " + ", ".join(f"{n} {v:+,}" for n, v in added.items()) + f". Actuals through {fmt.date(last)}."
    return (DONE if any(added.values()) else NO_DATA), msg, {"rows_added": added, "actuals_through": str(last.date())}


def step_features(ctx: Context):
    from src.features.build import build_daily_features

    daily, fc = build_daily_features(save=True)
    last = daily.dropna(subset=list(TARGETS), how="all").index.max()
    return DONE, f"{len(daily):,} days of features through {fmt.date(last)}; {len(fc):,} weather-forecast rows.", \
        {"days": len(daily), "through": str(pd.Timestamp(last).date())}


def archive_forecast(processed: Path) -> int:
    """Append latest_forecast.parquet to forecast_history.parquet (deduplicated). Returns rows."""
    src = processed / "latest_forecast.parquet"
    if not src.exists():
        return 0
    new = pd.read_parquet(src)
    hist_p = processed / HISTORY_FILE
    hist = pd.concat([pd.read_parquet(hist_p), new], ignore_index=True) if hist_p.exists() else new
    hist = hist.drop_duplicates(["origin", "model", "target", "date"], keep="last")
    tmp = hist_p.with_suffix(".tmp")
    hist.to_parquet(tmp, index=False)
    tmp.replace(hist_p)
    return len(hist)


def step_forecast(ctx: Context):
    from src.predict import run_predict
    from src.viz import MODEL_LABELS

    archive_forecast(ctx.paths.processed)          # keep the previous forecast so it can be scored
    meta = run_predict(use_live_weather=ctx.live_weather)
    archive_forecast(ctx.paths.processed)
    origin = pd.Timestamp(meta["origin"])
    first, last = origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=7)
    ctx.state["forecast_range"] = (first, last)
    msg = (f"Forecast for {fmt.date_range(first, last)} (origin {fmt.date(origin)}, model "
           f"{MODEL_LABELS.get(meta['best_model'], meta['best_model'])}, weather: {meta['weather_source']}).")
    if ctx.live_weather and "live" not in meta["weather_source"]:
        msg += " The live Open-Meteo forecast was unavailable, so archived forecasts / persistence were used."
    return DONE, msg, {"origin": meta["origin"], "best_model": meta["best_model"],
                       "weather_source": meta["weather_source"]}


def update_live_scorecard(processed: Path) -> pd.DataFrame:
    """Score archived forecasts whose target days now have actuals. Returns the newly scored rows
    (one per origin x date x target: our best model's P50 vs Grid-India vs actual)."""
    hist_p, feat_p, out_p = processed / HISTORY_FILE, processed / "features.parquet", processed / SCORECARD_FILE
    cols = ["origin", "date", "target", "h", "model", "recipe", "p10", "p50_ours", "p90", "p50_gi", "actual",
            "ae_ours", "ae_gi", "scored_at"]
    if not hist_p.exists() or not feat_p.exists():
        return pd.DataFrame(columns=cols)
    hist = pd.read_parquet(hist_p)
    feat = pd.read_parquet(feat_p).set_index("date")
    tgt = list(TARGETS)
    act = feat[tgt].copy()
    for t in tgt:                                 # never score against an imputed value
        if f"{t}_imputed" in feat:
            act.loc[feat[f"{t}_imputed"].fillna(False).astype(bool), t] = float("nan")
    act = act.rename_axis("date").reset_index().melt("date", var_name="target", value_name="actual").dropna()

    if "recipe" not in hist:
        hist["recipe"] = "baseline"
    hist["recipe"] = hist["recipe"].fillna("baseline")      # forecasts issued before recipes were tagged
    ours = hist[hist["is_best"].fillna(False).astype(bool)][["origin", "date", "target", "h", "model", "recipe",
                                                              "p10", "p50", "p90"]]
    gi = hist[hist["model"] == "grid_india"][["origin", "date", "target", "p50"]].rename(columns={"p50": "p50_gi"})
    j = ours.rename(columns={"p50": "p50_ours"}).merge(gi, on=["origin", "date", "target"], how="left")
    j = j.merge(act, on=["date", "target"], how="inner")
    j["ae_ours"] = (j["p50_ours"] - j["actual"]).abs()
    j["ae_gi"] = (j["p50_gi"] - j["actual"]).abs()
    j["scored_at"] = pd.Timestamp.now().floor("s")

    old = pd.read_parquet(out_p) if out_p.exists() else pd.DataFrame(columns=cols)
    if len(old) and "recipe" not in old:
        old["recipe"] = "baseline"
    key = ["origin", "date", "target"]
    if len(old):
        seen = pd.MultiIndex.from_frame(old[key])
        j = j[~pd.MultiIndex.from_frame(j[key]).isin(seen)]
    if len(j):
        allr = pd.concat([old, j[cols]], ignore_index=True) if len(old) else j[cols]
        tmp = out_p.with_suffix(".tmp")
        allr.sort_values(key).to_parquet(tmp, index=False)
        tmp.replace(out_p)
    return j[cols].reset_index(drop=True)


def step_scorecard(ctx: Context):
    new = update_live_scorecard(ctx.paths.processed)
    ctx.state["scored"] = new
    try:                                           # stage 3 of the evaluation: all models, all scored days
        from src.fixed_split import live_summary

        live_summary(ctx.paths.processed)
    except Exception as e:  # the summary is a report; never fail the run for it
        log.warning("live summary not updated: %s", e)
    if new.empty:
        return NO_DATA, "No forecast days gained actuals since the last run.", {"new_rows": 0}
    days = sorted(new["date"].dt.date.unique())
    parts = [f"{len(days)} newly scored day{'s' if len(days) != 1 else ''} ({_dates_text(days)})."]
    for t in ("evening_peak", "solar_peak"):
        g = new[new["target"] == t]
        if len(g):
            gi = g["ae_gi"].dropna()
            gi_txt = f", Grid-India {fmt.value(gi.mean(), t)}" if len(gi) else ", no Grid-India forecast"
            parts.append(f"{'Evening' if t == 'evening_peak' else 'Solar-hour'} peak mean absolute error: "
                         f"our model {fmt.value(g['ae_ours'].mean(), t)}{gi_txt}.")
    return DONE, " ".join(parts), {"new_rows": len(new), "days": [str(d) for d in days]}


# ============================================================================ run
def _summary(run: RunResult, ctx: Context) -> str:
    parts = []
    new = ctx.state.get("new_psp_dates") or []
    if new:
        parts.append(f"Added {len(new)} day{'s' if len(new) != 1 else ''} of data ({_dates_text(new)}).")
    elif run.step("psp").status == NO_DATA:
        parts.append("No new Grid-India data.")
    if run.step("forecast").status == DONE and "forecast_range" in ctx.state:
        parts.append(f"New forecast for {fmt.date_range(*ctx.state['forecast_range'])}.")
    scored = ctx.state.get("scored")
    if scored is not None and len(scored):
        parts.append(f"Scored {scored['date'].nunique()} day(s) against actuals.")
    bad = [s for s in run.steps if s.status == FAILED]
    if bad:
        parts.append("Failed: " + "; ".join(f"step {run.steps.index(s) + 1} ({s.short.lower()}): {s.error}"
                                              for s in bad) + ".")
    return " ".join(parts) or "Nothing to update."


def _append_run(paths: Paths, run: RunResult) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    rec = run.to_dict()
    for s in rec["steps"]:          # keep the log compact; full tracebacks stay in the run object
        if s.get("traceback"):
            s["traceback"] = s["traceback"][-4000:]
    with open(paths.runs_log, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def run_daily_pipeline(progress_callback: Callable[[RunResult, StepResult], None] | None = None,
                       trigger: str = "manual", paths: Paths | None = None, today: dt.date | None = None,
                       live_weather: bool = True, client_factory: Callable | None = None) -> RunResult:
    """Run all steps in order. `progress_callback(run, step)` is called whenever a step changes
    state. Raises PipelineLocked if another run holds the lock."""
    paths = paths or Paths()
    ctx = Context(paths=paths, today=today or dt.date.today(), live_weather=live_weather,
                  client_factory=client_factory)
    run = RunResult(run_id=uuid.uuid4().hex[:12], trigger=trigger,
                    started=pd.Timestamp.now().isoformat(timespec="seconds"),
                    steps=[StepResult(k, label, short) for k, label, short in STEPS])
    user_notify = progress_callback or (lambda r, s: None)

    def notify(r, s):
        write_progress(paths, r)          # first: the saved state must not depend on the callback
        user_notify(r, s)

    with pipeline_lock(paths, trigger, run.run_id):
        t_run = time.perf_counter()
        before = _snapshot(paths)
        write_progress(paths, run)
        try:
            _run_steps(run, ctx, notify)
        except BaseException as e:   # stopped from outside (app session closed, Ctrl+C): still log it
            run.status = "interrupted"
            run.finished = pd.Timestamp.now().isoformat(timespec="seconds")
            run.duration_s = round(time.perf_counter() - t_run, 2)
            for s in run.steps:
                if s.status == RUNNING:
                    s.status, s.error = FAILED, f"Interrupted ({type(e).__name__})"
                    s.message = s.error
                elif s.status == PENDING:
                    s.status, s.message = SKIPPED, "Skipped because the run was interrupted."
            run.summary = "Interrupted. " + _summary(run, ctx)
            run.changes = what_changed(before, _snapshot(paths))
            write_progress(paths, run)
            try:
                _append_run(paths, run)
            except Exception:
                pass
            raise
        run.duration_s = round(time.perf_counter() - t_run, 2)
        run.finished = pd.Timestamp.now().isoformat(timespec="seconds")
        if any(st.status == FAILED and st.key not in SOFT_STEPS for st in run.steps):
            run.status = "failed"
        elif any(st.status == FAILED for st in run.steps):
            run.status = "partial"
        else:
            run.status = "success"
        run.summary = _summary(run, ctx)
        run.changes = what_changed(before, _snapshot(paths))
        write_progress(paths, run)
        _append_run(paths, run)
    return run


# ============================================================================ background run
# The app starts the pipeline in a thread of the server process, not in the browser session's
# script run: a page refresh or a closed tab ends the session's script, and must not end the run.
# The lock file and the progress file (both under logs/) are how any session, or the command
# line, finds out about it.
_bg: threading.Thread | None = None
_bg_guard = threading.Lock()


def start_background(trigger: str = "button", wait_s: float = 5.0, **kw) -> threading.Thread:
    """Run the pipeline in a daemon thread and return it once the run holds the lock (or has
    already finished). Raises PipelineLocked when another run is in progress."""
    global _bg
    paths = kw.get("paths") or Paths()
    with _bg_guard:
        held = read_lock(paths)
        if held is not None:
            raise PipelineLocked(held)

        def target():
            try:
                run = run_daily_pipeline(trigger=trigger, **kw)
                write_progress(paths, run)          # final state, whichever way the run was made
            except PipelineLocked:
                pass                                # lost a race with another starter: theirs is running
            except BaseException:
                log.exception("background pipeline run failed")

        _bg = threading.Thread(target=target, name="pipeline-run", daemon=True)
        _bg.start()
        deadline = time.monotonic() + wait_s
        while _bg.is_alive() and read_lock(paths) is None and time.monotonic() < deadline:
            time.sleep(0.02)
        return _bg


def background_thread() -> threading.Thread | None:
    return _bg


def _run_steps(run: RunResult, ctx: Context, notify) -> None:
    """Run the steps in order; a failing non-download step marks the rest skipped."""
    stop = False
    for s in run.steps:
        if stop:
            s.status = SKIPPED
            s.message = "Skipped because an earlier step failed."
            notify(run, s)
            continue
        s.status, s.started = RUNNING, pd.Timestamp.now().isoformat(timespec="seconds")
        notify(run, s)
        t0 = time.perf_counter()
        try:
            fn = globals()[f"step_{s.key}"]          # looked up at run time (tests patch steps)
            s.status, s.message, s.details = fn(ctx)
        except Exception as e:
            s.status, s.error = FAILED, f"{type(e).__name__}: {e}"
            s.message = s.error
            s.traceback = traceback.format_exc()
            log.exception("step %s failed", s.key)
            stop = s.key not in SOFT_STEPS
        s.duration_s = round(time.perf_counter() - t0, 2)
        log.info("%-12s %-12s %6.1fs  %s", s.key, s.status, s.duration_s, s.message)
        notify(run, s)


# ============================================================================ reporting
def read_runs(paths: Paths | None = None, n: int = 10) -> list[dict]:
    """The last `n` runs, newest first."""
    p = (paths or Paths()).runs_log
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[::-1][:n]


def last_successful_run(paths: Paths | None = None) -> dict | None:
    return next((r for r in read_runs(paths, n=1000) if r.get("status") in ("success", "partial")), None)


def data_freshness(paths: Paths | None = None) -> pd.DataFrame:
    """Latest data date and file update time per source."""
    from src.config import WEATHER_DIR

    paths = paths or Paths()
    P = paths.processed

    def mtime(*files):
        ts = [f.stat().st_mtime for f in files if f.exists()]
        return pd.Timestamp.fromtimestamp(max(ts)).floor("s") if ts else None

    def maxdate(path, col):
        try:
            return pd.read_parquet(path, columns=[col])[col].max() if path.exists() else None
        except Exception:
            return None

    rows = []
    psp = list((paths.raw / "psp").glob("*NLDC_PSP*"))
    rows.append(("Grid-India daily PSP reports", maxdate(P / "daily_summary.parquet", "date"),
                 mtime(*psp) if psp else None, "Data date of the latest parsed report"))
    pdfs = list((paths.raw / "forecast").glob("*.pdf"))
    rows.append(("Grid-India week-ahead forecast", maxdate(P / "forecast_daily.parquet", "date"),
                 mtime(*pdfs) if pdfs else None, "Last day covered by a parsed forecast"))
    obs = list(WEATHER_DIR.glob("obs_*.parquet"))
    fc = list(WEATHER_DIR.glob("fc_*.parquet"))
    obs_last = max((d for d in (maxdate(p, "date") for p in obs) if d is not None), default=None)
    fc_last = max((d for d in (maxdate(p, "date") for p in fc) if d is not None), default=None)
    rows.append(("Open-Meteo observed weather", obs_last, mtime(*obs) if obs else None, "Latest observed day"))
    rows.append(("Open-Meteo archived forecasts", fc_last, mtime(*fc) if fc else None, "Latest forecast target day"))
    rows.append(("Features", maxdate(P / "features.parquet", "date"), mtime(P / "features.parquet"),
                 "Latest day in the feature table"))
    meta_p = P / "latest_meta.json"
    origin = None
    if meta_p.exists():
        try:
            origin = pd.Timestamp(json.loads(meta_p.read_text())["origin"])
        except Exception:
            origin = None
    rows.append(("Our 7-day forecast", origin + pd.Timedelta(days=7) if origin is not None else None,
                 mtime(P / "latest_forecast.parquet"), "Last forecast day"))
    rows.append(("Live scorecard", maxdate(P / SCORECARD_FILE, "date"), mtime(P / SCORECARD_FILE),
                 "Latest scored day"))
    return pd.DataFrame(rows, columns=["source", "latest_data", "file_updated", "meaning"])


# ============================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trigger", default="manual", choices=["manual", "timer", "button"])
    ap.add_argument("--no-live-weather", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    def show(run, step):
        if step.status not in (PENDING, RUNNING):
            i = run.steps.index(step) + 1
            print(f"[{i}/{len(run.steps)}] {step.label:<52} {step.status:<12} "
                  f"{(step.duration_s or 0):6.1f} s  {step.message}", flush=True)

    try:
        run = run_daily_pipeline(show, trigger=args.trigger, live_weather=not args.no_live_weather)
    except PipelineLocked as e:
        print(str(e), flush=True)
        return 0                       # not an error for the timer: the other run does the work
    from src.data import weather

    print(f"\n{run.status.upper()} in {run.duration_s:.0f} s. {run.summary}", flush=True)
    print(f"Open-Meteo requests this run: {weather.REQUESTS}", flush=True)
    return 1 if run.status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
