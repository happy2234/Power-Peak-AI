"""The daily pipeline (src/pipeline.py) with mocked downloads and stubbed heavy steps."""
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline as pl

TODAY = dt.date(2026, 11, 9)
YESTERDAY = TODAY - dt.timedelta(days=1)


class FakeClient:
    """Grid-India client double: `published` dates have a PSP report; forecasts are fixed names."""

    def __init__(self, published=(), forecasts=(), fail_psp=False, fail_forecasts=False):
        self.published, self.forecasts = set(published), list(forecasts)
        self.fail_psp, self.fail_forecasts = fail_psp, fail_forecasts
        self.psp_calls = []

    def download_psp(self, d, dest):
        self.psp_calls.append(d)
        if self.fail_psp:
            raise ConnectionError("grid-india.in timed out")
        existing = list(Path(dest).glob(f"{d:%d.%m.%y}_NLDC_PSP*"))
        if existing:
            return existing[0]
        if d not in self.published:
            return None
        p = Path(dest) / f"{d:%d.%m.%y}_NLDC_PSP_123.xls"
        p.write_bytes(b"\xd0\xcf\x11\xe0 fake")
        return p

    def download_forecasts(self, start, end, dest):
        if self.fail_forecasts:
            raise ConnectionError("listing unavailable")
        out = []
        for name in self.forecasts:
            p = Path(dest) / name
            p.write_bytes(b"%PDF fake")
            out.append(p)
        return out


@pytest.fixture
def paths(tmp_path):
    p = pl.Paths(raw=tmp_path / "raw", processed=tmp_path / "processed", logs=tmp_path / "logs")
    (p.raw / "psp").mkdir(parents=True)
    p.processed.mkdir(parents=True)
    # processed data through 01 Nov 2026 -> the pipeline should look for 02..08 Nov
    dates = pd.date_range("2026-10-01", "2026-11-01")
    pd.DataFrame({"date": dates}).to_parquet(p.processed / "daily_summary.parquet")
    for d in dates[-3:]:   # some recent raw files already on disk
        (p.raw / "psp" / f"{d:%d.%m.%y}_NLDC_PSP_1.xls").write_bytes(b"x")
    return p


@pytest.fixture
def stub_heavy(monkeypatch):
    """Replace steps 3-7 (network weather, parsing, features, models) with fast stand-ins that
    record their calls."""
    calls = []

    def make(key, status=pl.DONE, msg="ok"):
        def fn(ctx):
            calls.append(key)
            if key == "forecast":
                ctx.state["forecast_range"] = (pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-15"))
            return status, msg, {}
        return fn

    for key in ("weather", "datasets", "features", "forecast", "scorecard"):
        monkeypatch.setattr(pl, f"step_{key}", make(key))
    return calls


def _run(paths, client, **kw):
    events = []
    run = pl.run_daily_pipeline(lambda r, s: events.append((s.key, s.status)), trigger="button", paths=paths,
                                today=TODAY, client_factory=lambda: client, **kw)
    return run, events


# ============================================================================ new data
def test_new_data_downloads_missing_days_and_reports_them(paths, stub_heavy):
    new_days = [dt.date(2026, 11, d) for d in range(2, 9)]
    client = FakeClient(published=new_days,
                        forecasts=["Week ahead all India demand forecast 09-Nov-2026 to 15-Nov-2026_1.pdf"])
    run, events = _run(paths, client)

    assert run.status == "success"
    psp = run.step("psp")
    assert psp.status == pl.DONE
    assert psp.details["new_dates"] == [str(d) for d in new_days]
    assert "7 new reports: 02–08 Nov 2026." in psp.message
    assert "not published" not in psp.message
    assert set(client.psp_calls) >= set(new_days)
    assert run.step("gi_forecast").status == pl.DONE
    assert "09–15 Nov 2026" in run.step("gi_forecast").message
    assert stub_heavy == ["weather", "datasets", "features", "forecast", "scorecard"]
    assert run.summary.startswith("Added 7 days of data (02–08 Nov 2026). New forecast for 09–15 Nov 2026.")
    assert all(s.duration_s is not None for s in run.steps)
    # live progress: every step went running -> final state, in order
    assert events[0] == ("psp", pl.RUNNING) and events[1] == ("psp", pl.DONE)
    assert [k for k, s in events if s == pl.RUNNING] == [k for k, _, _ in pl.STEPS]


def test_files_already_on_disk_are_not_counted_as_new(paths, stub_heavy):
    for d in range(2, 9):
        (paths.raw / "psp" / f"{d:02d}.11.26_NLDC_PSP_9.xls").write_bytes(b"x")
    run, _ = _run(paths, FakeClient(published=[]))
    assert run.step("psp").status == pl.NO_DATA
    assert run.step("psp").details["new_dates"] == []


# ============================================================================ no new data
def test_no_new_data_is_not_an_error_and_the_forecast_still_runs(paths, stub_heavy):
    run, _ = _run(paths, FakeClient(published=[]))

    psp = run.step("psp")
    assert psp.status == pl.NO_DATA
    assert "The report for 08 Nov 2026 is not published yet." in psp.message
    assert run.status == "success"
    assert run.step("weather").status == pl.DONE
    assert run.step("forecast").status == pl.DONE
    assert run.step("gi_forecast").status == pl.NO_DATA
    assert run.summary.startswith("No new Grid-India data. New forecast for 09–15 Nov 2026.")


def test_yesterday_missing_but_earlier_days_new(paths, stub_heavy):
    run, _ = _run(paths, FakeClient(published=[dt.date(2026, 11, d) for d in range(2, 8)]))
    psp = run.step("psp")
    assert psp.status == pl.DONE
    assert "6 new reports: 02–07 Nov 2026." in psp.message
    assert "08 Nov 2026 is not published yet" in psp.message


# ============================================================================ failures
def test_failing_step_stops_the_run_and_keeps_earlier_results(paths, stub_heavy, monkeypatch):
    def broken(ctx):
        raise ValueError("feature table has no rows")

    monkeypatch.setattr(pl, "step_features", broken)
    new_days = [dt.date(2026, 11, d) for d in range(2, 9)]
    run, _ = _run(paths, FakeClient(published=new_days))

    assert run.status == "failed"
    f = run.step("features")
    assert f.status == pl.FAILED
    assert f.error == "ValueError: feature table has no rows"
    assert "Traceback" in f.traceback and "broken" in f.traceback
    assert [run.step(k).status for k in ("psp", "gi_forecast", "weather", "datasets")] == \
        [pl.DONE, pl.NO_DATA, pl.DONE, pl.DONE]
    assert [run.step(k).status for k in ("forecast", "scorecard")] == [pl.SKIPPED, pl.SKIPPED]
    assert "forecast" not in stub_heavy and "scorecard" not in stub_heavy
    # earlier step results stay saved on disk
    assert len(list((paths.raw / "psp").glob("0[2-8].11.26_*"))) == 7
    assert "Failed: step 5 (features): ValueError: feature table has no rows." in run.summary
    # the run is logged with the failed step, and the lock is released
    rec = json.loads(paths.runs_log.read_text().splitlines()[-1])
    assert rec["status"] == "failed" and rec["trigger"] == "button"
    assert {s["key"]: s["status"] for s in rec["steps"]}["features"] == pl.FAILED
    assert not paths.lock.exists()


def test_download_failure_is_reported_but_the_run_continues(paths, stub_heavy):
    run, _ = _run(paths, FakeClient(fail_psp=True, fail_forecasts=True))
    assert run.step("psp").status == pl.FAILED
    assert "Grid-India could not be reached" in run.step("psp").error
    assert run.step("gi_forecast").status == pl.FAILED
    assert run.step("forecast").status == pl.DONE
    assert run.status == "partial"


# ============================================================================ lock
def test_lock_blocks_a_second_run(paths, stub_heavy):
    with pl.pipeline_lock(paths, "timer") as info:
        with pytest.raises(pl.PipelineLocked) as e:
            _run(paths, FakeClient())
        assert str(e.value) == f"An update is already running (started {pd.Timestamp(info['started']):%H:%M})"
        assert e.value.info["trigger"] == "timer"
        assert paths.lock.exists()            # the blocked run did not remove the holder's lock
    assert not paths.lock.exists()
    assert not paths.runs_log.exists()         # nothing ran
    run, _ = _run(paths, FakeClient())        # free again
    assert run.status == "success"


def test_lock_released_when_a_step_raises_unexpectedly(paths, monkeypatch):
    monkeypatch.setattr(pl, "_append_run", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    for key in ("psp", "gi_forecast", "weather", "datasets", "features", "forecast", "scorecard"):
        monkeypatch.setattr(pl, f"step_{key}", lambda ctx: (pl.DONE, "ok", {}))
    with pytest.raises(OSError):
        _run(paths, FakeClient())
    assert not paths.lock.exists()


def test_interrupted_run_is_logged_and_lock_released(paths, stub_heavy):
    def progress(run, step):
        if step.key == "weather" and step.status == pl.RUNNING:
            raise KeyboardInterrupt          # e.g. the app session closed mid-run

    with pytest.raises(KeyboardInterrupt):
        pl.run_daily_pipeline(progress, trigger="button", paths=paths, today=TODAY,
                              client_factory=lambda: FakeClient())
    assert not paths.lock.exists()
    rec = pl.read_runs(paths)[0]
    assert rec["status"] == "interrupted" and rec["summary"].startswith("Interrupted.")
    st = {s["key"]: s["status"] for s in rec["steps"]}
    assert st["psp"] == pl.NO_DATA and st["weather"] == pl.FAILED and st["forecast"] == pl.SKIPPED


def test_stale_lock_from_dead_process_is_replaced(paths, stub_heavy):
    import socket

    paths.logs.mkdir(parents=True, exist_ok=True)
    dead_pid = 2 ** 22 + 12345
    paths.lock.write_text(json.dumps({"pid": dead_pid, "host": socket.gethostname(), "trigger": "timer",
                                      "started": pd.Timestamp.now().isoformat()}))
    run, _ = _run(paths, FakeClient())
    assert run.status == "success"


def test_old_lock_is_stale(paths):
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.lock.write_text(json.dumps({"pid": os.getpid(), "host": "other-host", "trigger": "timer",
                                      "started": (pd.Timestamp.now() - pd.Timedelta(hours=5)).isoformat()}))
    assert pl.read_lock(paths) is None and not paths.lock.exists()


def test_live_lock_is_kept_and_reported(paths):
    paths.logs.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp.now().floor("s") - pd.Timedelta(minutes=5)
    paths.lock.write_text(json.dumps({"pid": os.getpid(), "host": "x", "trigger": "button",
                                      "started": started.isoformat()}))
    info = pl.read_lock(paths)
    assert info is not None and paths.lock.exists()
    assert str(pl.PipelineLocked(info)) == f"An update is already running (started {started:%H:%M})"


# ============================================================================ history + scoring
def test_runs_are_logged_newest_first(paths, stub_heavy):
    _run(paths, FakeClient())
    _run(paths, FakeClient(published=[YESTERDAY]))
    runs = pl.read_runs(paths)
    assert len(runs) == 2
    assert runs[0]["steps"][0]["details"]["new_dates"] == [str(YESTERDAY)]
    assert {"started", "finished", "trigger", "status", "steps", "summary"} <= set(runs[0])
    assert pl.last_successful_run(paths)["run_id"] == runs[0]["run_id"]


def test_live_scorecard_scores_each_forecast_day_once(tmp_path):
    origin = pd.Timestamp("2026-11-01")
    days = pd.date_range("2026-11-02", periods=7)
    rows = []
    for t, lvl in [("solar_peak", 230_000), ("evening_peak", 210_000), ("energy", 4800)]:
        for i, d in enumerate(days, 1):
            rows.append(dict(model="lstm", origin=origin, date=d, h=i, target=t, p10=lvl * .97, p50=lvl,
                             p90=lvl * 1.03, actual=np.nan, is_best=True))
            rows.append(dict(model="lgbm", origin=origin, date=d, h=i, target=t, p10=lvl * .9, p50=lvl * 1.1,
                             p90=lvl * 1.2, actual=np.nan, is_best=False))
            if t != "energy":
                rows.append(dict(model="grid_india", origin=origin, date=d, h=i, target=t, p10=np.nan,
                                 p50=lvl * 1.02, p90=np.nan, actual=np.nan, is_best=False))
    pd.DataFrame(rows).to_parquet(tmp_path / "latest_forecast.parquet")
    assert pl.archive_forecast(tmp_path) == len(rows)
    # actuals for the first 5 days only; day 3 evening peak imputed
    idx = pd.date_range("2026-10-01", days[4])
    feat = pd.DataFrame({"date": idx, "solar_peak": 231_000.0, "evening_peak": 212_000.0, "energy": 4700.0,
                         "solar_peak_imputed": False, "evening_peak_imputed": idx == days[2],
                         "energy_imputed": False})
    feat.to_parquet(tmp_path / "features.parquet")

    new = pl.update_live_scorecard(tmp_path)
    assert len(new) == 5 + 4 + 5
    assert set(new["model"]) == {"lstm"}
    ev = new[new["target"] == "evening_peak"]
    assert np.allclose(ev["ae_ours"], 2000) and np.allclose(ev["ae_gi"], 2200)
    assert new[new["target"] == "energy"]["p50_gi"].isna().all()
    assert pl.update_live_scorecard(tmp_path).empty          # nothing new the second time
    assert len(pd.read_parquet(tmp_path / pl.SCORECARD_FILE)) == 14


def test_freshness_lists_every_source(paths):
    fr = pl.data_freshness(paths)
    assert {"Grid-India daily PSP reports", "Grid-India week-ahead forecast", "Open-Meteo observed weather",
            "Our 7-day forecast", "Live scorecard"} <= set(fr["source"])
    assert fr.set_index("source").loc["Grid-India daily PSP reports", "latest_data"] == pd.Timestamp("2026-11-01")


def test_cli_reports_lock_without_failing(paths, monkeypatch, capsys):
    monkeypatch.setattr(pl, "Paths", lambda: paths)
    with pl.pipeline_lock(paths, "button"):
        assert pl.main(["--trigger", "timer"]) == 0
    assert "An update is already running" in capsys.readouterr().out


# ============================================================================ the button (AppTest)
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

HOME = Path(__file__).resolve().parents[1] / "app" / "Home.py"


def _home(monkeypatch, enabled=True):
    import streamlit as st

    monkeypatch.setenv("ENABLE_FETCH_BUTTON", "true" if enabled else "false")
    st.cache_data.clear()
    at = AppTest.from_file(str(HOME), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _labels(at):
    return [b.label for b in at.button]


def test_button_hidden_when_disabled(monkeypatch):
    at = _home(monkeypatch, enabled=False)
    assert ui_label() not in _labels(at)
    assert any("Data updated:" in m.value for m in at.sidebar.markdown)


def ui_label():
    import ui

    return ui.FETCH_LABEL


class Gate:
    """A fake pipeline run that takes the real lock and holds it until the test lets it go, so the
    tests can look at the page while an update is 'in progress'. The run is what the app starts in
    its background thread; it never touches the browser session."""

    def __init__(self, paths, outcome="success"):
        import threading

        self.paths, self.outcome, self.release = paths, outcome, threading.Event()

    def __call__(self, progress_callback=None, trigger="manual", **kw):
        run = pl.RunResult("abc", trigger, pd.Timestamp.now().isoformat(timespec="seconds"),
                           [pl.StepResult(k, label, short) for k, label, short in pl.STEPS])
        notify = progress_callback or (lambda r, s: None)
        with pl.pipeline_lock(self.paths, trigger, run.run_id):
            for s in run.steps[:2]:
                s.status, s.duration_s, s.message = pl.DONE, 1.5, "ok"
            run.steps[1].status, run.steps[1].message = pl.NO_DATA, "No new week-ahead forecast."
            w = run.steps[2]
            w.status, w.started = pl.RUNNING, (pd.Timestamp.now() - pd.Timedelta(seconds=75)).isoformat()
            pl.write_progress(self.paths, run)
            notify(run, w)
            self.release.wait(10)
            for s in run.steps[2:]:
                s.status, s.duration_s, s.message = pl.DONE, 0.1, "ok"
            if self.outcome == "failed":
                f = run.steps[4]
                f.status, f.error, f.traceback = pl.FAILED, "ValueError: boom", "Traceback (most recent call last): ..."
                for s in run.steps[5:]:
                    s.status = pl.SKIPPED
                run.status = "failed"
            else:
                run.status = "success"
            run.finished, run.duration_s = pd.Timestamp.now().isoformat(timespec="seconds"), 60
            run.summary = "Added 7 days of data (02–08 Nov 2026). New forecast for 09–15 Nov 2026."
            run.changes = {"data_before": "2026-11-01", "data_after": "2026-11-08", "forecast_before": "2026-11-01",
                           "forecast_after": "2026-11-08", "forecast_new_dates": ["2026-11-09", "2026-11-10"],
                           "scorecard_days_added": 3, "scorecard_new_dates": ["2026-11-05", "2026-11-06", "2026-11-07"]}
            pl.write_progress(self.paths, run)
            pl._append_run(self.paths, run)
        return run


def _gated(monkeypatch, tmp_path, outcome="success"):
    paths = pl.Paths(logs=tmp_path)
    gate = Gate(paths, outcome)
    monkeypatch.setattr(pl, "run_daily_pipeline", gate)
    monkeypatch.setattr(pl, "Paths", lambda: paths)
    return gate, paths


def _finish(gate):
    gate.release.set()
    th = pl.background_thread()
    if th is not None:
        th.join(10)


def _md(at):
    return " ".join(m.value for m in at.markdown)


def test_button_starts_live_update_then_shows_summary_and_what_changed(monkeypatch, tmp_path):
    gate, paths = _gated(monkeypatch, tmp_path)
    at = _home(monkeypatch)
    assert _labels(at).count(ui_label()) == 2            # sidebar + Overview header
    at.button(key="fetch_sidebar").click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = _md(at)                                         # the run is in progress: seven steps, status, elapsed
    assert "Fetching weather for 11 cities, about 4 minutes in total." in md
    assert "No new data" in md and "Running" in md and "1 min 15 s" in md
    assert md.count("<div class='pp-live-row ") == 7
    _finish(gate)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    md = _md(at)
    assert "Update complete" in md and "What changed" in md
    assert "Added 7 days of data (02–08 Nov 2026). New forecast for 09–15 Nov 2026." in md
    assert "Sun 01 Nov 2026 → **Sun 08 Nov 2026**" in md or "→ **" in md
    assert "**3** (05–07 Nov 2026)" in md and "09–10 Nov 2026" in md
    at.run()                                             # shown once
    assert "What changed" not in _md(at)


def test_refresh_during_a_run_reattaches_and_the_lock_holds(monkeypatch, tmp_path):
    gate, paths = _gated(monkeypatch, tmp_path)
    at = _home(monkeypatch)
    at.button(key="fetch_sidebar").click().run()
    assert pl.read_lock(paths) is not None
    fresh = _home(monkeypatch)                           # a refresh = a brand-new session
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert "Updating data" in [h.value for h in fresh.subheader] and "pp-live-row" in _md(fresh)
    assert pl.read_lock(paths) is not None               # the run was not interrupted by the old session ending
    fresh.button(key="fetch_sidebar").click().run()      # pressing again joins the run, never starts a second one
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert pl.read_lock(paths)["run_id"] == "abc"
    _finish(gate)
    fresh.run()
    assert "Update complete" in _md(fresh)
    assert pl.read_runs(paths)[0]["run_id"] == "abc"     # and the run was logged exactly once
    assert len(pl.read_runs(paths)) == 1


def test_run_survives_its_session_and_logs(tmp_path):
    """A real run (stubbed steps) is not tied to any browser session: started in the background it
    finishes, appends to the log, and leaves no lock."""
    paths = pl.Paths(raw=tmp_path / "raw", processed=tmp_path / "processed", logs=tmp_path / "logs")
    (paths.raw / "psp").mkdir(parents=True)
    paths.processed.mkdir(parents=True)
    import threading

    go = threading.Event()
    orig = pl.step_psp

    def slow(ctx):
        go.wait(10)
        return pl.NO_DATA, "No new reports.", {}

    pl.step_psp = slow
    try:
        keep = {k: getattr(pl, f"step_{k}") for k in ("gi_forecast", "weather", "datasets", "features", "forecast", "scorecard")}
        for k in keep:
            setattr(pl, f"step_{k}", lambda ctx: (pl.NO_DATA, "No new data", {}))
        th = pl.start_background(paths=paths, today=TODAY, client_factory=lambda: None)
        assert pl.read_lock(paths) is not None
        prog = pl.read_progress(paths)
        assert prog["status"] == "running" and prog["steps"][0]["status"] == pl.RUNNING
        with pytest.raises(pl.PipelineLocked):
            pl.start_background(paths=paths, today=TODAY, client_factory=lambda: None)
        go.set()
        th.join(10)
    finally:
        pl.step_psp = orig
        for k, fn in keep.items():
            setattr(pl, f"step_{k}", fn)
    assert pl.read_lock(paths) is None
    last = pl.read_runs(paths)[0]
    assert last["status"] == "success" and all(s["status"] == pl.NO_DATA for s in last["steps"])
    assert pl.read_progress(paths)["status"] == "success"


def test_what_changed_compares_before_and_after(paths, stub_heavy):
    """latest data date, new forecast dates, days added to the live scorecard."""
    before = pl._snapshot(paths)
    assert before["latest_data"] == "2026-11-01" and before["scorecard_dates"] == []
    pd.DataFrame({"date": pd.date_range("2026-10-01", "2026-11-08")}).to_parquet(paths.processed / "daily_summary.parquet")
    (paths.processed / "latest_meta.json").write_text(json.dumps({"origin": "2026-11-08"}))
    pd.DataFrame({"date": pd.to_datetime(["2026-11-05", "2026-11-05", "2026-11-06"])}).to_parquet(
        paths.processed / pl.SCORECARD_FILE)
    ch = pl.what_changed(before, pl._snapshot(paths))
    assert (ch["data_before"], ch["data_after"]) == ("2026-11-01", "2026-11-08")
    assert ch["forecast_new_dates"][0] == "2026-11-09" and len(ch["forecast_new_dates"]) == 7
    assert ch["scorecard_days_added"] == 2


def test_run_records_changes_and_no_new_data_is_not_an_error(paths, stub_heavy, monkeypatch):
    monkeypatch.setattr(pl, "step_weather", lambda ctx: (pl.NO_DATA, "No new data: all 11 cities were already up to date.", {}))
    run, _ = _run(paths, FakeClient())
    assert run.status == "success"
    assert run.step("weather").status == pl.NO_DATA and run.step("psp").status == pl.NO_DATA
    assert run.changes["data_before"] == run.changes["data_after"] == "2026-11-01"
    assert pl.read_runs(paths)[0]["changes"]["scorecard_days_added"] == 0
    import ui

    md = ui.changes_markdown(run.changes)
    assert "unchanged" in md and "**0**" in md
    assert ui.STATE[pl.NO_DATA] == "No new data"


def test_overview_update_button_runs_pipeline(monkeypatch, tmp_path):
    gate, paths = _gated(monkeypatch, tmp_path)
    at = _home(monkeypatch)
    assert any("Updated " in m.value for m in at.markdown)       # last-updated time beside the button
    at.button(key="fetch_overview").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "pp-live-row" in _md(at)
    _finish(gate)
    at.run()
    assert "Update complete" in _md(at)


def test_overview_says_when_forecast_is_old(monkeypatch):
    import pandas as pd

    import src.app_data as ad

    real = ad.latest_meta
    now = pd.Timestamp.now()
    # The data on disk may be any age, so both cases pin the forecast's issue time.
    monkeypatch.setattr(ad, "latest_meta", lambda: {**(real() or {}), "issued_at": (now - pd.Timedelta(days=3)).isoformat()})
    at = _home(monkeypatch)
    assert any(m.value == "Forecast is 3 days old. Update data to refresh." for m in at.markdown)
    assert "fetch_stale" in [b.key for b in at.button]
    monkeypatch.setattr(ad, "latest_meta", lambda: {**(real() or {}), "issued_at": now.isoformat()})
    at = _home(monkeypatch)
    assert not any("days old" in m.value for m in at.markdown)


def test_button_shows_failed_step_and_traceback(monkeypatch, tmp_path):
    gate, paths = _gated(monkeypatch, tmp_path, outcome="failed")
    at = _home(monkeypatch)
    at.button(key="fetch_sidebar").click().run()
    _finish(gate)
    at.run()
    assert any("Step 5, Rebuild features, failed: ValueError: boom" in e.value for e in at.error)
    assert any("Traceback" in c.value for c in at.code)


def test_button_reports_running_update(monkeypatch, tmp_path):
    """An update started elsewhere (the timer): the button joins it and shows when it started."""
    paths = pl.Paths(logs=tmp_path)
    monkeypatch.setattr(pl, "Paths", lambda: paths)
    at = _home(monkeypatch)
    with pl.pipeline_lock(paths, "timer") as info:
        at.button(key="fetch_sidebar").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert "Updating data" in [h.value for h in at.subheader]
        assert f"started {pd.Timestamp(info['started']):%H:%M}" in " ".join(c.value for c in at.caption)


def test_a_lock_still_being_written_is_not_mistaken_for_a_stale_one(paths):
    """Found in the clean-environment run: reading the lock between its creation and its content removed it."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.lock.write_text("")                       # created, content not there yet
    assert pl.read_lock(paths) is not None and paths.lock.exists()
    os.utime(paths.lock, (0, 0))                    # the same empty file, long ago: now it is stale
    assert pl.read_lock(paths) is None and not paths.lock.exists()
    with pl.pipeline_lock(paths, "button") as info:  # and a real lock always has its content
        assert json.loads(paths.lock.read_text())["pid"] == info["pid"]
