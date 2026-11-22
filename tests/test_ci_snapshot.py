"""The safety checks of the scheduled snapshot job: it must refuse to commit a bad snapshot."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("ci_snapshot", Path(__file__).resolve().parents[1] / "scripts" / "ci_snapshot.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


@pytest.fixture
def world(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    (tmp_path / "logs").mkdir(parents=True)
    proc.mkdir(parents=True)
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "PROCESSED", proc)
    now = pd.Timestamp.now()
    yesterday = (now - pd.Timedelta(days=1)).normalize()
    pd.DataFrame({"date": pd.date_range(yesterday - pd.Timedelta(days=20), yesterday)}).to_parquet(proc / "daily_summary.parquet")
    (proc / "latest_meta.json").write_text(json.dumps({"issued_at": now.isoformat(), "best_model": "m",
                                                      "origin": str(yesterday.date())}))
    days = pd.date_range(yesterday + pd.Timedelta(days=1), periods=7)
    pd.DataFrame({"model": "m", "date": days, "p50": 200000.0}).to_parquet(proc / "latest_forecast.parquet")

    def log(status="success", steps=()):
        rec = {"status": status, "summary": "s", "steps": [{"short": "PSP", "status": x, "error": "boom", "message": ""}
                                                            for x in steps]}
        (tmp_path / "logs" / "pipeline_runs.jsonl").write_text(json.dumps(rec) + "\n")
    log()
    return proc, log, tmp_path


def test_good_snapshot_passes(world, capsys):
    ci.check()
    assert "ok:" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["partial", "failed", "interrupted"])
def test_a_run_that_did_not_succeed_blocks_the_commit(world, status):
    _, log, _ = world
    log(status, steps=["failed"])
    with pytest.raises(SystemExit):
        ci.check()


def test_incomplete_or_missing_forecast_blocks_the_commit(world):
    proc, _, _ = world
    fc = pd.read_parquet(proc / "latest_forecast.parquet")
    fc.iloc[:6].to_parquet(proc / "latest_forecast.parquet")
    with pytest.raises(SystemExit):
        ci.check()
    fc.assign(p50=float("nan")).to_parquet(proc / "latest_forecast.parquet")
    with pytest.raises(SystemExit):
        ci.check()


def test_stale_actuals_block_the_commit(world):
    proc, _, _ = world
    pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10)}).to_parquet(proc / "daily_summary.parquet")
    with pytest.raises(SystemExit):
        ci.check()


def test_old_meta_means_this_run_wrote_nothing(world):
    proc, _, _ = world
    m = json.loads((proc / "latest_meta.json").read_text())
    m["issued_at"] = (pd.Timestamp.now() - pd.Timedelta(days=2)).isoformat()
    (proc / "latest_meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        ci.check()


def test_finish_keeps_the_last_runs_and_the_quality_report(world):
    proc, _, root = world
    (proc / "pipeline_runs.jsonl").write_text("\n".join(f'{{"n": {i}}}' for i in range(100)) + "\n")
    (root / "reports" / "latest").mkdir(parents=True)
    (root / "reports" / "latest" / "data_quality.md").write_text("# quality")
    ci.finish()
    lines = (proc / "pipeline_runs.jsonl").read_text().splitlines()
    assert len(lines) == ci.KEEP_RUNS and json.loads(lines[-1])["status"] == "success"
    assert (root / "reports" / "data_quality.md").read_text() == "# quality"


def test_commit_paths_are_data_and_forecast_files_only():
    assert all(p.startswith(("data/", "reports/data_quality")) for p in ci.COMMIT_PATHS)


def _psp(root, *names):
    d = root / "data" / "raw" / "psp"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(n.encode())


def test_raw_psp_latest_is_a_commit_path():
    assert "data/raw/psp_latest" in ci.COMMIT_PATHS


def test_finish_keeps_exactly_the_newest_psp_file(world):
    _, _, root = world
    _psp(root, "31.12.25_NLDC_PSP_998.xls", "02.01.26_NLDC_PSP_1.xls", "14.11.26_NLDC_PSP_736.xls",
         "08.07.26_NLDC_PSP_H&E v1_222.xls")          # newest by date, not by name
    old = root / "data" / "raw" / "psp_latest"
    old.mkdir(parents=True)
    (old / "13.11.26_NLDC_PSP_735.xls").write_bytes(b"older")
    ci.finish()
    assert [p.name for p in old.iterdir()] == ["14.11.26_NLDC_PSP_736.xls"]
    assert (old / "14.11.26_NLDC_PSP_736.xls").read_bytes() == b"14.11.26_NLDC_PSP_736.xls"


def test_finish_does_nothing_without_a_psp_file(world):
    _, _, root = world
    keep = root / "data" / "raw" / "psp_latest"
    keep.mkdir(parents=True)
    (keep / "13.11.26_NLDC_PSP_735.xls").write_bytes(b"kept")
    ci.finish()                                         # no data/raw/psp at all
    assert [p.name for p in keep.iterdir()] == ["13.11.26_NLDC_PSP_735.xls"]
    assert ci.keep_latest_psp() is None
