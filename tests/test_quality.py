import numpy as np
import pandas as pd

from src.config import TARGETS
from src.data.quality import build_clean, merge_retries, parse_backfill_log


def _summary(drop):
    idx = pd.date_range("2023-04-01", "2023-05-31")
    df = pd.DataFrame({"date": idx})
    for i, col in enumerate(TARGETS.values()):
        df[col] = 200_000.0 + np.arange(len(idx)) * 10 + i
    for r in ["nr", "wr", "sr", "er", "ner"]:
        df[f"energy_met_mu_{r}"] = 900.0
    return df[~df["date"].isin(pd.to_datetime(drop))]


def test_impute_only_short_gaps_and_flag():
    one = ["2023-04-10"]
    two = ["2023-04-20", "2023-04-21"]
    three = ["2023-05-05", "2023-05-06", "2023-05-07"]
    clean, imputed = build_clean(_summary(one + two + three))
    c = clean.set_index("date")
    for d in one + two:
        assert c.loc[d, "is_imputed"] and c.loc[d, "energy_imputed"] and c.loc[d, "is_missing_report"]
        assert not np.isnan(c.loc[d, "energy"])
    # linear interpolation of a linear series is exact
    assert c.loc["2023-04-10", "energy"] == (c.loc["2023-04-09", "energy"] + c.loc["2023-04-11", "energy"]) / 2
    for d in three:  # 3-day gap: left missing, not imputed
        assert np.isnan(c.loc[d, "energy"]) and not c.loc[d, "is_imputed"]
    assert imputed["energy"] == one + two
    assert len(clean) == 61  # full calendar


def test_parse_backfill_log(tmp_path):
    log = tmp_path / "b.log"
    log.write_text(
        "23:03:59 WARNING 2024-08-05: not listed on Grid-India\n"
        "23:07:40 ERROR   2024-12-30: 404 Client Error: Not Found for url: https://x/30.12.24_NLDC_PSP.xls\n"
        "23:24:23 WARNING forecast 2023-03-25 failed: 404 Client Error\n"
        "23:30:03 WARNING   Week ahead x 16-Sep-2023.pdf -> ForecastParseError: fewer than 3 ticks\n")
    got = {(k, i) for k, i, _ in parse_backfill_log(log)}
    assert got == {("not_listed", "2024-08-05"), ("download_error", "2024-12-30"),
                   ("parse_error", "Week ahead x 16-Sep-2023.pdf")}


def test_retry_outcomes_survive_runs_without_retry():
    saved = {"2024-08-05": "still not listed", "2025-11-10": "still not listed", "2023-05-01": "still not listed"}
    missing = ["2024-08-05", "2025-11-10"]                 # 2023-05-01 has since arrived
    # daily pipeline run: no new retries -> earlier outcomes kept for dates still missing
    assert merge_retries(saved, {}, missing) == {"2024-08-05": "still not listed", "2025-11-10": "still not listed"}
    # a later retry replaces the outcome for that date
    assert merge_retries(saved, {"2025-11-10": "failed again: 404"}, missing)["2025-11-10"] == "failed again: 404"


def test_pipeline_report_goes_to_git_ignored_latest_dir():
    from src.config import REPORTS
    from src.data import quality

    assert quality.LATEST_REPORT == REPORTS / "latest" / "data_quality.md"
    assert quality.FULL_REPORT == REPORTS / "data_quality.md"
    import inspect

    # default of the writer is the daily copy, never the committed report
    assert "LATEST_REPORT" in inspect.getsource(quality.write_report)
    ignored = (REPORTS.parent / ".gitignore").read_text()
    assert "/reports/latest/" in ignored and "run.log" in ignored and "/logs/" in ignored


def test_gpu_label_marketing_name():
    from src.utils.device import gpu_label

    assert gpu_label("NVIDIA GeForce RTX 3050 Ti Laptop GPU", int(3.7 * 2**30)) == "NVIDIA RTX 3050 Ti Laptop (4 GB)"
