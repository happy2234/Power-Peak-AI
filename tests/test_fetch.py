"""Offline tests for the downloader helpers (no network)."""
import datetime as dt

import pytest

from src.fetch.grid_india import GridIndiaClient, financial_year, psp_date_from_name, safe_filename


@pytest.mark.parametrize("d,fy", [
    (dt.date(2026, 11, 1), "2026-27"), (dt.date(2027, 3, 31), "2026-27"),
    (dt.date(2023, 4, 1), "2023-24"), (dt.date(2099, 6, 1), "2099-00"),
])
def test_financial_year(d, fy):
    assert financial_year(d) == fy


def test_psp_date_from_name():
    assert psp_date_from_name("files/grdw/2026/11/01.11.26_NLDC_PSP_491.xls") == dt.date(2026, 11, 1)
    assert psp_date_from_name("31.12.23_NLDC_PSP.pdf") == dt.date(2023, 12, 31)
    assert psp_date_from_name("files/grdwNA") is None
    assert psp_date_from_name("32.13.23_NLDC_PSP.xls") is None


def test_safe_filename():
    assert safe_filename("files/grdw/2026/10/Week%20ahead%20x_413.pdf") == "Week ahead x_413.pdf"
    assert safe_filename("files/../../etc/passwd") == "passwd"
    with pytest.raises(ValueError):
        safe_filename("files/grdw/..")


def test_find_psp_prefers_xls_and_checks_next_fy(tmp_path):
    c = GridIndiaClient(cache_dir=tmp_path)
    # 31-Mar-2024 report is published in April -> listed under FY 2024-25
    c._listings[("DAILY_PSP_REPORT", "2023-24_00")] = []
    c._listings[("DAILY_PSP_REPORT", "2024-25_00")] = [
        {"Id": 1, "FilePath": "files/grdw/2024/04/31.03.24_NLDC_PSP_1.pdf"},
        {"Id": 2, "FilePath": "files/grdw/2024/04/31.03.24_NLDC_PSP_2.xls"},
        {"Id": 3, "FilePath": "files/grdwNA", "Title_": "31.03.24_NLDC_PSP"},
    ]
    rec = c.find_psp(dt.date(2024, 3, 31))
    assert rec["FilePath"].endswith("_2.xls")
    assert c.find_psp(dt.date(2024, 3, 30)) is None
