from pathlib import Path

import pandas as pd
import pytest

from src.parse.forecast_parser import parse_forecast_curve, parse_forecast_labels, parse_forecast_pdf

FIX = Path(__file__).parent / "fixtures"
RANGE_PDF = FIX / "Week ahead all India demand forecast 26-Oct-2026 to 01-Nov-2026_413.pdf"
POINT_PDF = FIX / "Week ahead all India demand forecast 26-Jan-2026 to 01-Feb-2026_805.pdf"
# Glyph-outline labels (no text) + typo'd title: PDF text "04-Dec-2025 to 10-Jan-2025",
# filename "04-Jan-2024 to 10-Jan-2025"; footer says prepared 1/3/2025 -> week of 04-Jan-2025.
OLD_PDF = FIX / "Week ahead all India demand forecast 04-Jan-2024 to 10-Jan-2025.pdf"


def test_range_format_daily():
    df = parse_forecast_pdf(RANGE_PDF)
    assert len(df) == 7
    assert list(df["date"]) == list(pd.date_range("2026-10-26", periods=7))
    first = df.iloc[0]
    assert (first["low_mw"], first["high_mw"]) == (234000, 238000)
    assert (first["non_solar_hr_low_mw"], first["non_solar_hr_high_mw"]) == (221000, 225000)
    last = df.iloc[-1]
    assert (last["low_mw"], last["high_mw"]) == (223000, 228000)
    assert (df["low_mw"] <= df["high_mw"]).all()
    assert df["curve_ok"].all()


def test_range_format_labels_two_per_day():
    lab = parse_forecast_labels(RANGE_PDF)
    assert len(lab) == 14
    assert lab.groupby("date")["peak_type"].apply(list).map(sorted).eq(
        [["non_solar_hr", "solar_hr"]] * 7).all()
    assert lab["curve_ok"].all()


def test_point_format_and_winter_evening_peak():
    lab = parse_forecast_labels(POINT_PDF)
    assert len(lab) == 14 and (lab["label_kind"] == "point").all()
    assert lab["curve_ok"].all()
    # 26-Jan evening peak is labelled at ~17:45 -> must still be non_solar_hr
    jan26 = lab[lab["date"] == "2026-01-26"].set_index("peak_type")
    assert jan26.loc["solar_hr", "low_mw"] == 230937
    assert jan26.loc["non_solar_hr", "low_mw"] == 201259
    # curve digitisation matches the published labels within 0.1 GW
    assert (lab["curve_peak_mw"] - lab["low_mw"]).abs().max() < 100


def test_old_format_dates_resolved_and_curve_fallback():
    df = parse_forecast_pdf(OLD_PDF)
    assert df["date"].iloc[0] == pd.Timestamp("2025-01-04")
    assert len(df) == 7
    assert (df["label_kind"] == "curve").all()
    assert df["high_mw"].between(150_000, 300_000).all()


def test_curve():
    c = parse_forecast_curve(RANGE_PDF)
    assert len(c) == 7 * 96
    assert c["timestamp"].iloc[0] == pd.Timestamp("2026-10-26")
    inner = c.dropna()
    assert len(inner) > 600
    assert (inner["band_low_mw"] <= inner["forecast_mw"] + 500).all()
    assert (inner["forecast_mw"] <= inner["band_high_mw"] + 500).all()
    assert inner["forecast_mw"].between(150_000, 260_000).all()


@pytest.mark.parametrize("pdf", [RANGE_PDF, POINT_PDF, OLD_PDF])
def test_daily_schema_stable(pdf):
    cols = ["forecast_start", "date", "low_mw", "high_mw", "solar_hr_low_mw", "solar_hr_high_mw",
            "non_solar_hr_low_mw", "non_solar_hr_high_mw", "label_kind", "n_labels", "curve_ok",
            "source_file"]
    assert list(parse_forecast_pdf(pdf).columns) == cols
