from pathlib import Path

import pandas as pd
import pytest

from src.parse.psp_parser import (
    date_from_filename,
    parse_daily_summary,
    parse_state_demand,
    parse_timeseries,
)

FIX = Path(__file__).parent / "fixtures"
SAMPLE = FIX / "01.11.26_NLDC_PSP_491.xls"   # current layout (with TimeSeries sheet)
OLD_XLS = FIX / "01.06.23_NLDC_PSP.xls"      # 2023 layout: shifted rows, no TimeSeries
OLD_XLSX = FIX / "20.04.23_NLDC_PSP.xlsx"    # 2023 .xlsx variant


def test_date_from_filename():
    assert str(date_from_filename(SAMPLE)) == "2026-11-01"
    assert str(date_from_filename("31.12.23_NLDC_PSP.xls")) == "2023-12-31"


# ---------------------------------------------------------------- daily summary
def test_summary_sample_values():
    s = parse_daily_summary(SAMPLE).iloc[0]
    assert s["date"] == pd.Timestamp("2026-11-01")
    assert s["report_date"] == pd.Timestamp("2026-11-02")
    assert s["energy_met_mu_all_india"] == pytest.approx(4613.82)
    assert s["max_demand_mw_all_india"] == 215463
    assert s["max_demand_time_all_india"] == "09:42"
    assert s["evening_peak_demand_mw_all_india"] == 200087
    assert s["energy_met_mu_nr"] == pytest.approx(1159.0)
    assert s["solar_gen_mu_all_india"] == pytest.approx(625.49)
    assert s["energy_shortage_mu_all_india"] == pytest.approx(0.15)
    assert pd.isna(s["wind_gen_mu_er"])  # '-' in sheet
    assert s["peak_demand_non_solar_hr_mw"] == 203172
    assert s["peak_demand_non_solar_hr_time"] == "19:00"


def test_summary_regions_sum_to_total():
    s = parse_daily_summary(SAMPLE).iloc[0]
    regional = sum(s[f"energy_met_mu_{r}"] for r in ["nr", "wr", "sr", "er", "ner"])
    assert regional == pytest.approx(s["energy_met_mu_all_india"], abs=0.05)


@pytest.mark.parametrize("path,energy,max_mw,time", [
    (OLD_XLS, 4458.07, 199477, "15:29"),
    (OLD_XLSX, 4752.78, 211804, "14:56"),
])
def test_summary_old_layouts(path, energy, max_mw, time):
    s = parse_daily_summary(path).iloc[0]
    assert s["energy_met_mu_all_india"] == pytest.approx(energy)
    assert s["max_demand_mw_all_india"] == max_mw
    assert s["max_demand_time_all_india"] == time
    # 2023 label is "Peak Shortage (MW)" instead of "Shortage (MW) during Evening Peak"
    assert s.notna()["evening_peak_shortage_mw_all_india"]


def test_summary_columns_stable_across_layouts():
    cols = [set(parse_daily_summary(p).columns) for p in (SAMPLE, OLD_XLS, OLD_XLSX)]
    assert cols[0] == cols[1] == cols[2]


# ---------------------------------------------------------------- states
def test_state_demand_sample():
    st = parse_state_demand(SAMPLE)
    assert {"date", "region", "state", "max_demand_mw", "energy_met_mu", "energy_shortage_mu"} <= set(st.columns)
    assert st["region"].notna().all()
    up = st.set_index("state").loc["UP"]
    assert up["region"] == "NR" and up["max_demand_mw"] == 18381 and up["energy_met_mu"] == pytest.approx(350.69)
    # Region label sits on the middle row of each group; first rows must still be right.
    assert st.set_index("state").loc["Punjab", "region"] == "NR"
    assert st.set_index("state").loc["Chhattisgarh", "region"] == "WR"
    assert st.set_index("state").loc["DVC", "energy_shortage_mu"] == pytest.approx(0.15)
    assert set(st["region"]) == {"NR", "WR", "SR", "ER", "NER"}


def test_state_energy_matches_region_totals():
    st = parse_state_demand(SAMPLE)
    s = parse_daily_summary(SAMPLE).iloc[0]
    by_region = st.groupby("region")["energy_met_mu"].sum()
    for reg in ["NR", "WR", "SR", "ER", "NER"]:
        assert by_region[reg] == pytest.approx(s[f"energy_met_mu_{reg.lower()}"], rel=0.01)


def test_state_demand_old_layout():
    st = parse_state_demand(OLD_XLS)
    assert st["region"].notna().all()
    idx = st.set_index("state")
    assert idx.loc["AMNSIL", "region"] == "WR"
    assert idx.loc["Railways_ER ISTS", "region"] == "ER"


# ---------------------------------------------------------------- timeseries
def test_timeseries_sample():
    ts = parse_timeseries(SAMPLE)
    assert len(ts) == 96
    assert ts["timestamp"].iloc[0] == pd.Timestamp("2026-11-01 00:00")
    assert ts["timestamp"].iloc[-1] == pd.Timestamp("2026-11-01 23:45")
    assert ts["timestamp"].diff().dropna().eq(pd.Timedelta("15min")).all()
    assert ts["demand_met_mw"].iloc[0] == 183886
    for col in ["frequency_hz", "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw", "gas_mw",
                "thermal_mw", "others_mw"]:
        assert col in ts.columns and ts[col].notna().all(), col


def test_timeseries_peak_consistent_with_summary():
    """15-min SCADA peak should be close to the reported daily max (which is 1-min resolution)."""
    ts = parse_timeseries(SAMPLE)
    s = parse_daily_summary(SAMPLE).iloc[0]
    assert ts["demand_met_mw"].max() == pytest.approx(s["max_demand_mw_all_india"], rel=0.02)


def test_timeseries_missing_sheet_returns_empty():
    assert parse_timeseries(OLD_XLS).empty
