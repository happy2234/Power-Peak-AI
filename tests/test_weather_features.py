"""All-weather features: derived variables, anomaly normals (train-only), 3-day rain, the
decoder's per-day forecast weather, and plain-language weather drivers."""
import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON, REGIONS
from src.features.build import (
    ANOM_COLS, DEC_FEATS, NORMALS_END, WX_FEATS, derive_weather, fit_normals, fog_proxy, heat_index,
    make_sequences, make_tabular, normal_at,
)
from tests.test_features import ORIGIN, synthetic

W = {"nr": 0.3, "wr": 0.3, "sr": 0.25, "er": 0.13, "ner": 0.02}


def _obs(seed=0, start="2023-03-01", end="2026-11-08"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, name="date")
    doy = idx.day_of_year.to_numpy()
    df = pd.DataFrame(index=idx)
    for c in ANOM_COLS:
        df[c] = 25 + 8 * np.sin(2 * np.pi * doy / 365.25) + rng.standard_normal(len(idx))
    return df


def test_normals_use_training_data_only():
    obs = _obs()
    a = fit_normals(obs, ANOM_COLS)
    late = obs.copy()
    late.loc[late.index > pd.Timestamp(NORMALS_END)] = 1e6          # corrupt everything after the cut-off
    assert fit_normals(late, ANOM_COLS) == a
    early = obs.copy()
    early.loc[early.index <= pd.Timestamp(NORMALS_END)] += 5         # change the training period
    b = fit_normals(early, ANOM_COLS)
    assert b["tmax_ai"][0] == pytest.approx(a["tmax_ai"][0] + 5, abs=1e-6)


def test_normals_cutoff_precedes_every_backtest_origin():
    from src.evaluate import BACKTEST_START
    from src.models.tune import BACKTEST_START as TUNE_START

    assert pd.Timestamp(NORMALS_END) <= pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=8)
    assert TUNE_START == BACKTEST_START


def test_anomaly_is_value_minus_seasonal_normal():
    obs = _obs()
    nrm = fit_normals(obs, ANOM_COLS)
    d = pd.DatetimeIndex(["2026-01-15", "2026-05-20"])
    want = normal_at(nrm, "tmax_ai", d)
    assert want[1] > want[0]                                         # May is warmer than January
    reg = pd.DataFrame({f"{v}_{r}": [20.0, 40.0] for v in ("tmax", "tmean", "tmin") for r in REGIONS})
    out = derive_weather(reg, W, dates=d, normals=nrm)
    np.testing.assert_allclose(out["tmax_ai_anom"], [20 - want[0], 40 - want[1]])
    # without normals the anomaly is unknown, never silently zero
    assert derive_weather(reg, W, dates=d)["tmax_ai_anom"].isna().all()


def test_degree_days_rain_fog_heat_index():
    reg = pd.DataFrame({f"{v}_{r}": vals for r in REGIONS for v, vals in {
        "tmean": [30.0, 10.0, 21.0], "tmax": [38.0, 16.0, 30.0], "tmin": [27.0, 6.0, 14.0],
        "precip": [0.0, 1.0, 12.0], "dew_mean": [26.0, 5.0, 16.0], "dpd_morning": [8.0, 0.5, 4.0]}.items()})
    out = derive_weather(reg, W)
    np.testing.assert_allclose(out["cdd_ai"], [6, 0, 0])
    np.testing.assert_allclose(out["hdd_ai"], [0, 8, 0])
    np.testing.assert_allclose(out["rain_day_nr"], [0, 0, 1])
    assert out["fog_nr"].iloc[1] > 0.8 and out["fog_nr"].iloc[0] == 0 and out["fog_nr"].iloc[2] == 0
    assert out["heat_index_ai"].iloc[0] > 38                          # muggy 38 °C feels hotter
    assert heat_index([20.0], [10.0])[0] == pytest.approx(19.4, abs=0.6)
    assert fog_proxy([10.0], [np.nan])[0] != fog_proxy([10.0], [np.nan])[0]  # unknown stays NaN


def test_rain3_mixes_observed_past_and_forecast_future():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc, [ORIGIN])
    f = fc.set_index(["date", "lead"])
    for h in (1, 2, 3, 5):
        row = tab[tab["h"] == h].iloc[0]
        want = 0.0
        for j in (h - 2, h - 1, h):
            day = ORIGIN + pd.Timedelta(days=j)
            want += daily.loc[day, "precip_ai"] if j <= 0 else f.loc[(day, j), "precip_ai"]
        assert row["rain3_ai"] == pytest.approx(want)


def test_lstm_decoder_gets_forecast_weather_for_each_future_day():
    daily, fc = synthetic()
    s = make_sequences(daily, fc, [ORIGIN], enc_days=60)
    f = fc.set_index(["date", "lead"])
    for c in ("tmax_ai", "precip_ai", "cloud_day_ai", "hdd_nr", "tmin_ai_anom"):
        i = DEC_FEATS.index(c)
        for h in range(1, HORIZON + 1):
            want = f.loc[(ORIGIN + pd.Timedelta(days=h), h), c]
            assert s["dec"][0, h - 1, i] == pytest.approx(want, rel=1e-5)
    # changing the forecast for day 3 only changes decoder step 3 (and rain3 of days 3..5)
    fc2 = fc.copy()
    m = (fc2["date"] == ORIGIN + pd.Timedelta(days=3)) & (fc2["lead"] == 3)
    fc2.loc[m, "tmax_ai"] += 10
    s2 = make_sequences(daily, fc2, [ORIGIN], enc_days=60)
    i = DEC_FEATS.index("tmax_ai")
    diff = s2["dec"][0, :, i] - s["dec"][0, :, i]
    assert diff[2] == pytest.approx(10) and np.allclose(np.delete(diff, 2), 0)
    np.testing.assert_array_equal(s["enc"], s2["enc"])


def test_every_weather_feature_reaches_both_models():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc, [ORIGIN])
    for c in WX_FEATS:
        assert c in tab.columns and c in DEC_FEATS
    for kind in ("tmin", "hdd", "precip", "cloud_day", "fog", "wind_max", "gust_max", "dew_mean",
                 "heat_index", "rh_mean", "rad", "rain_day"):
        assert any(c.startswith(kind) for c in WX_FEATS), kind


def test_missing_archive_variables_fall_back_to_persistence_not_hindsight():
    daily, fc = synthetic()
    fc2 = fc.copy()
    fc2["cloud_day_ai"] = np.nan              # like the archive before 2024
    tab = make_tabular(daily, fc2, [ORIGIN])
    pers = daily["cloud_day_ai"].loc[ORIGIN - pd.Timedelta(days=2):ORIGIN].mean()
    assert np.allclose(tab["cloud_day_ai"], pers)
    assert (tab["wx_source"] == 0).all()      # temperature still came from the forecast


def test_long_lead_lags_never_after_origin():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc, [ORIGIN], horizons=range(1, 13),
                       climatology=lambda d: pd.DataFrame(0.0, index=d, columns=WX_FEATS))
    for k in ("energy", "solar_peak"):
        assert tab.loc[tab["h"] > 7, f"{k}_lag7"].isna().all()
        assert tab.loc[tab["h"] <= 7, f"{k}_lag7"].notna().all()
    assert (tab.loc[tab["h"] > 7, "wx_source"] == 3).all()


def test_weather_driver_sentences():
    from src.insights import _group_label, feature_group, weather_drivers

    assert feature_group("precip_sr_anom") == "wx_rain_sr"
    assert feature_group("hdd_nr") == "wx_cold_nr"
    assert feature_group("rain3_ai") == "wx_rain_all"
    assert feature_group("d_tmax_ai") == "wx_heat_all"
    assert feature_group("energy_lag7") == "last weeks' pattern"
    row = pd.Series({"dow": 0})
    assert _group_label("wx_rain_sr", row) == "Rain in South India"
    assert _group_label("wx_cold_nr", row) == "Cold morning in North India"
    out = weather_drivers({"Rain in South India": -2.14, "Cold morning in North India": 3.4,
                           "Monday": 5.0, "Fog in North India": 0.01}, "GW")
    assert out == ["Cold morning in North India: +3.4 GW", "Rain in South India: −2.1 GW"]
