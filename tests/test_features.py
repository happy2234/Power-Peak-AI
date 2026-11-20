"""Feature tests: leakage, window shapes, reference levels, calendar.

Leakage tests use synthetic data: for a forecast origin t, every piece of information that
would only be known after t (actuals, observed weather, weather forecasts issued after t) is
replaced with random noise. The features for origin t must not change at all.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON, REGIONS
from src.features.build import (
    CAL_FEATS, DEC_FEATS, ENC_FEATS, TGT, WX_FEATS, make_sequences, make_tabular,
    reference_levels, tabular_feature_cols, valid_origins,
)
from src.features.calendar import calendar_features

ORIGIN = pd.Timestamp("2024-06-15")


def synthetic(seed=0, fc_gaps=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-04-01", "2024-09-30", name="date")
    n = len(idx)
    cols = {}
    for k, lvl in zip(TGT, (210_000, 200_000, 4500)):
        cols[k] = lvl * (1 + 0.05 * np.sin(np.arange(n) / 58) + 0.02 * rng.standard_normal(n))
        cols[f"{k}_imputed"] = False
    cols["is_imputed"] = False
    cols["is_outlier"] = False
    for c in WX_FEATS:
        cols[c] = 30 + 5 * rng.standard_normal(n)
    cal = calendar_features(idx)
    for c in CAL_FEATS:
        cols[c] = cal[c].to_numpy()
    daily = pd.DataFrame(cols, index=idx)
    rows = []
    for d in idx:
        for lead in range(1, HORIZON + 1):
            if fc_gaps and rng.random() < 0.4:
                continue
            rows.append({"date": d, "lead": lead, **{c: 30 + rng.standard_normal() for c in WX_FEATS}})
    fc = pd.DataFrame(rows)
    return daily, fc


def corrupt_after(daily, fc, t, seed=99):
    """Replace everything not known at origin t with noise."""
    rng = np.random.default_rng(seed)
    d2, f2 = daily.copy(), fc.copy()
    fut = d2.index > t
    for c in TGT + WX_FEATS:
        d2.loc[fut, c] = rng.uniform(1, 1e6, fut.sum())
    issued = f2["date"] - pd.to_timedelta(f2["lead"], unit="D")
    late = issued > t
    for c in WX_FEATS:
        f2.loc[late, c] = rng.uniform(-50, 99, late.sum())
    return d2, f2


@pytest.mark.parametrize("fc_gaps", [False, True])
def test_no_leakage_tabular(fc_gaps):
    daily, fc = synthetic(fc_gaps=fc_gaps)
    a = make_tabular(daily, fc, [ORIGIN])
    d2, f2 = corrupt_after(daily, fc, ORIGIN)
    b = make_tabular(d2, f2, [ORIGIN])
    cols = tabular_feature_cols(a) + [f"ref_{k}" for k in TGT]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    # sanity: the corruption really changed the labels, so the test has teeth
    assert not np.allclose(a["y_energy"], b["y_energy"])


@pytest.mark.parametrize("fc_gaps", [False, True])
def test_no_leakage_sequences(fc_gaps):
    daily, fc = synthetic(fc_gaps=fc_gaps)
    a = make_sequences(daily, fc, [ORIGIN], enc_days=60)
    d2, f2 = corrupt_after(daily, fc, ORIGIN)
    b = make_sequences(d2, f2, [ORIGIN], enc_days=60)
    np.testing.assert_array_equal(a["enc"], b["enc"])
    np.testing.assert_array_equal(a["dec"], b["dec"])
    np.testing.assert_array_equal(a["ref"], b["ref"])
    assert not np.allclose(a["y"], b["y"])


def test_weather_uses_forecast_issued_at_origin():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc, [ORIGIN])
    for h in range(1, HORIZON + 1):
        row = tab[tab["h"] == h].iloc[0]
        want = fc[(fc["date"] == ORIGIN + pd.Timedelta(days=h)) & (fc["lead"] == h)]["tmax_ai"].iloc[0]
        assert row["tmax_ai"] == pytest.approx(want)
        assert row["wx_source"] == 0


def test_persistence_fallback_when_no_forecast():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc.iloc[0:0], [ORIGIN])
    expected = daily["tmax_ai"].loc[ORIGIN - pd.Timedelta(days=2):ORIGIN].mean()
    assert (tab["wx_source"] == 2).all()
    assert np.allclose(tab["tmax_ai"], expected)


def test_window_shapes():
    daily, fc = synthetic()
    origins = valid_origins(daily, start="2024-01-01", end="2024-01-31")
    s = make_sequences(daily, fc, origins, enc_days=60)
    n = len(origins)
    assert s["enc"].shape == (n, 60, len(ENC_FEATS))
    assert s["dec"].shape == (n, HORIZON, len(DEC_FEATS))
    assert s["y"].shape == (n, HORIZON, len(TGT))
    assert s["ref"].shape == (n, len(TGT))
    assert np.isfinite(s["enc"]).all() and np.isfinite(s["dec"]).all()
    tab = make_tabular(daily, fc, origins)
    assert len(tab) == n * HORIZON
    assert set(tab["h"]) == set(range(1, HORIZON + 1))
    assert (tab["date"] - tab["origin"]).dt.days.eq(tab["h"]).all()


def test_reference_level_is_last_7_days_mean():
    daily, _ = synthetic()
    ref = reference_levels(daily, pd.DatetimeIndex([ORIGIN]))
    exp = daily.loc[ORIGIN - pd.Timedelta(days=6):ORIGIN, "energy"].mean()
    assert ref["energy"].iloc[0] == pytest.approx(exp)


def test_lags_never_after_origin():
    daily, fc = synthetic()
    tab = make_tabular(daily, fc, [ORIGIN])
    for _, r in tab.iterrows():
        for lag in (7, 14, 364):
            assert r["date"] - pd.Timedelta(days=lag) <= ORIGIN


def test_calendar_festivals_and_whatif_holiday():
    c = calendar_features(pd.date_range("2026-11-05", "2026-11-11"))
    assert c.loc["2026-11-08", "fest_diwali"] == 1 and c.loc["2026-11-08", "is_sunday"] == 1
    assert c.loc["2026-11-06":"2026-11-10", "diwali_window"].eq(1).all()
    assert c.loc["2026-11-05", "days_to_festival"] == 3
    c2 = calendar_features(pd.date_range("2026-11-02", "2026-11-04"), extra_holidays=["2026-11-03"])
    assert c2.loc["2026-11-03", "is_holiday"] == 1 and c2.loc["2026-11-02", "is_holiday"] == 0
    # subset/unordered input gives the same values as a contiguous range
    sub = calendar_features(pd.DatetimeIndex(["2026-11-10", "2026-11-06"]))
    assert sub.loc["2026-11-06"].equals(c.loc["2026-11-06"])


def test_regions_constant():
    assert REGIONS == ["nr", "wr", "sr", "er", "ner"]
