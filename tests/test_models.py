"""Model tests: quantile ordering, persistence, what-if behaviour, Monte Carlo correctness."""
import numpy as np
import pandas as pd
import pytest

from src.features.build import make_tabular, valid_origins
from src.insights import (Scenario, evening_stress, record_probability, reliability, sample_paths,
                          to_z, what_if)
from src.models.baselines import LGBMQuantile, SeasonalNaive

TRAIN_END, ORIGIN = "2024-10-31", pd.Timestamp("2024-11-20")


@pytest.fixture(scope="module")
def lgbm(demand_world):
    daily, fc = demand_world
    tab = make_tabular(daily, fc, valid_origins(daily, end=TRAIN_END))
    return LGBMQuantile({"n_estimators": 150, "learning_rate": 0.08}).fit(tab)


def _ordered(df):
    return bool(((df["p10"] <= df["p50"] + 1e-6) & (df["p50"] <= df["p90"] + 1e-6)).all())


def test_lgbm_and_naive_quantiles_ordered(demand_world, lgbm):
    daily, fc = demand_world
    te = make_tabular(daily, fc, valid_origins(daily, start="2024-11-01"))
    p = lgbm.predict_long(te)
    assert _ordered(p) and p["p50"].notna().all()
    tr = make_tabular(daily, fc, valid_origins(daily, end=TRAIN_END))
    assert _ordered(SeasonalNaive().fit(tr).predict_long(te))
    # the model learned something real: much better than seasonal naive on this world
    mape = lambda d: ((d.p50 - d.actual).abs() / d.actual).mean()  # noqa: E731
    assert mape(p) < mape(SeasonalNaive().fit(tr).predict_long(te))


@pytest.mark.parametrize("kind", ["lstm", "tft"])
def test_deep_quantiles_ordered_and_save_load(demand_world, kind, tmp_path, monkeypatch):
    import torch

    import src.models.deep as deep

    monkeypatch.setattr(deep, "REPORTS", tmp_path)
    daily, fc = demand_world
    m = deep.DeepForecaster(kind, {"batch_size": 64}, enc_days=30, max_epochs=3, patience=2,
                            device=torch.device("cpu"))
    m.fit(daily, fc, valid_origins(daily, start="2023-06-01", end=TRAIN_END))
    p = m.predict_long(daily, fc, [ORIGIN])
    assert len(p) == 7 * 3 and _ordered(p)
    out = m.save(root=tmp_path)
    m2 = deep.DeepForecaster.load(out, device=torch.device("cpu"))
    p2 = m2.predict_long(daily, fc, [ORIGIN])
    np.testing.assert_allclose(p[["p10", "p50", "p90"]], p2[["p10", "p50", "p90"]], rtol=1e-4)


def test_what_if_zero_scenario_is_identity(demand_world, lgbm):
    daily, fc = demand_world
    r = what_if(lgbm, daily, fc, ORIGIN, Scenario())
    assert len(r) == 21
    assert np.allclose(r["delta_p50"], 0)


def test_what_if_heat_raises_and_holiday_lowers_demand(demand_world, lgbm):
    daily, fc = demand_world
    hot = what_if(lgbm, daily, fc, ORIGIN, Scenario(temp_delta={"all": 4}))
    assert (hot["delta_p50"] > 0).mean() > 0.8
    north = what_if(lgbm, daily, fc, ORIGIN, Scenario(temp_delta={"nr": 4}))
    assert 0 < north["delta_p50"].mean() < hot["delta_p50"].mean()  # one region < all regions
    day = ORIGIN + pd.Timedelta(days=3)
    hol = what_if(lgbm, daily, fc, ORIGIN, Scenario(holidays=[day]))
    on_day = hol[hol["date"] == day]
    assert (on_day["delta_p50"] < 0).all()
    # neighbours move a little (days-to/from-festival features change), the day itself most
    for t, g in hol.groupby("target"):
        own = g.loc[g["date"] == day, "delta_p50"].abs().iloc[0]
        assert g.loc[g["date"] != day, "delta_p50"].abs().max() < own
    assert _ordered(hot.rename(columns={"p10_scen": "p10", "p50_scen": "p50", "p90_scen": "p90"}))


def _pred(p50=200_000.0, spread=10_000.0):
    rows = []
    for h in range(1, 8):
        for t, m in (("solar_peak", p50), ("evening_peak", p50 - 10_000), ("energy", 4500)):
            s = spread if t != "energy" else 200
            rows.append({"h": h, "target": t, "p10": m - s, "p50": m, "p90": m + 1.5 * s})
    return pd.DataFrame(rows)


def test_monte_carlo_reproduces_quantiles():
    s = sample_paths(_pred(), n=40000, corr=np.eye(21))
    q = np.quantile(s["solar_peak"][:, 0], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(q, [190_000, 200_000, 215_000], rtol=0.004)
    assert to_z(215_000, 190_000, 200_000, 215_000) == pytest.approx(1.2816, abs=1e-3)


def test_record_probability_monotone_and_correlation_matters():
    pred = _pred()
    lo = record_probability(pred, {"all_time": 210_000, "year": 205_000}, corr=np.eye(21))
    hi = record_probability(pred, {"all_time": 230_000, "year": 205_000}, corr=np.eye(21))
    assert 0 <= hi["p_all_time"] < lo["p_all_time"] <= 1
    assert lo["p_year"] > lo["p_all_time"]
    # perfectly correlated days: P(any) equals the single-day probability; independent: larger
    full = record_probability(pred, {"all_time": 215_000, "year": 1e9}, corr=np.ones((21, 21)) * 0.999 + np.eye(21) * 0.001)
    ind = record_probability(pred, {"all_time": 215_000, "year": 1e9}, corr=np.eye(21))
    assert full["p_all_time"] < ind["p_all_time"]


def test_reliability_bins():
    rel = reliability(np.array([0.01, 0.05, 0.3, 0.8]), np.array([0, 0, 1, 1]))
    assert rel["n"].sum() == 4


def test_evening_stress_table(demand_world):
    daily, _ = demand_world
    st = evening_stress(_pred(), daily, ORIGIN, corr=np.eye(21))
    assert len(st) == 7
    assert (st["gap_p10"] <= st["gap_p50"]).all() and (st["gap_p50"] <= st["gap_p90"]).all()
    assert st["alert"].dtype == bool
