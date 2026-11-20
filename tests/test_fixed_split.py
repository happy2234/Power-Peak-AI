"""Fixed train/test split: no test-period data reaches training, scaling or the normals."""
import numpy as np
import pandas as pd
import pytest

from src import fixed_split as fx
from src.config import HORIZON, TARGETS
from src.features.build import WX_FEATS, WX_RAW, make_tabular

# synthetic world (tests/conftest.py) runs 2023-04-01 .. 2024-12-31
VAL, END, TEST = "2024-07-01", "2024-09-30", "2024-10-01"
TGT = list(TARGETS)


def corrupt_after(daily, fc, cut, seed=7):
    """Replace every value dated after `cut` (actuals, observed weather, any weather forecast for
    or issued after it) with noise."""
    rng = np.random.default_rng(seed)
    d2, f2 = daily.copy(), fc.copy()
    late = d2.index > pd.Timestamp(cut)
    for c in TGT + [c for c in dict.fromkeys(WX_FEATS + WX_RAW) if c in d2]:
        d2.loc[late, c] = rng.uniform(1, 1e6, late.sum())
    bad = (f2["date"] > pd.Timestamp(cut)) | (f2["date"] - pd.to_timedelta(f2["lead"], unit="D") > pd.Timestamp(cut))
    for c in [c for c in f2.columns if c not in ("date", "lead")]:
        f2.loc[bad, c] = rng.uniform(-50, 99, bad.sum())
    return d2, f2


def test_split_boundaries(demand_world):
    daily, _ = demand_world
    fit_o, val_o, test_o = fx.split_origins(daily, VAL, END, TEST)
    assert fit_o.max() + pd.Timedelta(days=HORIZON) < pd.Timestamp(VAL)        # fit labels end before validation
    assert val_o.min() >= pd.Timestamp(VAL)
    assert val_o.max() + pd.Timedelta(days=HORIZON) <= pd.Timestamp(END)       # validation labels inside train period
    assert test_o.min() == pd.Timestamp(TEST) - pd.Timedelta(days=1)          # first test day gets all 7 horizons
    assert not set(fit_o) & set(test_o) and not set(val_o) & set(test_o)


def test_training_and_scaling_never_see_the_test_period(demand_world):
    daily, fc = demand_world
    fit_o, val_o, test_o = fx.split_origins(daily, VAL, END, TEST)
    params = {"lgbm": {"n_estimators": 60}, "lstm": {"hidden": 16, "batch_size": 64}}
    a = fx.train(daily, fc, fit_o, val_o, kinds=("seasonal_naive", "lgbm", "lstm"), params=params, max_epochs=3)
    d2, f2 = corrupt_after(daily, fc, END)
    b = fx.train(d2, f2, fit_o, val_o, kinds=("seasonal_naive", "lgbm", "lstm"), params=params, max_epochs=3)
    # same inputs for the train/validation origins, so identical fitted models
    tab = make_tabular(daily, fc, val_o)
    for k in TGT:
        np.testing.assert_allclose(a["lgbm"].predict(tab)[k], b["lgbm"].predict(tab)[k])
        np.testing.assert_allclose(a["seasonal_naive"].q[k], b["seasonal_naive"].q[k])
    for attr in ("enc_mu", "enc_sd", "dec_mu", "dec_sd", "y_sd"):
        np.testing.assert_array_equal(getattr(a["lstm"], attr), getattr(b["lstm"], attr))
    # the corruption is real: test-period predictions see different inputs
    t1 = make_tabular(daily, fc, test_o[:20])
    t2 = make_tabular(d2, f2, test_o[:20])
    assert not np.allclose(a["lgbm"].predict(t1)["energy"], a["lgbm"].predict(t2)["energy"])


def test_scaler_uses_the_training_slice_only(demand_world):
    """Even the validation months do not enter the neural scaler."""
    from src.models.deep import DeepForecaster

    daily, fc = demand_world
    fit_o, val_o, _ = fx.split_origins(daily, VAL, END, TEST)
    origins = fit_o.append(val_o)
    m1 = DeepForecaster("lstm", {"hidden": 16}, max_epochs=1, device=__import__("torch").device("cpu"))
    m1.fit(daily, fc, origins, refit_all=False, valid_start=val_o[0])
    d2, f2 = corrupt_after(daily, fc, val_o[0] - pd.Timedelta(days=1))
    m2 = DeepForecaster("lstm", {"hidden": 16}, max_epochs=1, device=__import__("torch").device("cpu"))
    m2.fit(d2, f2, origins, refit_all=False, valid_start=val_o[0])
    for attr in ("enc_mu", "enc_sd", "dec_mu", "dec_sd", "y_sd"):
        np.testing.assert_array_equal(getattr(m1, attr), getattr(m2, attr))


def test_lgbm_without_refit_is_fitted_on_the_training_slice(demand_world):
    """refit=False: changing the validation labels may move early stopping, but a model with a
    fixed number of trees is identical, i.e. the validation rows never enter the fit."""
    from src.models.baselines import LGBMQuantile

    daily, fc = demand_world
    fit_o, val_o, _ = fx.split_origins(daily, VAL, END, TEST)
    tab = make_tabular(daily, fc, fit_o.append(val_o))
    tab2 = tab.copy()
    v = tab2["origin"] >= val_o[0]
    for k in TGT:
        tab2.loc[v, f"y_{k}"] *= 1.3
    p = {"n_estimators": 40}
    a = LGBMQuantile(p).fit(tab, valid_start=val_o[0], refit=False)
    b = LGBMQuantile(p).fit(tab2, valid_start=val_o[0], refit=False)
    same_iters = {key: a.best_iters[key] == b.best_iters[key] for key in a.best_iters}
    x = make_tabular(daily, fc, fit_o[:30])
    for (k, q), same in same_iters.items():
        if same:
            np.testing.assert_allclose(a.models[(k, q)].predict(x[a.features]), b.models[(k, q)].predict(x[a.features]))


def test_normals_for_the_fixed_split_end_with_the_training_period(monkeypatch):
    import src.features.build as fb

    seen = {}
    monkeypatch.setattr(fb, "build_daily_features", lambda save, normals_end: seen.update(save=save, end=normals_end))
    fx.features()
    assert seen == {"save": False, "end": fx.TRAIN_END}
    assert pd.Timestamp(fx.TRAIN_END) < pd.Timestamp(fx.TEST_START)
    assert fx.fy("2025-03-31") == "FY 2024-25" and fx.fy("2025-04-01") == "FY 2025-26"


def test_live_summary_separates_real_time_from_back_filled(tmp_path):
    days = pd.date_range("2026-11-02", periods=3)
    hist = pd.DataFrame({"model": ["ensemble"] * 3 + ["grid_india"] * 3, "origin": pd.Timestamp("2026-11-01"),
                         "date": list(days) * 2, "h": [1, 2, 3] * 2, "target": "energy",
                         "p10": np.nan, "p50": [100.0, 110, 90, 120, 120, 120], "p90": np.nan, "is_best": [True] * 3 + [False] * 3,
                         "issued_at": [pd.Timestamp("2026-11-03 12:00")] * 6})
    hist.to_parquet(tmp_path / "forecast_history.parquet")
    feat = pd.DataFrame({"date": days, "energy": [100.0, 100, 100], "solar_peak": np.nan, "evening_peak": np.nan})
    feat.to_parquet(tmp_path / "features.parquet")
    s = fx.live_summary(tmp_path)
    assert s["days_scored"] == 3 and s["real_time_days"] == 1          # only 04 Nov was forecast in advance
    row = [r for r in s["by_model"] if r["model"] == "ensemble"][0]
    assert row["MAPE"] == pytest.approx((0 + 10 + 10) / 3)
    assert row["win_rate"] == pytest.approx(1.0)
