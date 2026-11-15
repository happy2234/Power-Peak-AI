"""Accuracy round: leakage tests for every new feature and option, and the test-period guard.

Every new input must be a function of data at or before the forecast origin, the calendar of the
target day, or the weather forecast known at the origin; weights and normals come from the
training rows only."""
import numpy as np
import pandas as pd
import pytest

from src import test_access
from src.config import TARGETS
from src.features.build import EXTRA_COLS, EXTRA_GROUPS, last_year_date, make_tabular, tabular_feature_cols
from src.features.build import WX_FEATS, WX_RAW

TGT = list(TARGETS)


def corrupt_after(daily, fc, cut, seed=7):
    """Scramble everything unknown at `cut`: actuals and observed weather after it, and weather
    forecasts issued after it (forecasts issued at or before `cut` for later days are legitimate)."""
    rng = np.random.default_rng(seed)
    d2, f2 = daily.copy(), fc.copy()
    late = d2.index > pd.Timestamp(cut)
    for c in TGT + [c for c in dict.fromkeys(WX_FEATS + WX_RAW) if c in d2]:
        d2.loc[late, c] = rng.uniform(1, 1e6, late.sum())
    issued = f2["date"] - pd.to_timedelta(f2["lead"], unit="D")
    bad = issued > pd.Timestamp(cut)
    for c in [c for c in f2.columns if c not in ("date", "lead")]:
        f2.loc[bad, c] = rng.uniform(-50, 99, bad.sum())
    return d2, f2
# origins well inside the synthetic world (2023-04-01 .. 2024-12-31), a year of history before them
ORIGINS = pd.date_range("2024-05-01", "2024-11-25", freq="17D")


def test_new_features_use_nothing_after_the_origin(demand_world):
    daily, fc = demand_world
    for o in ORIGINS:
        d2, f2 = corrupt_after(daily, fc, o)
        a = make_tabular(daily, fc, [o])
        b = make_tabular(d2, f2, [o])
        for c in EXTRA_COLS + WX_FEATS:          # new features and the forecast weather they build on
            np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), err_msg=f"{c} at {o.date()}")


@pytest.mark.parametrize("base", ["ref7", "last", "lag7"])
def test_target_bases_are_known_at_the_origin_and_invert(demand_world, base):
    daily, fc = demand_world
    for o in ORIGINS[:4]:
        d2, f2 = corrupt_after(daily, fc, o)
        a = make_tabular(daily, fc, [o], target_base=base)
        b = make_tabular(d2, f2, [o], target_base=base)
        for k in TGT:
            np.testing.assert_allclose(a[f"ref_{k}"], b[f"ref_{k}"])           # the base is known at issue time
            np.testing.assert_allclose(a[f"y_{k}"] * a[f"ref_{k}"], a[f"actual_{k}"])
    feats = tabular_feature_cols(a)
    assert not any(c.startswith(("ref_solar", "ref_evening", "ref_energy", "y_", "actual_")) for c in feats)


def test_extra_groups_are_opt_in(demand_world):
    daily, fc = demand_world
    tab = make_tabular(daily, fc, ORIGINS[:2])
    base = tabular_feature_cols(tab, "compact")
    assert not set(EXTRA_COLS) & set(base)                     # saved models keep their exact inputs
    with_g = tabular_feature_cols(tab, "compact", extra=("growth",))
    assert set(with_g) - set(base) == set(EXTRA_GROUPS["growth"])


def test_festival_aligned_last_year_date():
    d = pd.DatetimeIndex(["2025-10-20", "2025-10-22", "2025-07-15", "2025-03-14"])
    ly = last_year_date(d)
    assert ly[0] == pd.Timestamp("2024-11-01")           # Diwali 2025 -> Diwali 2024
    assert ly[1] == pd.Timestamp("2024-11-03")           # two days after, aligned the same way
    assert ly[2] == pd.Timestamp("2025-07-15") - pd.Timedelta(days=364)   # ordinary day: same weekday
    assert ly[3] == pd.Timestamp("2024-03-25")           # Holi 2025 -> Holi 2024


def test_recency_weights_come_from_training_rows_only(demand_world):
    """The decay is anchored on the newest TRAINING origin, so corrupting later data changes nothing."""
    from src.models.baselines import LGBMQuantile

    daily, fc = demand_world
    tr = pd.date_range("2023-06-01", "2024-06-30")
    va = pd.date_range("2024-07-10", "2024-08-31")
    params = {"n_estimators": 40, "num_leaves": 7}
    a = LGBMQuantile(params, recency_halflife=90).fit(make_tabular(daily, fc, tr.append(va)), valid_start=va[0],
                                                      refit=False)
    from tests.test_fixed_split import corrupt_after as corrupt_all

    d2, f2 = corrupt_all(daily, fc, "2024-09-07")
    b = LGBMQuantile(params, recency_halflife=90).fit(make_tabular(d2, f2, tr.append(va)), valid_start=va[0],
                                                      refit=False)
    tab = make_tabular(daily, fc, va[:10])
    for k in TGT:
        np.testing.assert_allclose(a.predict(tab)[k], b.predict(tab)[k])
    c = LGBMQuantile(params).fit(make_tabular(daily, fc, tr.append(va)), valid_start=va[0], refit=False)
    assert not np.allclose(a.predict(tab)["energy"], c.predict(tab)["energy"])   # the weights do something


def test_guard_blocks_test_period_origins(tmp_path):
    test_access.guard_origins(pd.date_range("2025-01-01", test_access.LAST_VALIDATION_ORIGIN))
    with pytest.raises(PermissionError):
        test_access.guard_origins(pd.date_range("2025-03-20", "2025-03-25"))
    test_access.guard_origins(pd.date_range("2025-03-20", "2025-04-30"), allow_test=True)
    p = tmp_path / "log.jsonl"
    test_access.record("baseline", "x", 10, path=p)
    test_access.record("final", "y", 12, path=p)
    assert [e["purpose"] for e in test_access.entries(p)] == ["baseline", "final"]


def test_recipe_tags_parse_and_label():
    from src.evaluate import parse_variant, recipe_options, recipe_tag
    from src.viz import MODEL_LABELS

    assert parse_variant("lstm_w@compact+nov26") == ("lstm", True, "compact")
    assert recipe_tag("lgbm@compact+nov26") == "nov26" and recipe_tag("lgbm@compact") is None
    assert recipe_options("lgbm@compact+nov26") == {"extra": ("interactions", "lag365"), "recency_halflife": 365}
    assert recipe_options("lstm_w@compact+nov26") == {"recency_halflife": 365}
    assert recipe_options("tft@v1") == {}
    assert MODEL_LABELS.get("lgbm@compact+nov26") == "LightGBM (compact weather, Nov 2026 recipe)"


def test_scorecard_tags_untagged_history_as_baseline(tmp_path):
    """Forecasts archived before recipes were tagged count as the baseline recipe."""
    from src import pipeline as pl

    d = pd.Timestamp("2026-11-03")
    hist = pd.DataFrame({"origin": [d - pd.Timedelta(days=1)] * 2, "date": [d] * 2, "target": ["energy"] * 2,
                         "h": [1, 1], "model": ["ensemble", "grid_india"], "p10": [4000.0, np.nan],
                         "p50": [4100.0, 4200.0], "p90": [4200.0, np.nan], "is_best": [True, False]})
    hist.to_parquet(tmp_path / pl.HISTORY_FILE, index=False)
    feat = pd.DataFrame({"date": [d], "energy": [4150.0], "solar_peak": [np.nan], "evening_peak": [np.nan]})
    feat.to_parquet(tmp_path / "features.parquet", index=False)
    new = pl.update_live_scorecard(tmp_path)
    assert new["recipe"].tolist() == ["baseline"]
