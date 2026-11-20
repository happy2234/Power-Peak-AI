"""Ensemble weights and per-horizon conformal calibration: past-only and doing what they claim."""
import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON, TARGETS
from src.postprocess import (
    apply_conformal, apply_ensemble, fit_conformal, fit_ensemble, rolling_postprocess,
)


def fake_bt(seed=0, start="2025-05-01", end="2026-02-28", noise=None):
    """Two members: 'lstm' unbiased with small noise, 'lgbm' noisier. Raw intervals too narrow at
    long horizons and too wide at h=1, like the real backtest."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range(start, end)
    rows = []
    for t in TARGETS:
        level = 4500.0 if t == "energy" else 200_000.0
        truth = pd.Series(level * (1 + 0.03 * rng.standard_normal(len(origins) + 10)),
                          index=pd.date_range(start, periods=len(origins) + 10))
        for o in origins:
            for h in range(1, HORIZON + 1):
                d = o + pd.Timedelta(days=h)
                y = truth[d]
                sd = 0.01 * h
                for m, k in (("lstm", 1.0), ("lgbm", 1.6)):
                    p50 = y * (1 + k * sd * rng.standard_normal())
                    half = p50 * (0.03 if h == 1 else 0.006 * h)   # wide at h=1, narrow later
                    rows.append((m, o, d, h, t, p50 - half, p50, p50 + half, y))
    bt = pd.DataFrame(rows, columns=["model", "origin", "date", "h", "target", "p10", "p50", "p90", "actual"])
    return bt


def cover(d):
    return ((d["actual"] >= d["p10"]) & (d["actual"] <= d["p90"])).mean()


def test_ensemble_weight_prefers_the_better_member():
    bt = fake_bt()
    w = fit_ensemble(bt)
    assert all(w[t][h] >= 0.6 for t in TARGETS for h in range(1, HORIZON + 1))
    ens = apply_ensemble(bt, w)
    assert len(ens) == len(bt) / 2 and (ens["model"] == "ensemble").all()
    j = ens.merge(bt[bt.model == "lgbm"], on=["origin", "date", "h", "target"], suffixes=("", "_g"))
    assert (abs(j.p50 - j.actual) / j.actual).mean() < (abs(j.p50_g - j.actual) / j.actual).mean()


def test_conformal_brings_every_horizon_near_80_percent():
    bt = fake_bt()
    past, fut = bt[bt.date < "2025-12-01"], bt[bt.origin >= "2025-12-01"]
    cal = fit_conformal(past, ["lstm", "lgbm"])
    before = fut.groupby("h").apply(cover)
    after = apply_conformal(fut, cal).groupby("h").apply(cover)
    assert before.loc[1] > 0.9 and before.loc[7] < 0.6               # the problem it fixes
    assert ((after - 0.8).abs() < 0.06).all(), after
    assert (cal["lstm"]["energy"][1] < 0) and (cal["lstm"]["energy"][7] > 0)  # narrows h=1, widens h=7


def test_rolling_postprocess_uses_past_actuals_only():
    bt = fake_bt(end="2025-12-31")
    res, log = rolling_postprocess(bt)
    month = pd.Timestamp("2025-11-01")
    bt2 = bt.copy()
    bt2.loc[bt2["date"] >= month, "actual"] *= 1.5                    # corrupt everything not yet known
    res2, log2 = rolling_postprocess(bt2)
    a = log.set_index("month").loc[str(month.date())]
    b = log2.set_index("month").loc[str(month.date())]
    assert a["ensemble"] == b["ensemble"] and a["conformal"] == b["conformal"]
    cols = ["model", "origin", "h", "target", "p10", "p50", "p90"]
    k = res[(res.origin >= month) & (res.origin < "2025-12-01")][cols].reset_index(drop=True)
    k2 = res2[(res2.origin >= month) & (res2.origin < "2025-12-01")][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(k, k2)
    # sanity: the corruption does change later months
    assert log.set_index("month").loc["2025-12-01", "conformal"] != log2.set_index("month").loc["2025-12-01", "conformal"]


def test_first_month_without_history_is_left_raw():
    bt = fake_bt(end="2025-06-30")
    res, log = rolling_postprocess(bt)
    first = res[res.origin < "2025-06-01"]
    assert (first["conf_q"] == 0).all()
    assert (first[first.model == "ensemble"]["ens_w"] == 0.5).all()
    assert pytest.approx(first["p10"].to_numpy()) == first["p10_raw"].to_numpy()


def test_adaptive_level_moves_the_right_way():
    from src.postprocess import update_alphas

    ms = pd.Timestamp("2026-01-01")
    d = pd.date_range("2025-12-01", "2025-12-31")
    base = dict(model="m", target="energy", h=1, p50=100.0, date=d)
    over = pd.DataFrame({**base, "p10": 0.0, "p90": 200.0, "actual": 100.0})      # covers 100 %
    under = pd.DataFrame({**base, "p10": 99.0, "p90": 99.5, "actual": 100.0})    # covers 0 %
    a_over = update_alphas({}, over, ms)[("m", "energy", 1)]
    a_under = update_alphas({}, under, ms)[("m", "energy", 1)]
    assert a_over > 0.2 > a_under          # too wide -> larger alpha (narrower); too narrow -> wider
    # outcomes on or after the month start are not used
    late = over.assign(date=pd.date_range("2026-01-01", periods=31))
    assert update_alphas({}, late, ms) == {}
