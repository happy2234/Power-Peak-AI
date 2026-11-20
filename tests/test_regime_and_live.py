"""Rare-regime sample weights and the live post-processing (ensemble, conformal, record status)."""
import numpy as np
import pandas as pd
import pytest

from src.models.regime_weights import fit_thresholds, flags, regime_weights


def test_regime_flags_and_weights():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"tmax_ai": rng.normal(33, 3, 1000), "heat_index_ai": rng.normal(36, 4, 1000),
                      "tmin_nr": rng.normal(15, 6, 1000), "fog_nr": np.zeros(1000),
                      "precip_ai": rng.exponential(3, 1000), "rain3_ai": rng.exponential(9, 1000),
                      "gust_max_ai": rng.normal(30, 5, 1000)})
    X.loc[0, "fog_nr"] = 0.9
    th = fit_thresholds(X)
    f = flags(X, th)
    assert f.loc[0, "cold_fog"]
    assert 0.08 < f["hot"].mean() < 0.25
    w = regime_weights(X, th, boost=1.0)
    assert w.mean() == pytest.approx(1.0)
    rare = f.any(axis=1).to_numpy()
    assert np.allclose(w[rare] / w[~rare].mean(), 2.0)
    assert (regime_weights(X, th, boost=0.0) == 1).all()


def test_regime_thresholds_come_from_training_rows_only():
    X = pd.DataFrame({"tmax_ai": np.arange(100.0)})
    th = fit_thresholds(X)
    X2 = pd.concat([X, pd.DataFrame({"tmax_ai": [1e6]})])
    assert fit_thresholds(X)[("hot", "tmax_ai")] == th[("hot", "tmax_ai")]
    assert flags(X2, th)["hot"].iloc[-1]


def test_live_postprocess_adds_ensemble_and_applies_conformal(monkeypatch):
    import src.postprocess as pp
    from src.predict import live_name, postprocess_live, recalibrate_record

    rows = []
    for m, p50 in (("lstm", 100.0), ("lgbm", 110.0), ("tft", 105.0)):
        for h in range(1, 8):
            rows.append({"model": m, "origin": pd.Timestamp("2026-11-08"), "date": pd.Timestamp("2026-11-08") + pd.Timedelta(days=h),
                         "h": h, "target": "energy", "p10": p50 - 5, "p50": p50, "p90": p50 + 5, "actual": np.nan})
    fc = pd.DataFrame(rows)
    params = {"ensemble_members": ["lstm_w", "lgbm"],
              "ensemble": {"energy": {h: 0.75 for h in range(1, 8)}},
              "conformal": {"lstm_w": {"energy": {h: 0.01 for h in range(1, 8)}},
                            "lstm": {"energy": {h: 0.5 for h in range(1, 8)}},   # the unchosen variant
                            "ensemble": {"energy": {h: -0.01 for h in range(1, 8)}}},
              "record": {"experimental": True, "skill": {"all_time": 0.01, "year": -0.02}, "threshold": 0.05,
                         "method": "platt", "platt": {"all_time": [1.0, -1.0], "year": None}}}
    monkeypatch.setattr(pp, "load_live", lambda: params)
    out = postprocess_live(fc)
    ens = out[out["model"] == "ensemble"]
    assert len(ens) == 7 and np.allclose(ens["p50"], 0.75 * 100 + 0.25 * 110)
    lstm = out[out["model"] == "lstm"]
    assert np.allclose(lstm["p90"] - lstm["p50"], 5 + 0.01 * 100)      # chosen variant's correction
    assert np.allclose(ens["p90"] - ens["p50"], 5 - 0.01 * 102.5)
    assert live_name("lstm_w") == "lstm" and live_name("ensemble") == "ensemble"
    rp, st = recalibrate_record({"p_all_time": 0.5, "p_year": 0.2})
    assert st["experimental"] and rp["p_all_time_raw"] == 0.5
    assert rp["p_all_time"] == pytest.approx(1 / (1 + np.exp(1.0)))   # logit(0.5)=0 -> sigmoid(-1)
    assert rp["p_year"] == 0.2                                          # no Platt fit -> unchanged
