"""src/walkthrough.py: the data layer of the "Raw to processed" page, on the 01.11.26 fixture and on small
synthetic processed files (so the tests do not depend on the local data/ folder)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import walkthrough as w
from src.config import HORIZON, QUANTILES, TARGETS
from src.parse import psp_parser as P

FIX = Path(__file__).parent / "fixtures" / "01.11.26_NLDC_PSP_491.xls"


def test_col_letter_and_cell_ref():
    assert [w.col_letter(i) for i in (0, 7, 25, 26, 27)] == ["A", "H", "Z", "AA", "AB"]
    assert w.cell_ref(8, 7) == "H9"


def test_raw_overview():
    o = w.raw_overview(FIX)
    assert o["name"] == FIX.name and o["size_bytes"] == FIX.stat().st_size
    assert str(o["data_date"]) == "2026-11-01" and str(o["report_date"]) == "2026-11-02"
    assert [s["sheet"] for s in o["sheets"]] == ["MOP_E", "MOP_E1", "IR-Line", "CrossBorder", "TimeSeries"]
    assert o["sheets"][0]["rows"] > 90 and o["sheets"][0]["columns"] == 9


def test_target_cells_find_the_three_values_by_label():
    cells = {c["target"]: c for c in w.target_cells(FIX)}
    assert list(cells) == list(TARGETS)
    assert cells["energy"]["value"] == 4613.82 and cells["energy"]["cell"] == "H9"
    assert cells["solar_peak"]["value"] == 215463 and cells["solar_peak"]["label"] == "Solar hr"
    assert cells["evening_peak"]["value"] == 203172 and cells["evening_peak"]["label"] == "Non-Solar hr"
    assert all(c["sheet"] == "MOP_E" for c in cells.values())
    assert {t: c["column"] for t, c in cells.items()} == TARGETS


def test_target_cells_agree_with_the_parser_and_the_cell_really_holds_the_value():
    row = P.parse_daily_summary(FIX).iloc[0]
    raw = P.find_sheet(P.read_sheets(FIX), "MOP_E")
    for c in w.target_cells(FIX):
        assert row[c["column"]] == c["value"]
        assert P.to_num(raw.iat[c["row"], c["col"]]) == c["value"]


def test_mop_e_sections_flag_the_target_cells():
    secs = w.mop_e_sections(FIX)
    assert [s["title"][:2] for s in secs] == ["A.", "I."]
    flagged = [(r["excel_row"], col, t) for s in secs for r in s["rows"] for col, t in r["targets"].items()]
    assert sorted(t for *_, t in flagged) == sorted(TARGETS)
    energy = next(r for r in secs[0]["rows"] if r["targets"])
    assert energy["cells"][0].startswith("Energy Met") and energy["cells"][7] == "4613.82"


# ---------------------------------------------------------------------------- processed files (synthetic)
DAY = "2026-11-01"


@pytest.fixture
def proc(tmp_path, monkeypatch):
    pdir, mdir, raw = tmp_path / "processed", tmp_path / "models", tmp_path / "raw"
    pdir.mkdir(); (mdir / "lstm" / "r1").mkdir(parents=True); (mdir / "lgbm" / "r2").mkdir(parents=True); raw.mkdir()
    for mod_attr in ("PROCESSED", "MODELS_DIR", "RAW_LATEST"):
        monkeypatch.setattr(w, mod_attr, {"PROCESSED": pdir, "MODELS_DIR": mdir, "RAW_LATEST": raw}[mod_attr])
    dates = pd.date_range("2026-10-01", periods=40)
    parsed = P.parse_daily_summary(FIX)
    parsed.to_parquet(pdir / "daily_summary.parquet")
    clean = pd.DataFrame({"date": dates, "solar_peak": 1.0, "evening_peak": 2.0, "energy": 3.0,
                          "is_imputed": [d == pd.Timestamp("2026-10-05") for d in dates], "is_outlier": False,
                          "is_missing_report": False})
    clean.to_parquet(pdir / "daily_clean.parquet")
    feat = pd.DataFrame({"date": dates, "tmax_ai": np.linspace(20, 30, 40), "dow_sin": 0.5, "dow": 1,
                         "solar_peak": 1.0, "solar_peak_lag7": 1.0, "cdd_ai": 4.0})
    feat.to_parquet(pdir / "features.parquet")
    (pdir / "region_weights.json").write_text(json.dumps({"nr": 0.3, "wr": 0.3, "sr": 0.2, "er": 0.15, "ner": 0.05}))
    (mdir / "lstm" / "latest.txt").write_text("r1")
    enc = ["solar_peak_r", "tmax_ai", "dow_sin", "cdd_ai"]
    (mdir / "lstm" / "r1" / "config.yaml").write_text(yaml.safe_dump(
        {"enc_features": enc, "dec_features": ["tmax_ai", "solar_peak_lag7"], "enc_days": 60, "horizon": 7,
         "quantiles": [0.1, 0.5, 0.9], "train_end": "2026-10-20"}))
    np.savez(mdir / "lstm" / "r1" / "scaler.npz", enc_mu=np.array([0, 20.0, 0, 4.0]), enc_sd=np.array([1, 4.0, 0.25, 1e-6]),
             dec_mu=np.zeros(2), dec_sd=np.ones(2))
    (mdir / "lgbm" / "latest.txt").write_text("r2")
    (mdir / "lgbm" / "r2" / "config.yaml").write_text(yaml.safe_dump({"features": 87, "train_end": "2026-10-20"}))
    return pdir, mdir, raw


def test_latest_raw_file_is_the_newest_by_date_in_the_name(proc):
    _, _, raw = proc
    assert w.latest_raw_file() is None
    for n in ("13.11.26_NLDC_PSP_1.xls", "14.11.26_NLDC_PSP_736.xls", "31.12.25_NLDC_PSP_9.xls", "notes.txt"):
        (raw / n).write_bytes(b"x")
    assert w.latest_raw_file().name == "14.11.26_NLDC_PSP_736.xls"


def test_every_loader_returns_none_when_its_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "PROCESSED", tmp_path)
    monkeypatch.setattr(w, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(w, "RAW_LATEST", tmp_path / "nope")
    assert w.latest_raw_file() is None
    for f in (w.daily_summary, w.daily_clean, w.city_weather, w.region_weights, w.features, w.latest_clean_date,
              w.clean_counts, lambda: w.daily_summary_row(DAY), lambda: w.daily_clean_row(DAY),
              lambda: w.features_row(DAY), lambda: w.city_weather_table(DAY), lambda: w.model_config("lstm"),
              lambda: w.model_scaler("lstm"), lambda: w.scaled_examples(DAY), lambda: w.feature_groups(DAY),
              lambda: w.weather_check(DAY), lambda: w.recent_clean(DAY), lambda: w.compare_with_stored(FIX)):
        assert f() is None
    assert w.model_summary() == []


def test_rows_counts_and_recent(proc):
    assert w.daily_summary_row(DAY)["energy_met_mu_all_india"] == 4613.82
    assert w.daily_clean_row("2026-10-05")["is_imputed"]
    assert w.daily_clean_row("2030-01-01") is None
    assert w.clean_counts() | {} == {"days": 40, "imputed": 1, "outliers": 0, "missing_reports": 0,
                                     "first": pd.Timestamp("2026-10-01"), "last": pd.Timestamp("2026-11-09")}
    r = w.recent_clean("2026-10-20", days=10)
    assert len(r) == 10 and r["date"].max() == pd.Timestamp("2026-10-20")
    assert w.latest_clean_date() == pd.Timestamp("2026-11-09")


def test_compare_with_stored_shows_match_and_mismatch(proc):
    pdir, _, _ = proc
    c = w.compare_with_stored(FIX)
    assert c["match"] and c["n_match"] == c["n_columns"] > 50
    s = pd.read_parquet(pdir / "daily_summary.parquet")
    s.loc[0, "energy_met_mu_all_india"] += 1
    s.to_parquet(pdir / "daily_summary.parquet")
    c = w.compare_with_stored(FIX)
    assert not c["match"] and [r[0] for r in c["rows"] if not r[3]] == ["energy_met_mu_all_india"]


def test_scaled_examples_use_the_scaler_and_clip(proc):
    ex = w.scaled_examples(DAY).set_index("feature")
    assert "solar_peak_r" not in ex.index                       # ratios are built with the sequences
    assert ex.loc["tmax_ai", "scaled"] == pytest.approx((ex.loc["tmax_ai", "raw"] - 20) / 4)
    assert ex.loc["dow_sin", "scaled"] == pytest.approx(2.0)
    assert ex.loc["cdd_ai", "scaled"] == 0.0 and not ex.loc["cdd_ai", "clipped"]
    f = pd.read_parquet(proc[0] / "features.parquet")
    f["cdd_ai"] = 9.0                                           # (9 - 4) / 1e-6 -> clipped to 10
    f.to_parquet(proc[0] / "features.parquet")
    ex = w.scaled_examples(DAY).set_index("feature")
    assert ex.loc["cdd_ai", "scaled"] == 10.0 and ex.loc["cdd_ai", "clipped"]


def test_model_summary_window_and_groups(proc):
    s = {m["model"]: m for m in w.model_summary()}
    assert (s["lstm"]["encoder"], s["lstm"]["decoder"], s["lstm"]["distinct"]) == (4, 2, 5)
    assert s["lgbm"]["distinct"] == 87 and s["lgbm"]["encoder"] is None
    win = w.window_info()
    assert (win["encoder_days"], win["horizon"], win["train_end"]) == (60, HORIZON, "2026-10-20")
    assert win["outputs"] == len(TARGETS) * HORIZON * len(QUANTILES) == 63
    g = w.feature_groups(DAY)
    assert list(g) == list(w.GROUPS)
    assert g["Weather"]["feature"].tolist() == ["tmax_ai", "cdd_ai"]
    assert "dow_sin" in g["Calendar"]["feature"].tolist() and g["Lag"]["feature"].tolist() == ["solar_peak_lag7"]
    assert "solar_peak_r" in w.sequence_only_features("lstm")


def test_city_weather_and_weighting(proc):
    from src.data.weather import CITIES
    pdir, _, _ = proc
    row = {"date": pd.Timestamp(DAY)} | {f"{v}_{c}": float(i) for i, c in enumerate(CITIES)
                                         for v in ("tmax", "tmean", "app_tmax", "rh_mean", "precip", "rad")}
    pd.DataFrame([row]).to_parquet(pdir / "weather_city.parquet")
    t = w.city_weather_table(DAY)
    assert len(t) == 11 and set(t["region"]) == {"NR", "WR", "SR", "ER", "NER"}
    rw = w.region_weights()
    reg = {r: np.mean([i for i, (c, v) in enumerate(CITIES.items()) if v[2] == r]) for r in rw}
    ai = sum(rw[r] * reg[r] for r in rw)
    f = pd.read_parquet(pdir / "features.parquet")
    f["tmax_ai"] = ai
    f.to_parquet(pdir / "features.parquet")
    chk = w.weather_check(DAY)
    assert chk["match"] and chk["rebuilt"] == pytest.approx(ai)


def test_file_chain_lists_each_file_with_size_and_rows(proc):
    _, _, raw = proc
    (raw / "01.11.26_NLDC_PSP_491.xls").write_bytes(FIX.read_bytes())
    chain = {c["file"]: c for c in w.file_chain()}
    assert chain["01.11.26_NLDC_PSP_491.xls"]["detail"] == "5 sheets"
    assert chain["daily_clean.parquet"]["rows"] == 40 and chain["daily_clean.parquet"]["size_kb"] > 0
    assert chain["features.parquet"]["columns"] == 7
    assert chain["latest_forecast.parquet"]["exists"] is False
