#!/usr/bin/env python
"""Rolling-origin backtest of all models vs Grid-India, then post-processing and reports.

    python scripts/backtest.py                       # all models: 6 warm-up + 12 reported months
    python scripts/backtest.py --models lgbm lgbm_w  # re-run some models, keep the others' saved rows
    python scripts/backtest.py --reuse               # recompute reports from saved predictions

Steps: raw backtest -> choose the regime-weighted or plain variant of LightGBM / LSTM (lower
overall MAPE, plain on a tie) -> ensemble + per-horizon conformal calibration fitted on past rows
only -> best model (lowest overall MAPE) -> record-probability recalibration -> reports/results.md.
Outputs in data/processed/: backtest_predictions.parquet (raw), backtest_processed.parquet,
best_model.txt, model_selection.json, postprocess.json, record_calibration.parquet, mc_corr.npy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED  # noqa: E402
from src.utils.device import describe_device  # noqa: E402

log = logging.getLogger("backtest")
ALL_MODELS = ["seasonal_naive", "lgbm", "lgbm_w", "lstm", "lstm_w", "tft"]


def overall_mape(ev: pd.DataFrame) -> pd.Series:
    from src.evaluate import leaderboard

    lb = leaderboard(ev)
    return lb.pivot_table(index="model", columns="target", values="MAPE").mean(axis=1)


def main():
    from src.evaluate import BACKTEST_START, WARMUP_START, backtest, eval_window
    from src.features.build import load_daily
    from src.insights import (RECORD_SKILL_MIN, brier, record_calibration, recalibration_live,
                              residual_correlation, rolling_recalibration)
    from src.postprocess import fit_live, rolling_postprocess
    from src.report import write_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--start", default=WARMUP_START)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--force", nargs="+", default=[], metavar="FAMILY=VARIANT",
                    help="pin the variant of a family instead of choosing the lowest backtest MAPE, e.g. "
                         "lgbm=lgbm@compact+nov26 (used for recipes chosen on the fixed split's validation months)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log.info("device: %s", describe_device())

    t0 = time.time()
    raw = pd.read_parquet(PROCESSED / "backtest_predictions.parquet") if args.reuse else \
        backtest(tuple(args.models), start=args.start)
    log.info("raw backtest ready in %.1f min", (time.time() - t0) / 60)
    daily, _ = load_daily()

    # ---- 1. variant selection per family (feature set x regime weighting): lowest overall MAPE;
    #      on a tie the simpler variant (fewer features, no weighting) wins
    from src.evaluate import parse_variant, recipe_tag

    om = overall_mape(eval_window(raw))
    rank_fs = {"v1": 0, "compact": 1, "full": 2}
    chosen, decisions = {}, []
    for fam in ("lgbm", "lstm", "tft"):
        cands = [m for m in om.index if parse_variant(m)[0] == fam]
        if not cands:
            continue
        key = lambda m: (round(om[m], 6), rank_fs[parse_variant(m)[2]], parse_variant(m)[1])  # noqa: E731
        forced = dict(f.split("=", 1) for f in args.force)
        chosen[fam] = forced[fam] if fam in forced else min(cands, key=key)
        for m in sorted(cands, key=key):
            _, w, fs = parse_variant(m)
            decisions.append({"family": fam, "variant": m, "feature_set": fs, "regime_weights": w,
                              "overall_MAPE": float(om[m]), "chosen": m == chosen[fam]})
    members = (chosen["lstm"], chosen["lgbm"])    # ---- 2. ensemble + conformal, month by month, past rows only
    processed, plog = rolling_postprocess(raw, members)
    live_alphas = processed.attrs.get("alphas", {})
    # for the report: the same calibration without the adaptive step (plain split-conformal)
    import src.postprocess as pp

    g0, pp.ACI_GAMMA = pp.ACI_GAMMA, 0.0
    plain, _ = rolling_postprocess(raw, members)
    pp.ACI_GAMMA = g0
    gi = raw[raw["model"] == "grid_india"]
    processed = pd.concat([processed, gi], ignore_index=True)
    processed.to_parquet(PROCESSED / "backtest_processed.parquet", index=False)
    ev = eval_window(processed)

    # ---- 3. best model = lowest overall MAPE among the final candidates
    cands = [chosen[f] for f in ("lgbm", "lstm", "tft") if f in chosen] + ["ensemble"]
    om2 = overall_mape(ev)
    best = min((m for m in cands if m in om2.index), key=lambda m: om2[m])
    (PROCESSED / "best_model.txt").write_text(best)
    selection = {"variants": decisions, "chosen": chosen, "ensemble_members": list(members), "candidates": cands,
                 "overall_mape": {m: float(om2[m]) for m in om2.index}, "best": best,
                 "regime_boost": {f: (1.0 if parse_variant(m)[1] else 0.0) for f, m in chosen.items()},
                 "feature_set": {f: parse_variant(m)[2] for f, m in chosen.items()},
                 "recipe": {f: recipe_tag(m) for f, m in chosen.items()},
                 "forced": dict(f.split("=", 1) for f in args.force)}
    (PROCESSED / "model_selection.json").write_text(json.dumps(selection, indent=1, default=float))
    log.info("best model: %s; variants %s", best, chosen)

    # ---- 4. record probability: raw vs recalibrated (rolling, past origins only)
    rows_best = processed[processed["model"] == best]
    corr = residual_correlation(rows_best[rows_best["origin"] < pd.Timestamp(BACKTEST_START)]
                                if (rows_best["origin"] < pd.Timestamp(BACKTEST_START)).sum() > 500 else rows_best, best)
    np.save(PROCESSED / "mc_corr.npy", corr)
    cal = record_calibration(rows_best, daily, best, corr=corr)
    for kind in ("all_time", "year"):
        cal = rolling_recalibration(cal, kind)
    cal.to_parquet(PROCESSED / "record_calibration.parquet", index=False)
    cal_e = cal[cal["origin"] >= pd.Timestamp(BACKTEST_START)]
    rec = {}
    for kind in ("all_time", "year"):
        rec[kind] = {m: brier(cal_e[f"p_{kind}{sfx}"], cal_e[f"hit_{kind}"])
                     for m, sfx in (("raw", ""), ("platt", "_platt"), ("isotonic", "_iso"))}
    skill = min(rec[k]["platt"]["skill"] for k in rec)
    status = {"method": "platt", "skill": {k: rec[k]["platt"]["skill"] for k in rec},
              "experimental": bool(not skill > RECORD_SKILL_MIN), "threshold": RECORD_SKILL_MIN,
              "events": {k: rec[k]["platt"]["events"] for k in rec}, "origins": int(len(cal_e)),
              "platt": recalibration_live(cal)}
    fit_live(raw, processed[processed["model"] != "grid_india"], members,
             extra={"record": status, "best_model": best}, alphas=live_alphas)

    # ---- 5. report
    timing_p = PROCESSED / "backtest_timing.json"
    timing = json.loads(timing_p.read_text()) if timing_p.exists() else {}
    baseline = PROCESSED / "baseline_v1" / "backtest_predictions.parquet"
    v1 = pd.read_parquet(baseline) if baseline.exists() else None
    text = write_results(raw, processed, daily, best, selection, plog, rec, status, timing, v1,
                         compute=describe_device(), plain=eval_window(plain))
    print(text[:3000])
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
