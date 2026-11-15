#!/usr/bin/env python
"""Baseline vs final recipe on the fixed split's test period, from the two saved prediction sets.

    python scripts/accuracy_report.py      # -> reports/accuracy_round.md, section in fixed_split_results.md

Reads models/baseline_snapshot/processed/fixed_split_predictions.parquet (baseline, test access #1)
and data/processed/accuracy_round/final/test_predictions.parquet (final, test access #2). No model
is run here; both prediction sets are compared on exactly the same (origin, horizon, target) rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import fixed_split as fx  # noqa: E402
from src import test_access  # noqa: E402
from src.config import MODELS_DIR, REPORTS, TARGETS  # noqa: E402
from src.models.common import metrics  # noqa: E402

TGT = list(TARGETS)
KEYS = ["origin", "h", "target"]
BASE_NAMES = {"ensemble": "ensemble", "lgbm@compact": "lgbm", "lstm_w@compact": "lstm"}
START_MARK = "<!-- accuracy-round:start -->"
END_MARK = "<!-- accuracy-round:end -->"


def load():
    from src.evaluate import weather_regimes

    b = pd.read_parquet(MODELS_DIR / "baseline_snapshot/processed/fixed_split_predictions.parquet")
    b = b[b["model"].isin(BASE_NAMES)].assign(model=lambda x: x["model"].map(BASE_NAMES))
    f = pd.read_parquet(ROOT / "data/processed/accuracy_round/final/test_predictions.parquet")
    cols = KEYS + ["model", "date", "p10", "p50", "p90", "actual"]
    b, f = b[cols].dropna(subset=["actual"]), f[cols].dropna(subset=["actual"])
    b = b[b["date"] >= pd.Timestamp(fx.TEST_START)]
    common = b[b["model"] == "ensemble"][KEYS].merge(f[f["model"] == "ensemble"][KEYS])
    b, f = b.merge(common), f.merge(common)
    daily, _ = fx.features()
    reg = weather_regimes(daily)
    for x in (b, f):
        x["fy"] = x["date"].map(fx.fy)
        x["regime"] = reg.reindex(x["date"]).to_numpy()
    return b, f


def table(b, f, key=None, model="ensemble") -> pd.DataFrame:
    rows = []
    gb = ["target"] + ([key] if key else [])
    for name, x in (("baseline", b), ("final", f)):
        for k, g in x[x["model"] == model].groupby(gb):
            k = k if isinstance(k, tuple) else (k,)
            m = metrics(g)
            rows.append({**dict(zip(gb, k)), "recipe": name, "MAPE": m["MAPE"], "coverage": m.get("coverage80"),
                         "bias": m["bias"], "days": g["date"].nunique()})
    t = pd.DataFrame(rows).pivot_table(index=gb, columns="recipe", values=["MAPE", "coverage"])
    out = pd.DataFrame({"baseline MAPE %": t[("MAPE", "baseline")], "final MAPE %": t[("MAPE", "final")]})
    out["change"] = out["final MAPE %"] - out["baseline MAPE %"]
    out["baseline coverage %"] = t[("coverage", "baseline")]
    out["final coverage %"] = t[("coverage", "final")]
    return out


def overall(b, f, model="ensemble") -> dict:
    t = table(b, f, model=model)
    return {"baseline": float(t["baseline MAPE %"].mean()), "final": float(t["final MAPE %"].mean()),
            "baseline_cov": float(t["baseline coverage %"].mean()), "final_cov": float(t["final coverage %"].mean())}


def paired_bootstrap(b, f, n=2000, seed=0) -> tuple[float, float]:
    """95 % interval of the change in ensemble MAPE (mean of targets), resampling whole weeks of
    target dates so that the day-to-day correlation of errors is kept."""
    rng = np.random.default_rng(seed)
    e = b[b.model == "ensemble"].merge(f[f.model == "ensemble"], on=KEYS + ["date"], suffixes=("_b", "_f"))
    e["d"] = (e["p50_f"] - e["actual_f"]).abs() / e["actual_f"] - (e["p50_b"] - e["actual_b"]).abs() / e["actual_b"]
    e["wk"] = e["date"].dt.to_period("W")
    wk = e.groupby(["wk", "target"])["d"].agg(["sum", "count"]).reset_index()
    weeks = wk["wk"].unique()
    stats = []
    for _ in range(n):
        pick = pd.Series(rng.choice(weeks, len(weeks)))
        s = wk.merge(pick.rename("wk").to_frame(), on="wk")
        g = s.groupby("target")[["sum", "count"]].sum()
        stats.append(100 * (g["sum"] / g["count"]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def rolling() -> dict:
    """Rolling backtest (monthly refits, reported window) of the ensemble built from the baseline
    members vs the final-recipe members, both post-processed the same way from the saved raw
    backtest predictions; plus the head-to-head with Grid-India."""
    from src.evaluate import BACKTEST_START, head_to_head
    from src.postprocess import rolling_postprocess

    raw = pd.read_parquet(ROOT / "data/processed/backtest_predictions.parquet")
    out = {}
    for name, members in (("baseline", ("lstm_w@compact", "lgbm@compact")),
                          ("final", ("lstm_w@compact+nov26", "lgbm@compact+nov26"))):
        proc, _ = rolling_postprocess(raw[raw["model"].isin(members)], members)
        proc = pd.concat([proc, raw[raw["model"] == "grid_india"]], ignore_index=True)
        ev = proc[proc["origin"] >= pd.Timestamp(BACKTEST_START)].dropna(subset=["actual"])
        res = {}
        for m, key in ((members[0], "lstm"), (members[1], "lgbm"), ("ensemble", "ensemble")):
            g = ev[ev["model"] == m]
            res[key] = {t: metrics(g[g["target"] == t]) for t in TGT}
        h = head_to_head(ev, "ensemble")
        res["gi"] = {t: {"win": float(x["ours_wins"].mean()), "imp": float(1 - x["ae_ours"].mean() / x["ae_gi"].mean())}
                     for t, x in h.groupby("target")}
        out[name] = res
    return out


def main():
    b, f = load()
    st = json.loads((ROOT / "data/processed/accuracy_round/state.json").read_text())
    hist = st["history"]
    base_val = hist[0]["before"]["ensemble"]
    final_val = hist[-1]["before"]["ensemble"]          # the final incumbent, as scored in the last experiment
    o = overall(b, f)
    lo, hi = paired_bootstrap(b, f)
    access = test_access.entries()
    wins_val = final_val["MAPE"] < base_val["MAPE"]
    wins_test = o["final"] < o["baseline"]
    cov_ok = o["final_cov"] >= o["baseline_cov"] - 2
    fixed_rule = wins_val and wins_test and cov_ok
    members = {m: overall(b, f, m) for m in ("lstm", "lgbm")}
    roll = rolling()
    rmean = {k: {m: float(np.mean([roll[k][m][t]["MAPE"] for t in TGT])) for m in ("ensemble", "lstm", "lgbm")}
             for k in roll}
    rcov = {k: float(np.mean([roll[k]["ensemble"][t]["coverage80"] for t in TGT])) for k in roll}
    wins_roll = rmean["final"]["ensemble"] < rmean["baseline"]["ensemble"]
    deploy = fixed_rule and wins_roll

    L = [START_MARK, "## Before/after the accuracy round (Nov 2026)", "",
         f"Final recipe: **{st['incumbent']['name']}** — LightGBM gains temperature × weekend and CDD × season "
         "interactions and festival-aligned same-day-last-year features; LightGBM and LSTM weight training samples "
         "with a 365-day half-life. Everything else as the baseline. Chosen on the validation months only "
         "(reports/experiments.md); trained once on the same fit origins with the same seed as the baseline.", "",
         "| | baseline | final | change |", "|---|---|---|---|",
         f"| Validation MAPE % (Jan–Mar 2025, OOF ensemble, 3 seeds) | {base_val['MAPE']:.3f} | {final_val['MAPE']:.3f} | "
         f"{final_val['MAPE'] - base_val['MAPE']:+.3f} |",
         f"| Validation P10–P90 coverage % | {base_val['coverage']:.1f} | {final_val['coverage']:.1f} | "
         f"{final_val['coverage'] - base_val['coverage']:+.1f} |",
         f"| **Test MAPE %** (Apr 2025 → Nov 2026, ensemble, mean of targets) | {o['baseline']:.3f} | {o['final']:.3f} | "
         f"{o['final'] - o['baseline']:+.3f} (95 % interval {lo:+.3f} to {hi:+.3f}) |",
         f"| Test P10–P90 coverage % | {o['baseline_cov']:.1f} | {o['final_cov']:.1f} | "
         f"{o['final_cov'] - o['baseline_cov']:+.1f} |",
         f"| Test MAPE %, LSTM alone | {members['lstm']['baseline']:.3f} | {members['lstm']['final']:.3f} | "
         f"{members['lstm']['final'] - members['lstm']['baseline']:+.3f} |",
         f"| Test MAPE %, LightGBM alone | {members['lgbm']['baseline']:.3f} | {members['lgbm']['final']:.3f} | "
         f"{members['lgbm']['final'] - members['lgbm']['baseline']:+.3f} |", "",
         "The test interval is a paired block bootstrap over weeks of target dates (2,000 resamples).", ""]
    L += ["### Rolling backtest with the final recipe (monthly refits, as production is operated)", "",
          "Same 12 reported months as reports/results.md (forecasts issued from 1 Nov 2025), every model refit "
          "monthly on all data known before the month, ensemble weights and calibration fitted on past months only. "
          "This window lies inside the fixed split's test period; it was run after the final measurement and used "
          "only for the deployment decision, never to change the recipe.", "",
          "| MAPE % (mean of targets) | baseline | final recipe | change |", "|---|---|---|---|"]
    for m, lab in (("ensemble", "Ensemble"), ("lstm", "LSTM"), ("lgbm", "LightGBM")):
        L.append(f"| {lab} | {rmean['baseline'][m]:.3f} | {rmean['final'][m]:.3f} | "
                 f"{rmean['final'][m] - rmean['baseline'][m]:+.3f} |")
    L.append(f"| Ensemble P10–P90 coverage % | {rcov['baseline']:.1f} | {rcov['final']:.1f} | "
             f"{rcov['final'] - rcov['baseline']:+.1f} |")
    L += ["", "Ensemble vs Grid-India (mean of our 1–7-day forecasts per day):", "",
          "| target | baseline: our error lower by | final: our error lower by | baseline win rate | final win rate |",
          "|---|---|---|---|---|"]
    for t in TGT:
        gb, gf = roll["baseline"]["gi"][t], roll["final"]["gi"][t]
        L.append(f"| {t} | {gb['imp']:.1%} | {gf['imp']:.1%} | {gb['win']:.0%} | {gf['win']:.0%} |")
    L.append("")
    if deploy:
        L += ["**Decision: the final recipe beats the baseline on validation, on the test period and in the rolling "
              "backtest, so it replaces the baseline in production.**", ""]
    elif fixed_rule and not wins_roll:
        L += ["**Decision: the baseline stays in production.** The final recipe meets the round's rule (better on "
              "validation and on the test period, coverage not worse), but with monthly refits, which is how the "
              f"production models are operated, it is worse: ensemble {rmean['baseline']['ensemble']:.3f} → "
              f"{rmean['final']['ensemble']:.3f} % in the rolling backtest, with both members worse. Its gains hold "
              "for a model frozen for 19 months, not for one retrained every month. Deploying it would most likely "
              "make the live forecasts worse, so nothing was switched; to deploy it anyway: "
              "`python scripts/backtest.py --reuse --force lgbm=lgbm@compact+nov26 lstm=lstm_w@compact+nov26 && "
              "python scripts/train.py --model all && python scripts/predict.py`.", ""]
    else:
        why = [w for w, ok in (("validation", wins_val), ("test", wins_test), ("coverage", cov_ok)) if not ok]
        L += [f"**Decision: the final recipe does not beat the baseline on {' and '.join(why)}, so the baseline stays "
              "in production.**", ""]
    L += ["### By target — ensemble", "", table(b, f).round(2).to_markdown(), "",
          "### By horizon (days ahead) — ensemble", "", table(b, f, "h").round(2).to_markdown(), "",
          "### By financial year — ensemble", "", table(b, f, "fy").round(2).to_markdown(), "",
          "### By weather regime (observed weather) — ensemble", "", table(b, f, "regime").round(2).to_markdown(), "",
          "### Test-period access", "",
          "The test period was evaluated exactly twice during this round:", ""]
    for e in access:
        L.append(f"- {e['at']}: **{e['purpose']}** — {e['what']} ({e['rows']:,} rows)")
    L += ["", f"Every experiment ran on validation origins only; the harness refuses origins after "
          f"{test_access.LAST_VALIDATION_ORIGIN} (src/test_access.py). The baseline's test numbers were read once "
          "when the baseline was snapshotted, before any experiment; the experiment list was fixed in advance, so "
          "they did not steer the choices.", END_MARK, ""]
    section = "\n".join(L)

    # ---- summary of every experiment (validation) with the final-only test numbers
    S = ["## Summary of all experiments", "",
         "Validation = OOF ensemble MAPE % on Jan–Mar 2025, mean of 3 seeds (reports/experiments.md). Test numbers "
         "exist only for the baseline and the final recipe: the test period was evaluated twice, never per experiment.",
         "", "| # | experiment | validation before | validation after (best variant) | kept | test |",
         "|---|---|---|---|---|---|",
         f"| – | baseline (production) | {base_val['MAPE']:.3f} | – | – | {o['baseline']:.3f} |"]
    for e in hist:
        a_ = next(c["result"]["ensemble"]["MAPE"] for c in e["candidates"] if c["name"] == e["best"])
        S.append(f"| {e['id']} | {e['title']} | {e['before']['ensemble']['MAPE']:.3f} | {a_:.3f} ({e['best']}) | "
                 f"{'yes' if e['kept'] else 'no'} | – |")
    S.append(f"| final | {st['incumbent']['name']} | {base_val['MAPE']:.3f} | {final_val['MAPE']:.3f} | – | "
             f"{o['final']:.3f} (rolling, refit monthly: {rmean['final']['ensemble']:.3f} vs "
             f"{rmean['baseline']['ensemble']:.3f}) |")
    kept = {e["id"]: e["gain"] for e in hist if e["kept"]}
    S += ["", "### What moved the needle", "",
          f"On validation, four changes each helped a little and add up to {base_val['MAPE'] - final_val['MAPE']:.3f} "
          f"points: recency weighting for LightGBM ({kept.get('c1', 0):+.3f}) and the LSTM ({kept.get('c2', 0):+.3f}, "
          "carried by one of three seeds), temperature × weekend and CDD × season interactions "
          f"({kept.get('a1', 0):+.3f}), and the festival-aligned same-day-last-year features ({kept.get('a3', 0):+.3f}, "
          "the only feature change that helped on every seed, in line with festivals being the largest error source). "
          "Each gain is about the size of the seed-to-seed spread (0.04), and choosing the half-lives on the same "
          "three winter months flatters them. Everything else did nothing or hurt: a different label base, growth "
          "features, a longer or shorter LSTM window, more or less LSTM capacity, TFT as a third member (−0.07) and "
          "the residual corrector (−0.18, overfits ~290 rows per fold). On the test period, with models trained once, "
          f"the recipe gained {o['baseline'] - o['final']:.3f} points, almost all of it through the LSTM "
          f"({members['lstm']['baseline'] - members['lstm']['final']:+.3f} alone vs "
          f"{members['lgbm']['baseline'] - members['lgbm']['final']:+.3f} for LightGBM), mostly on energy and the "
          "evening peak. But retrained monthly, as production runs, the same recipe loses "
          f"({rmean['baseline']['ensemble']:.3f} → {rmean['final']['ensemble']:.3f} %), LightGBM most. The honest "
          "reading: none of the planned changes is a robust improvement for the system as it is operated; the "
          "baseline stays. The next round should validate on rolling refits inside the training period rather than "
          "on one frozen three-month window.", ""]
    section_round = section.replace(START_MARK, "").replace(END_MARK, "") + "\n" + "\n".join(S)
    (REPORTS / "accuracy_round.md").write_text("# Accuracy round: final measurement\n\n" + section_round)
    fs = REPORTS / "fixed_split_results.md"
    text = fs.read_text()
    if START_MARK in text:
        text = text[:text.index(START_MARK)] + text[text.index(END_MARK) + len(END_MARK):].lstrip("\n")
    fs.write_text(text.rstrip("\n") + "\n\n" + section)
    (ROOT / "data/processed/accuracy_round/decision.json").write_text(json.dumps(
        {"deploy": deploy, "fixed_split_rule_met": fixed_rule, "rolling": rmean,
         "validation": {"baseline": base_val["MAPE"], "final": final_val["MAPE"]},
         "test": o, "test_change_95": [lo, hi], "members": members}, indent=1))
    print(section)


if __name__ == "__main__":
    main()
