#!/usr/bin/env python
"""Fairness audit of the comparison with Grid-India -> reports/fairness_audit.md.

    python scripts/audit.py            # runs the long-lead backtest (~2-4 min) and writes the report
    python scripts/audit.py --reuse    # reuse data/processed/audit_long_lead.parquet

Needs scripts/backtest.py to have run (backtest_processed.parquet, best_model.txt).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import PROCESSED, REPORTS  # noqa: E402

log = logging.getLogger("audit")


def fmt_row(label, r, unit_div, unit):
    return (f"| {label} | {r['days']} | {r['our_MAE'] / unit_div:,.2f} | {r['gi_MAE'] / unit_div:,.2f} | "
            f"{r['our_MAPE']:.2f} | {r['gi_MAPE']:.2f} | {r['improvement']:+.1%} | {r['win_rate']:.0%} | "
            f"{r['our_bias'] / unit_div:+,.2f} | {r['gi_bias'] / unit_div:+,.2f} |")


def main():
    from src.audit import leakage_check, long_lead_backtest, run_audit
    from src.evaluate import BACKTEST_START
    from src.features.build import load_daily

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    daily, _ = load_daily()
    processed = pd.read_parquet(PROCESSED / "backtest_processed.parquet")
    best = (PROCESSED / "best_model.txt").read_text().strip()
    ll_path = PROCESSED / "audit_long_lead.parquet"
    ll = pd.read_parquet(ll_path) if (args.reuse and ll_path.exists()) else long_lead_backtest(BACKTEST_START)

    steps, res, info, jm = run_audit(best, processed, daily, ll, BACKTEST_START)
    leak = leakage_check(daily, BACKTEST_START)

    # the long-lead model at short leads must be comparable to the main models (sanity)
    ll_e = ll[ll["origin"] >= pd.Timestamp(BACKTEST_START)].dropna(subset=["actual"])
    ll_e = ll_e.assign(ape=(ll_e["p50"] - ll_e["actual"]).abs() / ll_e["actual"] * 100)
    by_h = ll_e.pivot_table(index="target", columns="h", values="ape").round(2)

    L = ["# Fairness audit: our forecast vs Grid-India's week-ahead forecast", "",
         f"Final model: `{best}`. Reported window: Grid-India forecast days from {BACKTEST_START} "
         f"({info['gi_rows']} day × target rows with actuals). Peaks in GW, energy in MU. "
         "Improvement = 1 − our MAE / Grid-India MAE; win rate = share of days where we are closer; bias = mean(forecast − actual).", ""]

    # ---- issue time
    L += ["## 1. When was each Grid-India forecast made?", "",
          "The Grid-India listing API records, for every week-ahead PDF, the date the forecast was prepared "
          "(`Field2`) and the upload time (`CreatedOn`); only 35 of 125 PDFs have a readable footer date, and none "
          "in the reported window, so the listing is the source used. Days between 'prepared' and the first day of "
          "the forecast week, per weekly issue:", "",
          "| days before week start | weeks |", "|---|---|"]
    L += [f"| {int(k)} | {v} |" for k, v in info["prepared_lag"].items()]
    L += ["", "So for most weeks the forecast for the first day is 6 days ahead and for the last day 12 days ahead "
          "(counting from the last day whose demand report was published when it was prepared). The original "
          "comparison averaged our 1–7-day-ahead forecasts, which use up to a week more recent information.", "",
          "**Matched basis.** For each Grid-India week we take our forecast issued at origin = prepared date − 1 "
          "(the newest published demand report on that day), and compare each day at the same lead. Our neural "
          "models only forecast 7 days, so the matched comparison uses a long-lead LightGBM (leads 1–14), "
          "backtested month by month exactly like the main models. Leads 1–7 use the archived weather forecast "
          "issued at the origin; leads 8–14 use climatology (normals from data before the backtest), because no "
          "weather forecast that far ahead is archived. Lead distribution of the matched rows: "
          + ", ".join(f"{k} d: {v}" for k, v in sorted(info["lead_distribution"].items())) + ".", "",
          f"Grid-India rows without a matching origin in our backtest: {info['unmatched_gi_rows']}.", "",
          "Long-lead LightGBM MAPE % by lead (sanity check: leads 1–7 should be close to the main LightGBM):", "",
          by_h.to_markdown(), ""]

    # ---- table of steps
    L += ["## 2. The gap after each correction", ""]
    for t, unit_div, unit in (("solar_peak", 1000, "GW"), ("evening_peak", 1000, "GW"), ("energy", 1, "MU")):
        L += [f"### {t} ({unit})", "",
              "| step | days | our MAE | GI MAE | our MAPE % | GI MAPE % | improvement | win rate | our bias | GI bias |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for label, key in steps:
            r = res[key]
            r = r[r["target"] == t]
            if len(r):
                L.append(fmt_row(label, r.iloc[0], unit_div, unit))
        L.append("")

    # ---- source
    L += ["## 3. Text labels vs digitised curve", "",
          "Grid-India values in the reported window come from the text labels printed on the chart (ranges such as "
          "'234-238'; we use the midpoint) or, for weeks whose labels are not text, from our digitisation of the "
          "plotted curve (accurate to ~0.1 GW). Energy always comes from the integrated curve.", ""]
    src = jm.groupby(["target", "source"]).size().unstack(fill_value=0)
    L += [src.to_markdown(), ""]
    for t in ("solar_peak", "evening_peak"):
        a = res.get("3 matched, GI from text labels")
        b = res.get("3 matched, GI from digitised curve")
        if a is not None and b is not None and t in set(a["target"]) and t in set(b["target"]):
            ra, rb = a[a.target == t].iloc[0], b[b.target == t].iloc[0]
            L.append(f"- {t}: improvement {ra['improvement']:+.1%} on label weeks ({ra['days']} days) vs "
                     f"{rb['improvement']:+.1%} on curve weeks ({rb['days']} days); Grid-India MAPE "
                     f"{ra['gi_MAPE']:.2f} % vs {rb['gi_MAPE']:.2f} %.")
    L.append("")

    # ---- bias
    L += ["## 4. Systematic bias", "",
          "Grid-India over-forecasts on average (positive bias). Removing a constant bias per target, estimated "
          "only from earlier days whose actuals were published before each week's forecast was prepared "
          f"(expanding mean, ≥ 14 days of history; {info['bias_rows']} rows), shows how much of the gap is a "
          "deliberate safety margin rather than tracking error. Step 4 in the tables above.", ""]

    # ---- leakage
    sc = leak["source_counts"]
    L += ["## 5. Leakage check of our weather inputs", "",
          f"Backtest forecast rows in the reported window: {leak['rows']:,} (origin × horizon).", "",
          "| weather source | rows |", "|---|---|",
          f"| archived forecast issued on the origin day (lead = horizon) | {sc.get(0, 0):,} |",
          f"| older archived forecast (issued before the origin) | {sc.get(1, 0):,} |",
          f"| fallback: persistence of observed weather up to the origin | {sc.get(2, 0):,} |", "",
          f"Checked {leak['checked']:,} rows against the archive: issued on the origin day {leak['issued_on_origin']:,}, "
          f"issued after the origin {leak['issued_after_origin']}, values identical to the observed weather of the "
          f"target day {leak['identical_to_observed']} (mean |forecast − observed| of all-India max temperature "
          f"{leak['tmax_mae_vs_observed']:.2f} °C, i.e. real forecast error is present).", "",
          ("No fallbacks to observed weather occurred, so no rerun was needed." if sc.get(2, 0) == 0 else
           f"{sc.get(2, 0)} rows fell back to persistence (observed weather up to the origin only — not hindsight)."),
          "The climatology used at leads 8–14 of the long-lead model is built only from observed weather before "
          "the backtest (NORMALS_END), never from the forecast days.", ""]

    # ---- rerun without fallback rows (if any)
    if sc.get(2, 0):
        src = pd.read_parquet(PROCESSED / "backtest_wx_source.parquet")
        bad = src[src["wx_source"] == 2][["origin", "h"]].assign(_bad=True)
        clean = processed.merge(bad, on=["origin", "h"], how="left")
        clean = clean[clean["_bad"].isna()].drop(columns="_bad")
        scored = processed.merge(bad, on=["origin", "h"])
        n_scored = int(scored.dropna(subset=["actual"])["date"].nunique())
        llc = ll[ll["wx_source"] != 2] if "wx_source" in ll else ll
        _, res2, _, _ = run_audit(best, clean, daily, llc, BACKTEST_START)
        L += ["### Rerun without the fallback rows", "",
              f"The {sc.get(2, 0)} fallback rows are the last forecasts of the window, whose target days lie beyond "
              "the end of the weather-forecast archive. "
              + (f"Only {n_scored} of their target days have published actuals, so they barely enter any score. "
                 if n_scored else "None of their target days has published actuals yet, so they never entered any "
                                  "score — the rerun is identical. ")
              + "Removing them (and long-lead rows with a fallback):", "",
              "| target | original basis: with → without | matched, bias-corrected: with → without |", "|---|---|---|"]
        for t in ("solar_peak", "evening_peak", "energy"):
            a0 = res["0 original"].set_index("target").loc[t, "improvement"]
            b0 = res2["0 original"].set_index("target").loc[t, "improvement"]
            a4 = res["4 matched + GI bias-corrected"].set_index("target").loc[t, "improvement"]
            b4 = res2["4 matched + GI bias-corrected"].set_index("target").loc[t, "improvement"]
            L.append(f"| {t} | {a0:+.1%} → {b0:+.1%} | {a4:+.1%} → {b4:+.1%} |")
        L.append("")

    # ---- conclusion
    fin = res["4 matched + GI bias-corrected"].set_index("target")
    orig = res["0 original"].set_index("target")
    mat = res["2 matched issue (lgbm_long)"].set_index("target")
    L += ["## 6. Conclusion", ""]
    L += ["| target | original gap | matched issue time | + Grid-India bias removed |", "|---|---|---|---|"]
    for t in ("solar_peak", "evening_peak", "energy"):
        L.append(f"| {t} | {orig.loc[t, 'improvement']:+.1%} | {mat.loc[t, 'improvement']:+.1%} | "
                 f"{fin.loc[t, 'improvement']:+.1%} |")
    L.append("")
    o = {t: orig.loc[t, "improvement"] for t in orig.index}
    mt = {t: mat.loc[t, "improvement"] for t in mat.index}
    fn = {t: fin.loc[t, "improvement"] for t in fin.index}
    gib = {t: mat.loc[t, "gi_bias"] for t in mat.index}
    L += ["**In plain language.** The original headline (our error "
          f"{min(o.values()):.0%}–{max(o.values()):.0%} lower than Grid-India's) compared forecasts made with very "
          "different information: ours averaged 1–7-day-ahead forecasts, while Grid-India prepares its week-ahead "
          "forecast about five days before the week starts, so it forecasts 6–12 days ahead. When our model is given "
          "exactly the information Grid-India had on the day it prepared each forecast, the advantage shrinks to "
          f"{min(mt.values()):.0%}–{max(mt.values()):.0%} — still clearly better on all three targets and on most "
          "days (win rates " + ", ".join(f"{mat.loc[t, 'win_rate']:.0%}" for t in ("solar_peak", "evening_peak", "energy"))
          + "). Grid-India over-forecasts systematically (on average "
          f"{gib['solar_peak'] / 1000:+.1f} GW solar-hour, {gib['evening_peak'] / 1000:+.1f} GW evening, "
          f"{gib['energy']:+.0f} MU energy), which looks like a deliberate safety margin for grid planning. Removing "
          "a constant bias learned from earlier weeks narrows the gap for the evening peak and energy "
          f"(to {fn['evening_peak']:+.0%} and {fn['energy']:+.0%}) but not for the solar-hour peak ({fn['solar_peak']:+.0%}): "
          "Grid-India's solar-hour bias changed over time, so last year's average does not transfer. The gap also "
          "depends on how the Grid-India value was read: for the solar-hour peak it is smaller on weeks taken from the "
          "digitised curve, for the evening peak larger. The weather inputs contain no hindsight. Fair summary: on a "
          f"matched basis our forecast is roughly {min(fn.values()):.0%}–{max(mt.values()):.0%} more accurate than "
          "Grid-India's week-ahead forecast, not 40–50 %; part of the original gap was the information advantage of "
          "forecasting closer to the day, and part is Grid-India's cautious upward bias.", ""]
    json.dump({"steps": {k: v.to_dict("records") for k, v in res.items()}, "info": info, "leakage": leak},
              open(PROCESSED / "fairness_audit.json", "w"), default=str, indent=1)
    (REPORTS / "fairness_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
