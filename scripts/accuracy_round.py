#!/usr/bin/env python
"""Accuracy round driver. Validation only, except `final`.

    python scripts/accuracy_round.py baseline            # score the production recipe (3 seeds)
    python scripts/accuracy_round.py exp a1              # run one experiment against the incumbent
    python scripts/accuracy_round.py table               # rewrite reports/experiments.md
    python scripts/accuracy_round.py final               # train the incumbent once, score the test period

State: data/processed/accuracy_round/state.json (incumbent recipe + every decision).
Keep rule: lower out-of-fold validation MAPE (ensemble, mean of 3 targets, mean of 3 seeds) and
coverage no more than 2 points lower; a gain under MIN_GAIN counts as no change (simpler recipe wins).
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import accuracy_round as ar  # noqa: E402
from src.config import REPORTS  # noqa: E402

STATE = ar.OUT / "state.json"
MIN_GAIN = 0.01
COVERAGE_SLACK = 2.0


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"incumbent": asdict(ar.Recipe()), "history": []}


def recipe(d: dict) -> ar.Recipe:
    d = dict(d)
    for k in ("lgbm_extra", "lgbm_params", "lstm_params", "members"):
        d[k] = tuple(tuple(x) if isinstance(x, list) else x for x in d[k])
    return ar.Recipe(**d)


# id -> (title, why, candidates(incumbent) -> [Recipe])
def _hl(inc, member, values):
    return [replace(inc, name=f"{member} half-life {v} d", **{f"{member}_halflife": v}) for v in values]


EXPERIMENTS = {
    "a0": ("Lead-7 weather debiasing (extra, from the error analysis)",
           "The archived 7-day-ahead weather is far warmer/drier than leads 1–6; both models over-forecast at h = 7.",
           lambda inc: [replace(inc, name="lead-7 weather debiased", lead7_debias=True)]),
    "a1": ("Features: interactions (temperature × weekend, CDD × season)",
           "Error source 3: the temperature response differs by season and weekday.",
           lambda inc: [replace(inc, name="+interactions", lgbm_extra=tuple(inc.lgbm_extra) + ("interactions",))]),
    "a2": ("Features: demand growth (YoY 7- and 30-day, 30/90-day momentum)",
           "Error source 2: level drift at seasonal turning points.",
           lambda inc: [replace(inc, name="+growth", lgbm_extra=tuple(inc.lgbm_extra) + ("growth",))]),
    "a3": ("Features: same day last year, festival-aligned (level and shape)",
           "Error source 1: festivals occur once or twice in training; last year's festival sizes this one.",
           lambda inc: [replace(inc, name="+lag365 festival-aligned", lgbm_extra=tuple(inc.lgbm_extra) + ("lag365",))]),
    "b": ("Target transform (LightGBM): day-over-day and week-over-week ratio vs the 7-day-mean ratio",
          "The baseline already predicts a ratio (to the last 7 days); test the other natural bases for drift.",
          lambda inc: [replace(inc, name="target / origin day", lgbm_base="last"),
                       replace(inc, name="target / same weekday last week", lgbm_base="lag7")]),
    "c1": ("Recency weighting, LightGBM (half-life tuned)",
           "Track the latest demand level by weighting recent training months higher.",
           lambda inc: _hl(inc, "lgbm", [90, 180, 365, 730])),
    "c2": ("Recency weighting, LSTM (half-life tuned)",
           "Same idea for the LSTM's training samples.",
           lambda inc: _hl(inc, "lstm", [90, 180, 365, 730])),
    "d1": ("LSTM window length 45 / 90 days (current 60)",
           "Capacity/regularisation sweep around the Optuna best: look-back first.",
           lambda inc: [replace(inc, name=f"LSTM window {w} d", lstm_enc_days=w) for w in (45, 90)]),
    "d2": ("LSTM hidden size 64 / 128 (current 96)", "Capacity.",
           lambda inc: [replace(inc, name=f"LSTM hidden {hdn}",
                                lstm_params=tuple(dict(inc.lstm_params, hidden=hdn).items())) for hdn in (64, 128)]),
    "d3": ("LSTM dropout 0.2 / 0.5 (current 0.34)", "Regularisation.",
           lambda inc: [replace(inc, name=f"LSTM dropout {dp}",
                                lstm_params=tuple(dict(inc.lstm_params, dropout=dp).items())) for dp in (0.2, 0.5)]),
    "e": ("Ensemble: add TFT, weights per target × horizon on a 3-member simplex",
          "The baseline blends LSTM + LightGBM per target × horizon; TFT is trained but unused.",
          lambda inc: [replace(inc, name="LSTM + LightGBM + TFT", members=("lstm", "lgbm", "tft"))]),
    "f": ("Residual correction: small LightGBM on the ensemble's validation residuals",
          "Horizon/weekday/weather patterns left in the ensemble's errors.",
          lambda inc: [replace(inc, name="+residual correction", residual=True)]),
}


def run(eid: str) -> dict:
    st = load_state()
    inc = recipe(st["incumbent"])
    title, why, make = EXPERIMENTS[eid]
    base = ar.evaluate(inc)
    ar.log_result(base, inc)
    results = []
    for cand in make(inc):
        res = ar.evaluate(cand)
        ar.log_result(res, cand)
        results.append((cand, res))
    best_c, best = min(results, key=lambda cr: cr[1]["ensemble"]["MAPE"])
    gain = base["ensemble"]["MAPE"] - best["ensemble"]["MAPE"]
    cov_drop = base["ensemble"]["coverage"] - best["ensemble"]["coverage"]
    paired = [a - b for a, b in zip(base["ensemble"]["per_seed"], best["ensemble"]["per_seed"])]
    keep = gain >= MIN_GAIN and cov_drop <= COVERAGE_SLACK
    entry = {"id": eid, "title": title, "why": why, "incumbent": inc.name, "before": base,
             "candidates": [{"name": c.name, "result": r} for c, r in results], "best": best_c.name,
             "gain": gain, "paired_gain_per_seed": paired, "coverage_drop": cov_drop, "kept": keep}
    if keep:
        st["incumbent"] = asdict(replace(best_c, name=f"{inc.name} + {eid}" if inc.name != "baseline" else f"baseline + {eid}"))
    st["history"].append(entry)
    ar.OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str))
    write_table(st)
    return entry


def write_table(st: dict) -> str:
    L = ["# Accuracy round: experiments", "",
         "All numbers are on the **validation months (Jan–Mar 2025)** of the fixed split; the test period is not "
         "used here. Score: MAPE of the ensemble's P50, mean of the three targets, averaged over 3 seeds; "
         "ensemble weights and P10–P90 calibration fitted out of fold (January ↔ February–March). Coverage: share "
         "of actuals inside the calibrated P10–P90 (target 80 %).", "",
         f"Keep rule: lower validation MAPE and coverage at most {COVERAGE_SLACK:.0f} points lower. A gain under "
         f"{MIN_GAIN} points counts as no change. Seed-to-seed spread of the score is about 0.04 points, so gains "
         "smaller than that are within noise; each row shows the paired gain per seed (same seeds, before minus after).",
         "", "| # | experiment | variants tried (validation MAPE %) | before | best after | gain | paired gain per seed "
         "| coverage before → after | kept |", "|---|---|---|---|---|---|---|---|---|"]
    for e in st["history"]:
        cands = "; ".join(f"{c['name']}: {c['result']['ensemble']['MAPE']:.3f}" for c in e["candidates"])
        b = e["before"]["ensemble"]
        a = next(c["result"]["ensemble"] for c in e["candidates"] if c["name"] == e["best"])
        L.append(f"| {e['id']} | {e['title']} | {cands} | {b['MAPE']:.3f} | {a['MAPE']:.3f} | {e['gain']:+.3f} | "
                 + " / ".join(f"{x:+.3f}" for x in e["paired_gain_per_seed"])
                 + f" | {b['coverage']:.1f} → {a['coverage']:.1f} | {'yes' if e['kept'] else 'no'} |")
    L += ["", "## Details", ""]
    for e in st["history"]:
        L += [f"### {e['id']}. {e['title']}", "", f"Why: {e['why']}", "",
              "| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | "
              "LightGBM MAPE % |", "|---|---|---|---|---|---|---|---|---|"]
        for name, res in [(f"before ({e['incumbent']})", e["before"])] + [(c["name"], c["result"]) for c in e["candidates"]]:
            en = res["ensemble"]
            L.append(f"| {name} | {en['MAPE']:.3f} | {en['MAPE_sd']:.3f} | {en['MAPE_solar_peak']:.3f} | "
                     f"{en['MAPE_evening_peak']:.3f} | {en['MAPE_energy']:.3f} | {en['coverage']:.1f} | "
                     f"{res.get('lstm', {}).get('MAPE', float('nan')):.3f} | {res.get('lgbm', {}).get('MAPE', float('nan')):.3f} |")
        L += ["", f"Decision: {'kept' if e['kept'] else 'not kept'} ({e['best']}, gain {e['gain']:+.3f}, coverage "
              f"change {-e['coverage_drop']:+.1f} points).", ""]
    L += [f"Incumbent after the last experiment: **{st['incumbent']['name']}**.", ""]
    text = "\n".join(L) + "\n"
    (REPORTS / "experiments.md").write_text(text)
    return text


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cmd = sys.argv[1]
    if cmd == "baseline":
        r = ar.Recipe()
        res = ar.evaluate(r)
        ar.log_result(res, r)
        print(json.dumps(res, indent=1, default=str))
    elif cmd == "exp":
        e = run(sys.argv[2])
        print(json.dumps({k: e[k] for k in ("id", "best", "gain", "paired_gain_per_seed", "coverage_drop", "kept")},
                         indent=1, default=str))
    elif cmd == "final":
        r = recipe(load_state()["incumbent"])
        print(ar.final_run(r, dry="--dry" in sys.argv))
    elif cmd == "table":
        print(write_table(load_state()))


if __name__ == "__main__":
    main()
