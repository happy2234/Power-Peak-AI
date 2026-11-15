#!/usr/bin/env python
"""Error analysis of the baseline on the validation months (never the test period).

    python scripts/error_analysis.py      # -> reports/error_analysis.md

Forecasts: the baseline recipe (seed 42), ensemble built out of fold inside the validation window
(src/accuracy_round.py). Second view: the inner fold Oct–Dec 2024 (models trained on earlier
origins), which adds the festival season the January–March validation lacks.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS, TARGETS  # noqa: E402

TGT = list(TARGETS)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def frame(inner: bool) -> pd.DataFrame:
    """OOF-calibrated rows of the ensemble and its members, with error columns and context."""
    from src import accuracy_round as ar
    from src.evaluate import weather_regimes
    from src.features.calendar import calendar_features

    r = ar.Recipe()
    daily = ar.data(inner)[0]
    rows = pd.concat([ar.member_predictions(m, r, 42, inner) for m in r.members], ignore_index=True)
    cal = ar.oof(rows, r, daily)
    cal["err_pct"] = 100 * (cal["p50"] - cal["actual"]) / cal["actual"]
    cal["ape"] = cal["err_pct"].abs()
    cal["inside"] = (cal["actual"] >= cal["p10"]) & (cal["actual"] <= cal["p90"])
    c = calendar_features(cal["date"].unique())
    cal["weekday"] = cal["date"].dt.day_of_week.map(dict(enumerate(DAYS)))
    cal["month"] = cal["date"].dt.strftime("%b %Y")
    cal["holiday"] = cal["date"].map(c["is_holiday"]).astype(bool)
    near = (c["days_to_festival"] <= 3) | (c["days_from_festival"] <= 2) | c["diwali_window"].astype(bool)
    cal["calendar"] = np.where(cal["holiday"], "holiday/festival day",
                               np.where(cal["date"].map(near), "within 3 days of a festival", "ordinary day"))
    cal["regime"] = weather_regimes(daily).reindex(cal["date"]).to_numpy()
    return cal


def by(cal: pd.DataFrame, key: str, order=None) -> pd.DataFrame:
    e = cal[cal["model"] == "ensemble"]
    g = e.groupby(["target", key])
    t = pd.DataFrame({"MAPE %": g["ape"].mean(), "bias %": g["err_pct"].mean(), "sd %": g["err_pct"].std(),
                      "coverage %": 100 * g["inside"].mean(), "days": g["date"].nunique()})
    t = t.unstack("target")
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    # one MAPE column per target, then the mean, then bias of the mean
    out = pd.DataFrame({f"MAPE {k}": t[("MAPE %", k)] for k in TGT})
    out["MAPE mean"] = out.mean(axis=1)
    for k in TGT:
        out[f"bias {k}"] = t[("bias %", k)]
    out["days"] = t[("days", TGT[0])]
    return out.round(2)


def share(cal: pd.DataFrame, key: str) -> pd.Series:
    """Share of total absolute % error (ensemble) contributed by each group, and its share of rows."""
    e = cal[cal["model"] == "ensemble"]
    s = e.groupby(key)["ape"].sum() / e["ape"].sum() * 100
    n = e.groupby(key).size() / len(e) * 100
    return pd.DataFrame({"share of error %": s, "share of rows %": n}).round(1)


def worst(cal: pd.DataFrame, daily: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    e = cal[cal["model"] == "ensemble"]
    d = e.groupby(["date", "target"]).agg(actual=("actual", "first"), forecast=("p50", "mean"),
                                          err=("err_pct", "mean"), ape=("ape", "mean"),
                                          calendar=("calendar", "first"), regime=("regime", "first"),
                                          weekday=("weekday", "first")).reset_index()
    d = d.sort_values("ape", ascending=False).head(n)
    wx = daily.reindex(d["date"])
    d["tmax °C"] = wx["tmax_ai"].to_numpy()
    d["tmax vs normal"] = wx["tmax_ai_anom"].to_numpy() if "tmax_ai_anom" in wx else np.nan
    d["N-India tmin °C"] = wx["tmin_nr"].to_numpy()
    d["rain mm"] = wx["precip_ai"].to_numpy()
    scale = d["target"].map({"solar_peak": 1000, "evening_peak": 1000, "energy": 1})
    d["actual"] = (d["actual"] / scale).round(1)
    d["forecast"] = (d["forecast"] / scale).round(1)
    d["date"] = d["date"].dt.strftime("%a %d %b %Y")
    return d[["date", "target", "actual", "forecast", "err", "calendar", "regime", "tmax °C", "tmax vs normal",
              "N-India tmin °C", "rain mm"]].rename(columns={"err": "error % (mean of 1–7 d)"}).round(2)


def lead_bias(daily: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Archived forecast weather minus observed, by lead (training-period dates only)."""
    from src.features.build import _fc_matrix
    from src.fixed_split import TRAIN_END

    fcm = _fc_matrix(fc)
    out = {}
    for c in ("tmax_ai", "tmin_nr", "precip_ai", "cdd_ai"):
        obs = daily.loc[:TRAIN_END, c]
        out[c] = {lead: float((fcm[lead][c].reindex(obs.index) - obs).mean()) for lead in range(1, 8)}
    return pd.DataFrame(out).round(2)


def findings(cal: pd.DataFrame, ic: pd.DataFrame | None) -> list[str]:
    from src import accuracy_round as ar

    def stats(c):
        e = c[c["model"] == "ensemble"]
        sh = share(c, "calendar")
        fest_err = sh.loc[["holiday/festival day", "within 3 days of a festival"], "share of error %"].sum()
        fest_rows = sh.loc[["holiday/festival day", "within 3 days of a festival"], "share of rows %"].sum()
        fb = e[e["calendar"] != "ordinary day"]["err_pct"].mean()
        h = e.groupby("h")["ape"].mean()
        mb = e.groupby(["target", "month"])["err_pct"].mean()
        reg = e.groupby("regime")["ape"].mean()
        return fest_err, fest_rows, fb, h, mb, reg

    fe, fr, fb, h, mb, reg = stats(cal)
    lb = lead_bias(*ar.data(False)[:2])
    L = ["## Findings: the three biggest error sources", "",
         f"1. **Festivals and the days around them.** Holiday/festival days and the 3 days around them are "
         f"{fr:.0f} % of the validation rows but {fe:.0f} % of the error, with a systematic over-forecast "
         f"({fb:+.1f} % on those rows; Holi on 14 Mar 2025 and Makar Sankranti on 14 Jan lead the worst-day list)."]
    if ic is not None:
        fe2, fr2, fb2, *_ = stats(ic)
        L[-1] += (f" The festival season confirms it: Oct–Dec 2024 festival rows are {fr2:.0f} % of the rows, "
                  f"{fe2:.0f} % of the error, bias {fb2:+.1f} % (Dussehra, Diwali week).")
    L += ["   *Hypothesis:* each festival occurs only once or twice in the training data and the calendar flags "
          "are binary, so the model learns one generic holiday dip and cannot size it per festival. The same "
          "festival last year, aligned on the festival date rather than on the weekday, is a direct analogue of "
          "its size and of the days around it (experiment a, `lag365`).", "",
          f"2. **Level drift that grows with the horizon.** Ensemble MAPE rises from {h.iloc[0]:.2f} % one day "
          f"ahead to {h.iloc[-1]:.2f} % at seven days, and some months carry a systematic bias (energy "
          f"{mb.get(('energy', 'Jan 2025'), float('nan')):+.1f} % in January, solar-hour "
          f"{mb.get(('solar_peak', 'Mar 2025'), float('nan')):+.1f} % in March; Oct 2024 "
          + (f"{ic[ic.model == 'ensemble'].groupby(['target', 'month'])['err_pct'].mean().get(('energy', 'Oct 2024'), float('nan')):+.1f} % energy"
             if ic is not None else "–") + ").",
          "   *Hypothesis:* the label is demand relative to the last 7 days, so the model must infer where the "
          "level is heading at seasonal turning points (winter peak, the March warm-up, the post-monsoon decline) "
          "from weather alone. Year-on-year growth and momentum features (experiment a, `growth`), a different "
          "label base (b) and weighting recent months (c) each attack this directly.", "",
          f"3. **Weather-sensitive transition days.** Cold/fog days score {reg.get('cold/fog', float('nan')):.2f} % "
          f"MAPE against {reg.get('normal', float('nan')):.2f} % on normal days, and the March heat spell "
          "(tmax about 3 °C above normal) coincides with a solar-hour over-forecast.",
          "   *Hypothesis:* the temperature response differs by season and by weekday (cooling load on a working "
          "day vs a Sunday), which a model with additive weather effects and few examples per season captures "
          "poorly. Season × degree-day and temperature × weekend interactions (experiment a, `interactions`) test "
          "this.", "",
          "**Also found: the 7-day-ahead weather inputs are biased.** Both members jump to an over-forecast at "
          "h = 7 in both windows. The archived Open-Meteo forecasts issued 7 days ahead are much further from the "
          "observed weather than leads 1–6 (training period, forecast minus observed):", "",
          lb.to_markdown(), "",
          "All 24 hours are present (checked on a sample), so the day-7 series is a different, coarser forecast, "
          "not a truncated day. *Hypothesis:* removing the lead-7 offset (measured on the training period only) "
          "removes the h = 7 bias. Whether today's live day-7 forecast carries the same offset cannot be checked "
          "yet (7 live days). Tested as an extra experiment (a0) before the planned list.", ""]
    return L


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    from src import accuracy_round as ar
    from src.evaluate import REGIME_ORDER

    cal = frame(inner=False)
    daily = ar.data(False)[0]
    e = cal[cal["model"] == "ensemble"]
    mem = cal.groupby(["model", "target"])["ape"].mean().unstack().round(2)
    mem["mean"] = mem.mean(axis=1).round(2)
    L = ["# Error analysis (validation, Jan–Mar 2025)", "",
         "Baseline recipe (production): LSTM + LightGBM ensemble, weights per target × horizon. Forecasts from "
         f"{cal['origin'].min().date()} to {cal['origin'].max().date()} ({cal['origin'].nunique()} origins, 1–7 days "
         "ahead), seed 42. The ensemble and its P10–P90 calibration are fitted out of fold (January ↔ February–March), "
         "the way every experiment of this round is scored. Errors are signed % of the actual (+ = forecast too "
         "high). No test-period data is used anywhere in this report.", "",
         "## By model and target — MAPE %", "", mem.to_markdown(), "",
         "## By horizon", "", by(cal, "h").to_markdown(), "",
         "## By weekday of the target day", "", by(cal, "weekday", DAYS).to_markdown(), "",
         "## By month (bias = systematic over/under-forecast)", "", by(cal, "month").to_markdown(), "",
         "## By weather regime (observed weather; evaluation only)", "", by(cal, "regime", REGIME_ORDER).to_markdown(), "",
         "## Holidays and festivals", "", by(cal, "calendar", ["ordinary day", "within 3 days of a festival",
                                                                  "holiday/festival day"]).to_markdown(), "",
         "Share of the total error:", "", share(cal, "calendar").to_markdown(), ""]

    # bias vs variance by month and target
    bv = e.groupby(["target", "month"])["err_pct"].agg(["mean", "std", "count"])
    bv["bias² share of MSE %"] = 100 * bv["mean"] ** 2 / (bv["mean"] ** 2 + bv["std"] ** 2)
    L += ["## Bias vs variance", "",
          "For each target and month: mean signed error (bias), its spread, and how much of the squared error is "
          "systematic (bias²) rather than day-to-day noise.", "",
          bv.rename(columns={"mean": "bias %", "std": "sd %", "count": "rows"}).round(2).to_markdown(), ""]
    L += ["## The 20 worst validation days (ensemble, target-day error averaged over the 1–7-day forecasts)", "",
          worst(cal, daily).to_markdown(index=False), ""]

    # inner fold
    ic = None
    try:
        ic = frame(inner=True)
        L += ["## Second view: inner fold Oct–Dec 2024 (training period; festival season)", "",
              "Same recipe, trained on origins up to Sep 2024, normals up to Sep 2024.", "",
              "By month:", "", by(ic, "month").to_markdown(), "",
              "Holidays and festivals:", "", by(ic, "calendar", ["ordinary day", "within 3 days of a festival",
                                                                  "holiday/festival day"]).to_markdown(), "",
              "Share of the total error:", "", share(ic, "calendar").to_markdown(), "",
              "By horizon:", "", by(ic, "h").to_markdown(), "",
              "Worst days:", "", worst(ic, ar.data(True)[0], 10).to_markdown(index=False), ""]
    except Exception as ex:  # pragma: no cover
        L += [f"(inner fold failed: {ex})", ""]
    L = L[:4] + findings(cal, ic) + L[4:]
    text = "\n".join(L) + "\n"
    (REPORTS / "error_analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
