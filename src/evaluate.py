"""Rolling-origin backtest + metrics + leaderboard.

For each month M in the backtest window, every model is refit on all origins whose labels are
fully known before M starts (origin <= M_start - 8 days), then forecasts every origin in M
(daily origins, 7 days ahead each). Results:
    data/processed/backtest_predictions.parquet   long format, all models
    reports/results.md                            leaderboard + per-horizon tables
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED, REPORTS, TARGETS
from src.models.common import metrics

log = logging.getLogger(__name__)
BACKTEST_START = "2025-11-01"   # first month that is reported
WARMUP_START = "2025-05-01"     # months before BACKTEST_START: out-of-sample predictions used only
                                # to learn ensemble weights / conformal corrections / recalibration
DL_KINDS = ("lstm", "tft")
REGIME_BOOST = 1.0              # "<model>_w": rare-weather regimes count double in the loss
# Model recipes from the accuracy rounds, selected by a "+<tag>" suffix on the variant name
# ("lgbm@compact+nov26"). Tag -> options per family (reports/accuracy_round.md).
RECIPES = {
    "nov26": {"lgbm": {"extra": ("interactions", "lag365"), "recency_halflife": 365},
              "lstm": {"recency_halflife": 365}},
}


def actuals_long(daily: pd.DataFrame) -> pd.DataFrame:
    a = daily[list(TARGETS)].stack().rename("actual").reset_index()
    a.columns = ["date", "target", "actual"]
    return a


def recipe_tag(name: str) -> str | None:
    """'lgbm@compact+nov26' -> 'nov26'; None without a tag."""
    return name.partition("+")[2] or None


def recipe_options(name: str) -> dict:
    """Options of the variant's recipe for its family ({} for untagged variants)."""
    tag = recipe_tag(name)
    return dict(RECIPES[tag].get(parse_variant(name)[0], {})) if tag else {}


def parse_variant(name: str) -> tuple[str, bool, str]:
    """'lstm_w@compact' -> ('lstm', True, 'compact'); 'lgbm' -> ('lgbm', False, 'full').
    A recipe tag ('+nov26') is ignored here; see recipe_options()."""
    name = name.partition("+")[0]
    name, _, fs = name.partition("@")
    base, weighted = (name[:-2], True) if name.endswith("_w") else (name, False)
    return base, weighted, fs or "full"


def _fit_predict(name, daily, fc, train_o, test_o, tab_tr, tab_te) -> pd.DataFrame:
    from src.models.baselines import LGBMQuantile, SeasonalNaive
    from src.models.tune import best_params

    base, weighted, fs = parse_variant(name)
    boost = REGIME_BOOST if weighted else 0.0
    if base == "seasonal_naive":
        p = SeasonalNaive().fit(tab_tr).predict_long(tab_te)
    elif base == "lgbm":
        p = LGBMQuantile(best_params("lgbm", fs), regime_boost=boost, feature_set=fs,
                         **recipe_options(name)).fit(tab_tr).predict_long(tab_te)
    elif base in DL_KINDS:
        from src.models.deep import DeepForecaster
        from src.utils.device import free_gpu

        m = DeepForecaster(base, best_params(base, fs), regime_boost=boost, feature_set=fs,
                           **recipe_options(name)).fit(daily, fc, train_o)
        p = m.predict_long(daily, fc, test_o)
        del m
        free_gpu()
    else:
        raise ValueError(name)
    p["model"] = name
    return p


def backtest(models=("seasonal_naive", "lgbm", "lstm", "tft"), start: str = WARMUP_START,
             end: str | None = None, save: bool = True, merge: bool = True) -> pd.DataFrame:
    """Monthly refits from `start`; `merge`: keep saved predictions of models not in `models`
    (so variants can be run one at a time)."""
    from src.features.build import load_daily, make_tabular, valid_origins
    from src.models.baselines import grid_india_long

    daily, fc = load_daily()
    end = pd.Timestamp(end) if end else daily.index.max() - pd.Timedelta(days=1)
    months = pd.date_range(pd.Timestamp(start), end, freq="MS")
    preds, timing, wxsrc = [], {m: 0.0 for m in models}, []
    for ms in months:
        me = min(ms + pd.offsets.MonthEnd(0), end)
        test_o = valid_origins(daily, start=ms, end=me)
        train_o = valid_origins(daily, start="2023-06-01", end=ms - pd.Timedelta(days=8))
        if len(test_o) == 0:
            continue
        log.info("backtest month %s: train origins %d, test origins %d", ms.strftime("%Y-%m"),
                 len(train_o), len(test_o))
        tab_tr = make_tabular(daily, fc, train_o)
        tab_te = make_tabular(daily, fc, test_o)
        wxsrc.append(tab_te[["origin", "h", "date", "wx_source"]])
        for name in models:
            t0 = time.time()
            p = _fit_predict(name, daily, fc, train_o, test_o, tab_tr, tab_te)
            timing[name] += time.time() - t0
            p["refit_month"] = ms
            preds.append(p)
    bt = pd.concat(preds, ignore_index=True)
    bt = bt[bt["date"] <= daily.index.max()]
    path = PROCESSED / "backtest_predictions.parquet"
    if merge and path.exists():
        old = pd.read_parquet(path)
        old = old[~old["model"].isin(list(models) + ["grid_india"])]
        bt = pd.concat([old, bt], ignore_index=True)
    else:
        bt = bt[bt["model"] != "grid_india"]

    gi = grid_india_long(actuals_long(daily))
    if len(gi):
        gi = gi[(gi["date"] >= pd.Timestamp(start)) & (gi["date"] <= daily.index.max())]
        bt = pd.concat([bt, gi], ignore_index=True)
    if save:
        import json

        bt.to_parquet(path, index=False)
        tp = PROCESSED / "backtest_timing.json"
        old_t = json.loads(tp.read_text()) if (merge and tp.exists()) else {}
        tp.write_text(json.dumps({**old_t, **timing}))
        pd.concat(wxsrc, ignore_index=True).to_parquet(PROCESSED / "backtest_wx_source.parquet", index=False)
    return bt


def eval_window(bt: pd.DataFrame, start: str = BACKTEST_START) -> pd.DataFrame:
    """The reported 12 months: our forecasts issued from `start` on; Grid-India days from `start`."""
    s = pd.Timestamp(start)
    ours = (bt["model"] != "grid_india") & (bt["origin"] >= s)
    gi = (bt["model"] == "grid_india") & (bt["date"] >= s)
    return bt[ours | gi]


# ============================================================================ tables
def leaderboard(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, t), g in bt.groupby(["model", "target"]):
        rows.append({"model": m, "target": t, **metrics(g)})
    return pd.DataFrame(rows)


def by_horizon(bt: pd.DataFrame, metric: str = "MAPE") -> pd.DataFrame:
    rows = []
    for (m, t, h), g in bt[bt["model"] != "grid_india"].groupby(["model", "target", "h"]):
        rows.append({"model": m, "target": t, "h": h, metric: metrics(g).get(metric)})
    return pd.DataFrame(rows).pivot_table(index=["target", "model"], columns="h", values=metric)


def head_to_head(bt: pd.DataFrame, model: str, horizon: int | None = None) -> pd.DataFrame:
    """Our model vs Grid-India on the same target days. horizon=None: all horizons averaged per
    day (mean of the 7 forecasts for that day); horizon=k: only our h=k forecast."""
    gi = bt[bt["model"] == "grid_india"][["date", "target", "p50", "actual"]]
    ours = bt[bt["model"] == model]
    if horizon:
        ours = ours[ours["h"] == horizon]
    ours = ours.groupby(["date", "target"], as_index=False).agg(p50=("p50", "mean"), actual=("actual", "first"))
    j = gi.merge(ours, on=["date", "target"], suffixes=("_gi", "_ours")).dropna(subset=["actual_gi"])
    j["actual"] = j["actual_gi"]
    j["ae_gi"] = (j["p50_gi"] - j["actual"]).abs()
    j["ae_ours"] = (j["p50_ours"] - j["actual"]).abs()
    j["ours_wins"] = j["ae_ours"] < j["ae_gi"]
    return j


def best_model(bt: pd.DataFrame) -> str:
    """Lowest mean scaled pinball (pinball / mean actual) across targets."""
    lb = leaderboard(bt[bt["model"] != "grid_india"]).dropna(subset=["pinball"])
    scale = bt.groupby("target")["actual"].mean()
    lb["scaled"] = lb["pinball"] / lb["target"].map(scale)
    return lb.groupby("model")["scaled"].mean().idxmin()


def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def write_results(bt: pd.DataFrame, extra: dict | None = None) -> str:
    lb = leaderboard(bt)
    best = best_model(bt)
    L = ["# Backtest results", "",
         f"Rolling-origin backtest {bt['date'].min():%Y-%m-%d} .. {bt['date'].max():%Y-%m-%d}: models "
         "refit monthly on all data known before the month, daily forecast origins, 7-day horizon. "
         "Weather inputs are the archived forecasts issued at the origin (no hindsight). "
         "Peaks in MW (errors shown in MW), energy in MU.", "",
         f"**Best model (lowest scaled pinball across targets): `{best}`**", ""]
    for t in TARGETS:
        L += [f"## {t}", "", "| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |",
              "|---|---|---|---|---|---|---|---|"]
        sub = lb[lb["target"] == t].sort_values("MAPE")
        for _, r in sub.iterrows():
            L.append(f"| {r['model']} | {int(r['n'])} | {_fmt(r['MAPE'])} | {_fmt(r['MAE'], 0)} | "
                     f"{_fmt(r['RMSE'], 0)} | {_fmt(r['bias'], 0)} | {_fmt(r.get('pinball'), 0)} | "
                     f"{_fmt(r.get('coverage80'), 1)} |")
        L.append("")
    L += ["Grid-India rows: only days with a published week-ahead forecast; it is a point forecast "
          "(no quantiles). Energy for Grid-India = integral of its 15-min forecast curve.", "",
          "## MAPE by horizon (days ahead)", "", by_horizon(bt).round(2).to_markdown(), "",
          "## Coverage of the P10–P90 interval by horizon (target 80 %)", "",
          by_horizon(bt, "coverage80").round(1).to_markdown(), ""]
    L += ["## Head-to-head vs Grid-India (same days)", "",
          "| target | our model | days | our MAE | Grid-India MAE | improvement | win rate |",
          "|---|---|---|---|---|---|---|"]
    for hz, label in [(None, f"{best} (mean of h=1..7)"), (7, f"{best} @ h=7")]:
        j = head_to_head(bt, best, hz)
        for t, g in j.groupby("target"):
            imp = 1 - g["ae_ours"].mean() / g["ae_gi"].mean()
            L.append(f"| {t} | {label} | {len(g)} | {g['ae_ours'].mean():,.0f} | {g['ae_gi'].mean():,.0f} | "
                     f"{imp:+.1%} | {g['ours_wins'].mean():.0%} |")
    L.append("")
    if extra:
        for title, body in extra.items():
            L += [f"## {title}", "", body, ""]
    text = "\n".join(L) + "\n"
    (REPORTS / "results.md").write_text(text)
    return text


def failure_analysis(bt: pd.DataFrame, daily: pd.DataFrame, models=("lgbm", "tft", "lstm", "grid_india")) -> pd.DataFrame:
    """MAPE (%) of each model by day condition, all targets pooled per target."""
    from src.features.calendar import calendar_features

    cal = calendar_features(daily.index)
    hot = daily["tmax_ai"] >= daily["tmax_ai"].quantile(0.95)
    fest_win = pd.Series(False, index=daily.index)
    for d in daily.index[cal["is_festival"] == 1]:
        fest_win.loc[d - pd.Timedelta(days=1):d + pd.Timedelta(days=1)] = True
    conds = {
        "all days": pd.Series(True, index=daily.index),
        "normal weekdays": (cal["is_holiday"] == 0) & (cal["dow"] < 5) & ~fest_win,
        "Sundays": cal["dow"] == 6,
        "national holidays": cal["is_national_holiday"] == 1,
        "festival ±1 day": fest_win,
        "Diwali window": cal["diwali_window"] == 1,
        "heatwave (top 5 % temp)": hot,
        "monsoon (Jul–Sep)": pd.Series(daily.index.month.isin([7, 8, 9]), index=daily.index),
    }
    b = bt[bt["model"].isin(models)].dropna(subset=["actual", "p50"]).copy()
    b["ape"] = (b["p50"] - b["actual"]).abs() / b["actual"] * 100
    rows = []
    for name, mask in conds.items():
        days = set(mask[mask].index)
        sub = b[b["date"].isin(days)]
        for (m, t), g in sub.groupby(["model", "target"]):
            rows.append({"condition": name, "model": m, "target": t, "MAPE": g["ape"].mean(),
                         "days": g["date"].nunique()})
    return pd.DataFrame(rows)


def worst_days(bt: pd.DataFrame, model: str, n: int = 10) -> pd.DataFrame:
    b = bt[bt["model"] == model].dropna(subset=["actual"]).copy()
    b["ape"] = (b["p50"] - b["actual"]).abs() / b["actual"] * 100
    d = b.groupby(["date", "target"])["ape"].mean().reset_index()
    return d.sort_values("ape", ascending=False).head(n)


# ============================================================================ weather & season
SEASONS = {12: "winter (Dec–Feb)", 1: "winter (Dec–Feb)", 2: "winter (Dec–Feb)",
           3: "pre-monsoon summer (Mar–May)", 4: "pre-monsoon summer (Mar–May)",
           5: "pre-monsoon summer (Mar–May)", 6: "monsoon (Jun–Sep)", 7: "monsoon (Jun–Sep)",
           8: "monsoon (Jun–Sep)", 9: "monsoon (Jun–Sep)", 10: "post-monsoon (Oct–Nov)",
           11: "post-monsoon (Oct–Nov)"}
SEASON_ORDER = ["winter (Dec–Feb)", "pre-monsoon summer (Mar–May)", "monsoon (Jun–Sep)",
                "post-monsoon (Oct–Nov)"]
REGIME_ORDER = ["heavy rain/storm", "cold/fog", "hot", "rainy", "cloudy", "humid", "normal"]


def regime_thresholds(daily: pd.DataFrame) -> dict:
    """Thresholds from OBSERVED weather over all days (evaluation labels only, never inputs)."""
    q = lambda c, p: float(daily[c].quantile(p))  # noqa: E731
    return {"storm_precip_ai": q("precip_ai", 0.95), "storm_gust_ai": q("gust_max_ai", 0.95),
            "cold_tmin_nr": 8.0, "fog_nr": 0.3, "hot_tmax_ai": q("tmax_ai", 0.90),
            "hot_heat_index_ai": q("heat_index_ai", 0.90), "rain_precip_ai": 2.5,
            "cloud_day_ai": 60.0, "humid_dew_ai": 22.0}


def weather_regimes(daily: pd.DataFrame, th: dict | None = None) -> pd.Series:
    """One label per day from OBSERVED weather, first match in REGIME_ORDER:
    heavy rain/storm  all-India rain or gusts in the top 5 % of days
    cold/fog          Nov–Feb and North-India night <= 8 °C or North fog proxy >= 0.3
    hot               all-India max temperature or heat index in the top 10 %
    rainy             all-India (demand-weighted) rain >= 2.5 mm
    cloudy            daytime cloud cover >= 60 %
    humid             mean dew point >= 22 °C (muggy)
    normal            none of the above"""
    th = th or regime_thresholds(daily)
    d = daily
    winter = pd.Series(d.index.month.isin([11, 12, 1, 2]), index=d.index)
    rules = [
        ("heavy rain/storm", (d["precip_ai"] >= th["storm_precip_ai"]) | (d["gust_max_ai"] >= th["storm_gust_ai"])),
        ("cold/fog", winter & ((d["tmin_nr"] <= th["cold_tmin_nr"]) | (d["fog_nr"] >= th["fog_nr"]))),
        ("hot", (d["tmax_ai"] >= th["hot_tmax_ai"]) | (d["heat_index_ai"] >= th["hot_heat_index_ai"])),
        ("rainy", d["precip_ai"] >= th["rain_precip_ai"]),
        ("cloudy", d["cloud_day_ai"] >= th["cloud_day_ai"]),
        ("humid", d["dew_mean_ai"] >= th["humid_dew_ai"]),
    ]
    lab = pd.Series("normal", index=d.index)
    done = pd.Series(False, index=d.index)
    for name, m in rules:
        m = m.fillna(False) & ~done
        lab[m] = name
        done |= m
    return lab


def mape_by(bt: pd.DataFrame, labels: pd.Series, order=None) -> pd.DataFrame:
    """MAPE % per target x label x model, with the number of distinct target days."""
    b = bt.dropna(subset=["actual", "p50"]).copy()
    b["label"] = labels.reindex(b["date"]).to_numpy()
    b["ape"] = (b["p50"] - b["actual"]).abs() / b["actual"] * 100
    g = b.groupby(["target", "label", "model"]).agg(MAPE=("ape", "mean"), days=("date", "nunique"))
    out = g.reset_index()
    if order:
        out["label"] = pd.Categorical(out["label"], order, ordered=True)
    return out.sort_values(["target", "label", "model"])


def mape_table(t: pd.DataFrame, models) -> pd.DataFrame:
    """Wide table: rows target x label (with day count), one MAPE column per model."""
    piv = t.pivot_table(index=["target", "label"], columns="model", values="MAPE", observed=True)
    days = t[t["model"] == models[0]].set_index(["target", "label"])["days"]
    piv = piv.reindex(columns=[m for m in models if m in piv.columns])
    piv.insert(0, "days", days.reindex(piv.index).astype("Int64"))
    return piv


def heatwave_mask(daily: pd.DataFrame) -> pd.Series:
    """The 'hottest 5 % of days' condition of the original failure analysis (observed tmax_ai)."""
    return daily["tmax_ai"] >= daily["tmax_ai"].quantile(0.95)


def tag_days(dates, daily: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Weather + calendar context for a list of dates (worst-day tables)."""
    from src.features.calendar import calendar_features

    dates = pd.DatetimeIndex(dates)
    cal = calendar_features(dates.unique()).reindex(dates)   # dates may repeat (one row per target)
    d = daily.reindex(dates)
    fcols = [k for k in cal.columns if k.startswith("fest_")]
    fest = [", ".join(k[5:].replace("_", " ").title() for k, v in zip(fcols, row) if v == 1)
            for row in cal[fcols].to_numpy()]
    out = pd.DataFrame({
        "day": dates.strftime("%a"),
        "weather": labels.reindex(dates).to_numpy(),
        "holiday": np.where(cal["is_national_holiday"].to_numpy() == 1, "national holiday",
                            np.where(cal["is_holiday"].to_numpy() == 1, "holiday", "")),
        "festival": fest,
        "tmax_ai": d["tmax_ai"].round(1).to_numpy(),
        "tmax_anom": d["tmax_ai_anom"].round(1).to_numpy(),
        "rain_ai_mm": d["precip_ai"].round(1).to_numpy(),
        "rain_max_region_mm": d[[f"precip_{r}" for r in ("nr", "wr", "sr", "er", "ner")]].max(axis=1).round(1).to_numpy(),
        "gust_ai_kmh": d["gust_max_ai"].round(0).to_numpy(),
        "cloud_day_ai": d["cloud_day_ai"].round(0).to_numpy(),
    }, index=dates)
    return out
