"""Accuracy round on the fixed split: experiment harness (validation only) and final measurement.

Protocol
    fit         origins whose labels end before VAL_START (Apr 2023 .. Dec 2024)
    validation  Jan–Mar 2025 (the last three months of the training period): early stopping and
                every selection decision of this round
    test        Apr 2025 onwards: touched only by final() (logged in test_access.py)

Decision metric (experiments.md). Every recipe is trained with SEEDS; for each seed the ensemble is
built OUT OF FOLD inside the validation window: ensemble weights and conformal calibration are
fitted on one half (January, or February–March) and applied to the other, so the score never uses
weights fitted on the same days. Score = MAPE of the ensemble's P50, mean of the three targets,
averaged over seeds; coverage = share of actuals inside the calibrated P10–P90 (same OOF rows).

Optional second view (error analysis only): an inner fold inside the training period, Oct–Dec 2024,
with models trained on earlier origins and weather normals fitted up to Sep 2024.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from src import fixed_split as fx
from src import test_access
from src.config import HORIZON, PROCESSED, TARGETS

log = logging.getLogger(__name__)
TGT = list(TARGETS)
OUT = PROCESSED / "accuracy_round"
CACHE = OUT / "cache"
SEEDS = (42, 43, 44)
INNER = {"val_start": "2024-10-01", "train_end": "2024-12-31", "normals_end": "2024-09-30"}


@dataclass(frozen=True)
class Recipe:
    """Everything an experiment can change. Defaults = the production (baseline) recipe."""
    name: str = "baseline"
    lead7_debias: bool = False             # remove the archived lead-7 weather offset (training dates only)
    lgbm_extra: tuple = ()                 # feature groups added to LightGBM (features.EXTRA_GROUPS)
    lgbm_base: str = "ref7"                # LightGBM target: actual / base (ref7 | last | lag7)
    lgbm_halflife: float | None = None     # recency weighting of LightGBM training rows (days)
    lgbm_params: tuple = ()                # overrides of the tuned LightGBM params, as (key, value) pairs
    lstm_halflife: float | None = None     # recency weighting of LSTM training samples (days)
    lstm_params: tuple = ()                # overrides of the tuned LSTM params
    lstm_enc_days: int = 60
    members: tuple = ("lstm", "lgbm")      # ensemble members (weights per target x horizon)
    residual: bool = False                 # residual-correction LightGBM on top of the ensemble
    note: str = ""

    def part(self, member: str) -> dict:
        """The settings that change a member's predictions (its cache key)."""
        d7 = {"d7": True} if self.lead7_debias else {}
        if member == "lgbm":
            return {"extra": list(self.lgbm_extra), "base": self.lgbm_base, "hl": self.lgbm_halflife,
                    "params": [list(x) for x in self.lgbm_params], **d7}
        if member == "lstm":
            return {"hl": self.lstm_halflife, "params": [list(x) for x in self.lstm_params],
                    "enc": self.lstm_enc_days, **d7}
        return d7


# ============================================================================ data
_DATA: dict = {}


def data(inner: bool = False, lead7_debias: bool = False):
    """(daily, fc, fit_o, val_o) for the official split, or the inner Oct–Dec 2024 fold."""
    key = ("inner" if inner else "official") + ("-d7" if lead7_debias else "")
    if lead7_debias and key not in _DATA:
        from src.features.build import debias_lead

        daily, fc, fit_o, val_o = data(inner)
        end = INNER["normals_end"] if inner else fx.VAL_START          # offsets from fit-period dates only
        fc2, offs = debias_lead(fc, daily, end=str(pd.Timestamp(end) - pd.Timedelta(days=1)))
        _DATA[key] = (daily, fc2, fit_o, val_o)
        _DATA[key + "-offsets"] = offs
        return _DATA[key]
    if key not in _DATA:
        if inner:
            daily, fc = fx.features(INNER["normals_end"])
            fit_o, val_o, _ = fx.split_origins(daily, INNER["val_start"], INNER["train_end"],
                                               test_start=fx.TEST_START)
        else:
            daily, fc = fx.features()
            fit_o, val_o, _ = fx.split_origins(daily)
        test_access.guard_origins(fit_o.append(val_o))
        _DATA[key] = (daily, fc, fit_o, val_o)
    return _DATA[key]


def _decay(origins, halflife: float | None) -> np.ndarray:
    """Exponential recency weights (newest origin = 1), rescaled to mean 1."""
    o = pd.DatetimeIndex(pd.to_datetime(origins))
    if not halflife:
        return np.ones(len(o))
    age = (o.max() - o).days.to_numpy(float)
    w = 0.5 ** (age / halflife)
    return w / w.mean()


# ============================================================================ members
def train_lgbm(r: Recipe, daily, fc, fit_o, val_o, seed: int = 42):
    from src.features.build import make_tabular
    from src.models.baselines import LGBMQuantile

    fs, boost = fx.VARIANTS["lgbm"]
    tab = make_tabular(daily, fc, fit_o.append(val_o), target_base=r.lgbm_base)
    params = {**fx.best_params("lgbm"), **dict(r.lgbm_params)}
    m = LGBMQuantile(params, seed=seed, regime_boost=boost, feature_set=fs, extra=r.lgbm_extra, target_base=r.lgbm_base,
                     recency_halflife=r.lgbm_halflife)
    return m.fit(tab, valid_start=val_o[0], refit=False)


def predict_lgbm(m, daily, fc, origins) -> pd.DataFrame:
    from src.features.build import make_tabular

    tab = make_tabular(daily, fc, origins, target_base=getattr(m, "target_base", "ref7"))
    return m.predict_long(tab).assign(model="lgbm")


def train_deep(kind: str, r: Recipe, daily, fc, fit_o, val_o, seed: int = 42, max_epochs: int = 200):
    from src.models.deep import DeepForecaster

    fs, boost = fx.VARIANTS[kind]
    params = {**fx.best_params(kind), **(dict(r.lstm_params) if kind == "lstm" else {})}
    m = DeepForecaster(kind, params, feature_set=fs, regime_boost=boost, max_epochs=max_epochs, seed=seed,
                       enc_days=r.lstm_enc_days if kind == "lstm" else 60,
                       recency_halflife=r.lstm_halflife if kind == "lstm" else None)
    return m.fit(daily, fc, fit_o.append(val_o), refit_all=False, valid_start=val_o[0])


def _key(member: str, r: Recipe, seed: int, inner: bool) -> str:
    s = json.dumps({"m": member, "p": r.part(member), "seed": seed, "inner": inner}, sort_keys=True)
    return f"{member}-{hashlib.sha1(s.encode()).hexdigest()[:12]}"


def member_predictions(member: str, r: Recipe, seed: int, inner: bool = False) -> pd.DataFrame:
    """Validation predictions of one member (cached on disk by its settings and seed)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{_key(member, r, seed, inner)}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from src.utils.device import free_gpu, set_seed

    daily, fc, fit_o, val_o = data(inner, r.lead7_debias)
    set_seed(seed)
    t0 = time.time()
    if member == "lgbm":
        m = train_lgbm(r, daily, fc, fit_o, val_o, seed)
        pred = predict_lgbm(m, daily, fc, val_o)
    else:
        m = train_deep(member, r, daily, fc, fit_o, val_o, seed)
        pred = m.predict_long(daily, fc, val_o).assign(model=member)
        free_gpu()
    pred = pred[pred["origin"].isin(val_o)].dropna(subset=["actual"])
    pred.to_parquet(p, index=False)
    log.info("%s seed %d (%s): %.0f s", member, seed, "inner" if inner else "validation", time.time() - t0)
    return pred


# ============================================================================ ensemble + calibration (OOF)
def _ensemble(rows: pd.DataFrame, members, weights) -> pd.DataFrame:
    from src.postprocess import apply_ensemble

    if len(members) == 2:
        return apply_ensemble(rows, weights, members)
    return apply_multi(rows, weights, members)


def fit_weights(rows: pd.DataFrame, members) -> dict:
    from src.postprocess import fit_ensemble

    return fit_ensemble(rows, members) if len(members) == 2 else fit_multi(rows, members)


MULTI_STEP = 0.1


def fit_multi(rows: pd.DataFrame, members) -> dict:
    """Weights for 3+ members per target x horizon on a simplex grid (step MULTI_STEP), minimising
    the MAPE of the weighted P50. -> {target: {h: tuple(weights)}}."""
    import itertools

    grid = [w for w in itertools.product(np.round(np.arange(0, 1.0001, MULTI_STEP), 2), repeat=len(members))
            if abs(sum(w) - 1) < 1e-9]
    out = {}
    for t in TGT:
        out[t] = {}
        for h in range(1, HORIZON + 1):
            g = rows[(rows["target"] == t) & (rows["h"] == h) & rows["model"].isin(members)]
            piv = g.pivot_table(index=["origin", "date"], columns="model", values="p50").dropna()
            y = g.drop_duplicates(["origin", "date"]).set_index(["origin", "date"])["actual"].reindex(piv.index)
            if len(piv) < 20:
                out[t][h] = tuple([1 / len(members)] * len(members))
                continue
            P = piv[list(members)].to_numpy()
            err = [np.mean(np.abs(P @ np.array(w) - y.to_numpy()) / y.to_numpy()) for w in grid]
            out[t][h] = tuple(grid[int(np.argmin(err))])
    return out


def apply_multi(rows: pd.DataFrame, weights: dict, members, name: str = "ensemble") -> pd.DataFrame:
    keys = ["origin", "date", "h", "target"]
    base = rows[rows["model"] == members[0]][keys + ["actual"]]
    out = base.copy()
    for q in ("p10", "p50", "p90"):
        out[q] = 0.0
    for i, m in enumerate(members):
        x = rows[rows["model"] == m].set_index(keys)
        w = np.array([weights[t][h][i] for t, h in zip(out["target"], out["h"])])
        for q in ("p10", "p50", "p90"):
            out[q] += w * x[q].reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
    out["model"] = name
    return out


def oof(rows: pd.DataFrame, r: Recipe, daily=None) -> pd.DataFrame:
    """Ensemble (+ optional residual correction) and conformal intervals for every member,
    each fold fitted on the other fold. Returns calibrated rows of the members and 'ensemble'."""
    from src.postprocess import apply_conformal, fit_conformal

    # fold A = the window's first calendar month, fold B = the rest (official: Jan | Feb–Mar)
    cut = pd.Timestamp(rows["origin"].min()).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    folds = rows["origin"] < cut
    out = []
    for fit_mask in (~folds, folds):
        fit_rows, ap_rows = rows[fit_mask], rows[~fit_mask]
        w = fit_weights(fit_rows, r.members)
        fit_all = pd.concat([fit_rows, _ensemble(fit_rows, r.members, w)], ignore_index=True)
        ap_all = pd.concat([ap_rows, _ensemble(ap_rows, r.members, w)], ignore_index=True)
        if r.residual:
            corr = ResidualCorrector().fit(fit_all[fit_all["model"] == "ensemble"], daily)
            fit_all = corr.replace(fit_all, daily)
            ap_all = corr.replace(ap_all, daily)
        cal = fit_conformal(fit_all, sorted(fit_all["model"].unique()))
        out.append(apply_conformal(ap_all, cal))
    return pd.concat(out, ignore_index=True)


# ============================================================================ residual correction
RESID_FEATS = ["h", "dow", "is_weekend", "is_holiday", "tmax_ai", "tmin_nr", "precip_ai", "cloud_day_ai"]


class ResidualCorrector:
    """Small LightGBM per target on the ensemble's relative P50 error, from horizon, weekday,
    holiday and the observed-at-origin forecast weather of the target day (known at issue time:
    the daily table holds the archived forecast weather only for dates <= origin, so we use the
    calendar of the target day and the weather columns of make_tabular instead)."""

    params = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, reg_lambda=5.0, verbose=-1)

    def _X(self, ens: pd.DataFrame, daily) -> pd.DataFrame:
        from src.features.build import make_tabular

        origins = pd.DatetimeIndex(sorted(ens["origin"].unique()))
        fc = _DATA["official"][1] if "official" in _DATA else None
        tab = make_tabular(daily, fc, origins)[["origin", "h"] + [c for c in RESID_FEATS if c != "h"]]
        return ens[["origin", "h"]].merge(tab, on=["origin", "h"], how="left")[RESID_FEATS]

    def fit(self, ens: pd.DataFrame, daily) -> "ResidualCorrector":
        import lightgbm as lgb

        self.models = {}
        for t in TGT:
            e = ens[ens["target"] == t].dropna(subset=["actual"])
            X = self._X(e, daily)
            y = (e["actual"] / e["p50"] - 1).to_numpy()
            self.models[t] = lgb.LGBMRegressor(objective="l1", random_state=42, **self.params).fit(X, y)
        return self

    def replace(self, rows: pd.DataFrame, daily) -> pd.DataFrame:
        """Scale the ensemble's quantiles by (1 + predicted relative error)."""
        rows = rows.copy()
        for t in TGT:
            m = (rows["model"] == "ensemble") & (rows["target"] == t)
            if not m.any():
                continue
            f = 1 + self.models[t].predict(self._X(rows[m], daily))
            for q in ("p10", "p50", "p90"):
                rows.loc[m, q] = rows.loc[m, q].to_numpy() * f
        return rows


# ============================================================================ scoring
def score(cal: pd.DataFrame, model: str = "ensemble") -> dict:
    from src.models.common import metrics

    g = cal[cal["model"] == model]
    per = {t: metrics(g[g["target"] == t]) for t in TGT}
    return {"MAPE": float(np.mean([per[t]["MAPE"] for t in TGT])),
            "coverage": float(np.mean([per[t].get("coverage80", np.nan) for t in TGT])),
            **{f"MAPE_{t}": per[t]["MAPE"] for t in TGT}}


def evaluate(r: Recipe, seeds=SEEDS, inner: bool = False) -> dict:
    """Validation score of a recipe: OOF ensemble + each member, per seed and averaged."""
    daily = data(inner)[0]
    per_seed = []
    for s in seeds:
        rows = pd.concat([member_predictions(m, r, s, inner) for m in sorted(set(r.members))], ignore_index=True)
        cal = oof(rows, r, daily)
        res = {"seed": s, "ensemble": score(cal)}
        for m in r.members:
            res[m] = score(cal, m)
        per_seed.append(res)
    agg = {"recipe": r.name, "seeds": list(seeds)}
    for k in per_seed[0]:
        if k == "seed":
            continue
        vals = pd.DataFrame([x[k] for x in per_seed])
        agg[k] = vals.mean().to_dict()
        agg[k]["MAPE_sd"] = float(vals["MAPE"].std(ddof=1)) if len(vals) > 1 else 0.0
        agg[k]["per_seed"] = [float(v) for v in vals["MAPE"]]
    return agg


def log_result(res: dict, r: Recipe, path=None) -> None:
    p = path or OUT / "results.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"at": str(pd.Timestamp.now().floor("s")), "recipe": asdict(r), "result": res},
                            default=str) + "\n")


def baseline() -> Recipe:
    return Recipe()


__all__ = ["Recipe", "evaluate", "member_predictions", "oof", "score", "data", "replace", "field"]


# ============================================================================ final measurement
FINAL = OUT / "final"


def final_run(r: Recipe, dry: bool = False, seed: int = 42) -> dict:
    """Train recipe `r` once (seed 42, as production), fit the ensemble weights and the conformal
    calibration on the validation months, then predict the test period ONCE.

    dry=True stops before the test period and checks that the validation predictions equal the
    experiment cache (the same code path the experiments scored)."""
    from src.postprocess import apply_conformal, apply_ensemble, fit_conformal, fit_ensemble
    from src.utils.device import set_seed

    daily, fc, fit_o, val_o = data(False, r.lead7_debias)
    set_seed(seed)
    models = {"lgbm": train_lgbm(r, daily, fc, fit_o, val_o, seed),
              "lstm": train_deep("lstm", r, daily, fc, fit_o, val_o, seed)}

    def predict(origins):
        return pd.concat([predict_lgbm(models["lgbm"], daily, fc, origins),
                          models["lstm"].predict_long(daily, fc, origins).assign(model="lstm")], ignore_index=True)

    val_pred = predict(val_o)
    val_pred = val_pred[val_pred["origin"].isin(val_o)].dropna(subset=["actual"])
    check = {}
    for m in ("lgbm", "lstm"):
        cached = member_predictions(m, r, seed).sort_values(["origin", "h", "target"]).reset_index(drop=True)
        mine = val_pred[val_pred["model"] == m].sort_values(["origin", "h", "target"]).reset_index(drop=True)
        check[m] = float(np.abs(mine["p50"].to_numpy() - cached["p50"].to_numpy()).max())
    log.info("validation predictions vs experiment cache, max |diff| of P50: %s", check)
    if dry:
        return {"check": check}
    if max(check.values()) > 1e-6 * 1e3:
        raise RuntimeError(f"final pipeline does not reproduce the validated recipe: {check}")

    members = tuple(r.members)
    w = fit_ensemble(val_pred, members)
    val_all = pd.concat([val_pred, apply_ensemble(val_pred, w, members)], ignore_index=True)
    cal = fit_conformal(val_all, sorted(val_all["model"].unique()))

    # ---- the one look at the test period for this recipe
    _, _, test_o = fx.split_origins(daily)
    test_access.guard_origins(test_o, allow_test=True)
    test_pred = predict(test_o)
    test_all = apply_conformal(pd.concat([test_pred, apply_ensemble(test_pred, w, members)], ignore_index=True), cal)
    test_all = test_all[test_all["date"] >= pd.Timestamp(fx.TEST_START)]
    test_access.record("final", f"final recipe '{r.name}' (seed {seed}), trained once, scored on the test period",
                       int(test_all["actual"].notna().sum()), note=json.dumps(asdict(r), default=str))
    FINAL.mkdir(parents=True, exist_ok=True)
    test_all.to_parquet(FINAL / "test_predictions.parquet", index=False)
    apply_conformal(val_all, cal).to_parquet(FINAL / "validation_predictions.parquet", index=False)
    (FINAL / "recipe.json").write_text(json.dumps({"recipe": asdict(r), "ensemble": w, "conformal": cal,
                                                   "check": check}, indent=1, default=str))
    # keep the trained models for the record
    from src.models.baselines import save_lgbm

    root = FINAL / "models"
    import pickle

    root.mkdir(exist_ok=True)
    with open(root / "lgbm.pkl", "wb") as fh:
        pickle.dump(models["lgbm"], fh)
    models["lstm"].save(root=root, tag="lstm")
    _ = save_lgbm
    return {"check": check, "rows": len(test_all)}
