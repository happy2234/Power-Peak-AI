"""Optuna hyper-parameter search (small budget, one trial at a time on the GPU).

Only origins before `BACKTEST_START` are used (the model's own time-ordered validation slice
inside that period is the objective), so the backtest months never influence tuning.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import yaml

from src.config import MODELS_DIR, QUANTILES, TARGETS
from src.utils.device import free_gpu

log = logging.getLogger(__name__)
BACKTEST_START = "2025-11-01"


def tuning_origins(daily):
    from src.features.build import valid_origins

    return valid_origins(daily, start="2023-06-01",
                         end=pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=8))


def suggest(trial, kind: str) -> dict:
    p = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "dropout": trial.suggest_float("dropout", 0.0, 0.4),
         "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256])}
    if kind == "lstm":
        p["hidden"] = trial.suggest_categorical("hidden", [32, 48, 64, 96, 128])
        p["layers"] = trial.suggest_int("layers", 1, 2)
    elif kind == "tft":
        p["d_model"] = trial.suggest_categorical("d_model", [16, 24, 32, 48, 64])
        p["heads"] = trial.suggest_categorical("heads", [1, 2, 4])
    elif kind == "lgbm":
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
             "min_child_samples": trial.suggest_int("min_child_samples", 5, 80, log=True),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    return p


def _lgbm_objective(params, tab, feature_set: str = "full", valid_start=None) -> float:
    """Mean validation pinball (ratio units) over targets x quantiles, last 15 % of origins."""
    import lightgbm as lgb

    from src.features.build import tabular_feature_cols
    from src.models.baselines import DEFAULT_LGBM
    from src.models.common import pinball

    feats = tabular_feature_cols(tab, feature_set)
    o = np.sort(tab["origin"].unique())
    cut = pd.Timestamp(valid_start) if valid_start is not None else o[int(len(o) * 0.85)]
    losses = []
    for k in TARGETS:
        d = tab.dropna(subset=[f"y_{k}"])
        tr, va = d[d["origin"] < cut - pd.Timedelta(days=7)], d[d["origin"] >= cut]
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=42,
                                  **{**DEFAULT_LGBM, **params})
            m.fit(tr[feats], tr[f"y_{k}"], eval_set=[(va[feats], va[f"y_{k}"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            losses.append(pinball(va[f"y_{k}"].to_numpy(), m.predict(va[feats]), q).mean())
    return float(np.mean(losses))


def params_path(kind: str, feature_set: str = "full"):
    return MODELS_DIR / kind / ("best_params.yaml" if feature_set == "full" else f"best_params_{feature_set}.yaml")


def tune(kind: str, n_trials: int = 30, seed: int = 42, feature_set: str = "full") -> dict:
    import optuna

    from src.features.build import load_daily, make_tabular

    daily, fc = load_daily()
    origins = tuning_origins(daily)
    tab = make_tabular(daily, fc, origins) if kind == "lgbm" else None

    def objective(trial):
        params = suggest(trial, kind)
        t0 = time.time()
        try:
            if kind == "lgbm":
                val = _lgbm_objective(params, tab, feature_set)
            else:
                from src.models.deep import DeepForecaster

                m = DeepForecaster(kind, params, max_epochs=150, patience=15, feature_set=feature_set)
                m.fit(daily, fc, origins, trial=trial, refit_all=False)
                val = m.best_val
                trial.set_user_attr("epochs", len(m.history))
                trial.set_user_attr("batch_size_used", m.params.get("batch_size_used"))
                del m
        finally:
            free_gpu()  # torch.cuda.empty_cache() between trials
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True,
                   callbacks=[lambda s, t: log.info("%s trial %d: %s value=%s (best %.4f)", kind, t.number,
                                                     t.state.name, t.value, s.best_value)])
    best = {**study.best_params}
    out = {"kind": kind, "feature_set": feature_set, "best_params": best, "best_value": float(study.best_value),
           "n_trials": len(study.trials), "minutes": round((time.time() - t0) / 60, 1),
           "tuned_on_origins": [str(origins.min().date()), str(origins.max().date())]}
    (MODELS_DIR / kind).mkdir(parents=True, exist_ok=True)
    params_path(kind, feature_set).write_text(yaml.safe_dump(out))
    df = study.trials_dataframe()
    sfx = "" if feature_set == "full" else f"_{feature_set}"
    df.to_csv(MODELS_DIR / kind / f"optuna_trials{sfx}.csv", index=False)
    return out


def best_params(kind: str, feature_set: str = "full") -> dict:
    """Tuned parameters for this feature set (falls back to the full-set parameters)."""
    for p in (params_path(kind, feature_set), params_path(kind)):
        if p.exists():
            return yaml.safe_load(p.read_text())["best_params"]
    return {}
