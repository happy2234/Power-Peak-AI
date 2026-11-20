"""Baselines: seasonal-naive, LightGBM quantile regression, Grid-India's official forecast."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config import PROCESSED, QUANTILES, TARGETS
from src.features.build import tabular_feature_cols
from src.models.common import tab_to_long

log = logging.getLogger(__name__)
TGT = list(TARGETS)


# ============================================================================ seasonal naive
class SeasonalNaive:
    """P50 = same weekday last week. P10/P90 = P50 x empirical quantiles of y(d)/y(d-7) on the
    training period (per target)."""

    name = "seasonal_naive"

    def fit(self, tab: pd.DataFrame) -> "SeasonalNaive":
        self.q = {}
        for k in TGT:
            r = (tab[f"y_{k}"] / tab[f"{k}_lag7"]).replace([np.inf, -np.inf], np.nan).dropna()
            self.q[k] = np.quantile(r, [QUANTILES[0], QUANTILES[2]]) if len(r) > 20 else np.array([0.95, 1.05])
        return self

    def predict(self, tab: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for k in TGT:
            base = (tab[f"{k}_lag7"] * tab[f"ref_{k}"]).to_numpy()
            out[k] = np.stack([base * self.q[k][0], base, base * self.q[k][1]], 1)
        return out

    def predict_long(self, tab):
        return tab_to_long(tab, self.predict(tab), self.name)


# ============================================================================ LightGBM
DEFAULT_LGBM = dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    max_bin=127, verbose=-1)


class LGBMQuantile:
    """Direct multi-horizon LightGBM: one model per target x quantile, horizons pooled with h as
    a feature. Predicts the ratio y(t+h)/ref(t); levels = ratio x ref."""

    name = "lgbm"

    def __init__(self, params: dict | None = None, seed: int = 42, regime_boost: float = 0.0,
                 feature_set: str = "full", extra=(), recency_halflife: float | None = None,
                 target_base: str = "ref7"):
        self.params = {**DEFAULT_LGBM, **(params or {})}
        self.seed = seed
        self.feature_set = feature_set
        self.extra = tuple(extra)                       # feature groups from features.build.EXTRA_GROUPS
        self.recency_halflife = recency_halflife        # days; None = equal weights
        self.target_base = target_base                  # must match make_tabular(target_base=...)
        self.regime_boost = regime_boost     # > 0: up-weight rare weather regimes (regime_weights.py)
        self.models: dict[tuple[str, float], object] = {}

    def fit(self, tab: pd.DataFrame, valid_frac: float = 0.1, valid_start=None,
            refit: bool = True) -> "LGBMQuantile":
        """valid_start: first origin of the validation slice (default: the last `valid_frac`).
        refit=False keeps the models trained on the training slice only (fixed train/test split:
        the validation months are used for early stopping and calibration, never for fitting)."""
        import lightgbm as lgb

        warnings.filterwarnings("ignore", message=".*eval_set.*")
        from src.models.regime_weights import fit_thresholds, regime_weights

        self.features = tabular_feature_cols(tab, self.feature_set, getattr(self, "extra", ()))
        self.regime_th = fit_thresholds(tab) if self.regime_boost else {}
        wts = pd.Series(regime_weights(tab, self.regime_th, self.regime_boost), index=tab.index)
        if getattr(self, "recency_halflife", None):
            # newer origins count more; weights relative to the last origin used for fitting
            age = (tab["origin"].max() - tab["origin"]).dt.days.to_numpy(float)
            dec = 0.5 ** (age / self.recency_halflife)
            wts = wts * dec / dec.mean()
        origins = np.sort(tab["origin"].unique())
        cut = pd.Timestamp(valid_start) if valid_start is not None else origins[int(len(origins) * (1 - valid_frac))]
        self.best_iters, self.cqr = {}, {}
        for k in TGT:
            d = tab.dropna(subset=[f"y_{k}"])
            # time-ordered validation slice, with a horizon-long gap so labels don't overlap
            tr = d[d["origin"] < cut - pd.Timedelta(days=7)]
            va = d[d["origin"] >= cut]
            val_pred = {}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, **self.params)
                m.fit(tr[self.features], tr[f"y_{k}"], sample_weight=wts.loc[tr.index].to_numpy(),
                      eval_set=[(va[self.features], va[f"y_{k}"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                best = max(m.best_iteration_ or self.params["n_estimators"], 50)
                val_pred[q] = m.predict(va[self.features], num_iteration=best)
                if refit:   # refit on all data with the chosen number of trees
                    m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed,
                                          **{**self.params, "n_estimators": best})
                    m.fit(d[self.features], d[f"y_{k}"], sample_weight=wts.loc[d.index].to_numpy())
                else:       # keep the early-stopped model fitted on the training slice only
                    m.set_params(n_estimators=best)
                    m.fit(tr[self.features], tr[f"y_{k}"], sample_weight=wts.loc[tr.index].to_numpy())
                self.models[(k, q)] = m
                self.best_iters[(k, q)] = best
            # conformalized quantile regression (CQR) on the validation slice
            y = va[f"y_{k}"].to_numpy()
            sc = np.maximum(val_pred[QUANTILES[0]] - y, y - val_pred[QUANTILES[-1]])
            n = len(sc)
            self.cqr[k] = float(np.quantile(sc, min(1.0, 0.8 * (n + 1) / n))) if n > 10 else 0.0
        return self

    def predict_ratio(self, tab: pd.DataFrame) -> dict[str, np.ndarray]:
        X = tab[self.features]
        out = {}
        for k in TGT:
            r = np.sort(np.stack([self.models[(k, q)].predict(X) for q in QUANTILES], 1), 1)
            r[:, 0] -= self.cqr.get(k, 0.0)
            r[:, 2] += self.cqr.get(k, 0.0)
            out[k] = np.sort(r, 1)
        return out

    def predict(self, tab: pd.DataFrame) -> dict[str, np.ndarray]:
        r = self.predict_ratio(tab)
        return {k: r[k] * tab[f"ref_{k}"].to_numpy()[:, None] for k in TGT}

    def predict_long(self, tab):
        return tab_to_long(tab, self.predict(tab), self.name)

    # ---------------------------------------------------------------- explanations
    def shap_contributions(self, tab: pd.DataFrame, target: str) -> pd.DataFrame:
        """Per-row SHAP values of the P50 model, converted to target units (MW / MU).

        The model predicts ratio = y/ref, so a SHAP value s (ratio units) is worth s*ref."""
        import shap

        m = self.models[(target, 0.5)]
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(tab[self.features])
        ref = tab[f"ref_{target}"].to_numpy()[:, None]
        out = pd.DataFrame(sv * ref, columns=self.features, index=tab.index)
        out["_base"] = float(np.atleast_1d(expl.expected_value)[0]) * ref[:, 0]
        return out


# ============================================================================ Grid-India
GI_SOLAR_WINDOW = (6.0, 17.5)   # hours used to split the curve into solar / non-solar peaks


def grid_india_forecasts() -> pd.DataFrame:
    """Grid-India week-ahead forecast per target date (latest issue for that date).

    solar_peak / evening_peak: midpoint of the published range (labels) or, for charts without
    text labels, the peak of the digitised curve inside / outside solar hours.
    energy: integral of the 15-min forecast curve (MW x 0.25 h -> MU).
    Columns: date, target, p50, forecast_start, source."""
    fd_path, fc_path = PROCESSED / "forecast_daily.parquet", PROCESSED / "forecast_curve.parquet"
    if not fd_path.exists():
        return pd.DataFrame(columns=["date", "target", "p50", "forecast_start", "source"])
    fd = pd.read_parquet(fd_path)
    rows = []
    lab = fd[fd["label_kind"].isin(["range", "point"]) & fd["curve_ok"].fillna(False).astype(bool)]
    for _, r in lab.iterrows():
        for t, lo, hi in [("solar_peak", "solar_hr_low_mw", "solar_hr_high_mw"),
                          ("evening_peak", "non_solar_hr_low_mw", "non_solar_hr_high_mw")]:
            if pd.notna(r[lo]):
                rows.append((r["date"], t, (r[lo] + r[hi]) / 2, r["forecast_start"], "label"))
    if fc_path.exists():
        c = pd.read_parquet(fc_path).dropna(subset=["forecast_mw"])
        c["date"] = c["timestamp"].dt.normalize()
        hr = c["timestamp"].dt.hour + c["timestamp"].dt.minute / 60
        c["solar"] = (hr >= GI_SOLAR_WINDOW[0]) & (hr < GI_SOLAR_WINDOW[1])
        g = c.groupby(["forecast_start", "date"])
        n = g["forecast_mw"].count()
        energy = (g["forecast_mw"].sum() * 0.25 / 1000)[n >= 90]  # complete days only
        sol = c[c["solar"]].groupby(["forecast_start", "date"])["forecast_mw"].max()
        eve = c[~c["solar"]].groupby(["forecast_start", "date"])["forecast_mw"].max()
        have_label = {(r[3], r[0], r[1]) for r in rows}
        for (fs, d), v in energy.items():
            rows.append((d, "energy", v, fs, "curve"))
        for name, ser in [("solar_peak", sol), ("evening_peak", eve)]:
            for (fs, d), v in ser.items():
                if (fs, d, name) not in have_label:
                    rows.append((d, name, v, fs, "curve"))
    df = pd.DataFrame(rows, columns=["date", "target", "p50", "forecast_start", "source"])
    df = df[df["date"] >= df["forecast_start"]]
    # latest issue per date/target; prefer labels over curve within the same issue
    df["_pri"] = (df["source"] == "label").astype(int)
    df = df.sort_values(["date", "target", "forecast_start", "_pri"])
    return df.drop_duplicates(["date", "target"], keep="last").drop(columns="_pri").reset_index(drop=True)


def grid_india_long(actuals: pd.DataFrame) -> pd.DataFrame:
    """Grid-India forecasts in the common long format (p10/p90 NaN), joined to actuals
    (actuals: date, target, actual)."""
    gi = grid_india_forecasts()
    if gi.empty:
        return pd.DataFrame()
    gi = gi.merge(actuals, on=["date", "target"], how="left")
    return pd.DataFrame({"model": "grid_india", "origin": gi["forecast_start"] - pd.Timedelta(days=1),
                         "date": gi["date"], "h": (gi["date"] - gi["forecast_start"]).dt.days + 1,
                         "target": gi["target"], "p10": np.nan, "p50": gi["p50"], "p90": np.nan,
                         "actual": gi["actual"], "gi_source": gi["source"]})


# ============================================================================ persistence
def save_lgbm(model: LGBMQuantile, train_end: str, tag: str | None = None):
    import datetime as dt
    import pickle

    import yaml

    from src.config import MODELS_DIR

    ts = tag or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = MODELS_DIR / "lgbm" / ts
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    cfg = {"kind": "lgbm", "params": model.params, "features": model.features, "train_end": train_end,
           "feature_set": getattr(model, "feature_set", "full"), "regime_boost": float(model.regime_boost),
           "extra": list(getattr(model, "extra", ())), "recency_halflife": getattr(model, "recency_halflife", None),
           "target_base": getattr(model, "target_base", "ref7"),
           "best_iters": {f"{k}|{q}": int(v) for (k, q), v in model.best_iters.items()},
           "cqr": {k: float(v) for k, v in model.cqr.items()}, "quantiles": list(QUANTILES)}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (MODELS_DIR / "lgbm" / "latest.txt").write_text(ts)
    return out


def load_lgbm(path) -> LGBMQuantile:
    import pickle

    with open(path / "model.pkl", "rb") as fh:
        return pickle.load(fh)
