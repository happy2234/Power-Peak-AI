"""Training / inference wrapper for the neural forecasters (GPU when available)."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import ENCODER_DAYS, HORIZON, MODELS_DIR, QUANTILES, REPORTS, TARGETS
from src.features.build import dec_feats, enc_feats, make_sequences
from src.utils.device import autocast, free_gpu, get_device, is_oom, set_seed

log = logging.getLogger(__name__)
TGT = list(TARGETS)

DEFAULTS = {
    "lstm": dict(hidden=64, layers=1, dropout=0.15, lr=2e-3, weight_decay=1e-4, batch_size=64),
    "tft": dict(d_model=32, heads=2, dropout=0.15, lr=2e-3, weight_decay=1e-4, batch_size=64),
}
ARCH_KEYS = {"lstm": ["hidden", "layers", "dropout"], "tft": ["d_model", "heads", "dropout"]}


class DeepForecaster:
    def __init__(self, kind: str, params: dict | None = None, enc_days: int = ENCODER_DAYS,
                 max_epochs: int = 200, patience: int = 20, seed: int = 42, device=None,
                 regime_boost: float = 0.0, feature_set: str = "full", recency_halflife: float | None = None):
        assert kind in DEFAULTS
        self.kind, self.name = kind, kind
        self.params = {**DEFAULTS[kind], **(params or {})}
        self.enc_days, self.max_epochs, self.patience, self.seed = enc_days, max_epochs, patience, seed
        self.device = device or get_device()
        self.history: list[dict] = []
        self.model = None
        self.regime_boost = regime_boost     # > 0: up-weight rare weather regimes in the loss
        self.regime_th: dict = {}
        self.feature_set = feature_set
        self.recency_halflife = recency_halflife   # days; None = equal weights
        self.enc_feats, self.dec_feats = enc_feats(feature_set), dec_feats(feature_set)

    # ------------------------------------------------------------------ data
    def _arrays(self, daily, fc, origins, **kw):
        return make_sequences(daily, fc, origins, self.enc_days, feature_set=self.feature_set, **kw)

    def _fit_scaler(self, a):
        e = a["enc"].reshape(-1, a["enc"].shape[-1])
        d = a["dec"].reshape(-1, a["dec"].shape[-1])
        # near-constant features (e.g. a flag that never fires in training) get sd=1, otherwise
        # an unseen value at inference becomes ~1e6 and overflows fp16
        self.enc_mu, self.enc_sd = e.mean(0), np.where(e.std(0) < 1e-3, 1.0, e.std(0))
        self.dec_mu, self.dec_sd = d.mean(0), np.where(d.std(0) < 1e-3, 1.0, d.std(0))
        self.y_sd = np.nanstd(a["y"] - 1, axis=(0, 1)) + 1e-6

    def _scale(self, a):
        enc = np.clip((a["enc"] - self.enc_mu) / self.enc_sd, -10, 10)
        dec = np.clip((a["dec"] - self.dec_mu) / self.dec_sd, -10, 10)
        y = (a["y"] - 1) / self.y_sd
        return enc.astype(np.float32), dec.astype(np.float32), y.astype(np.float32)

    def _build(self, f_enc, f_dec):
        from src.models.nets import MODELS

        arch = {k: self.params[k] for k in ARCH_KEYS[self.kind]}
        if self.kind == "tft":
            arch["horizon"] = HORIZON
        return MODELS[self.kind](f_enc, f_dec, n_targets=len(TGT), **arch).to(self.device)

    def _weights(self, a):
        """[N, H] loss weights from the decoder's (unscaled) forecast weather."""
        import torch

        from src.models.regime_weights import regime_weights

        N, H, F = a["dec"].shape
        X = pd.DataFrame(a["dec"].reshape(-1, F), columns=self.dec_feats)
        w = regime_weights(X, self.regime_th, self.regime_boost).reshape(N, H)
        hl = getattr(self, "recency_halflife", None)
        if hl:   # newer origins count more (relative to the newest origin in this array)
            o = pd.DatetimeIndex(a["origins"])
            dec = 0.5 ** ((o.max() - o).days.to_numpy(float) / hl)
            w = w * (dec / dec.mean())[:, None]
        return torch.from_numpy(w.astype(np.float32))

    # ------------------------------------------------------------------ training
    def fit(self, daily, fc, origins, val_frac: float = 0.15, verbose: bool = False,
            trial=None, refit_all: bool = True, valid_start=None) -> "DeepForecaster":
        """valid_start: first origin of the validation slice (default: the last `val_frac`).
        The scaler is fitted on the training slice only."""
        import torch

        set_seed(self.seed)
        origins = pd.DatetimeIndex(origins)
        if valid_start is not None:
            val_o = origins[origins >= pd.Timestamp(valid_start)]
        else:
            n_val = max(int(len(origins) * val_frac), 20)
            val_o = origins[-n_val:]
        tr_o = origins[origins <= val_o[0] - pd.Timedelta(days=HORIZON)]  # no label overlap
        a_tr, a_va = self._arrays(daily, fc, tr_o), self._arrays(daily, fc, val_o)
        self._fit_scaler(a_tr)
        if self.regime_boost:
            from src.models.regime_weights import fit_thresholds

            F = a_tr["dec"].shape[-1]
            self.regime_th = fit_thresholds(pd.DataFrame(a_tr["dec"].reshape(-1, F), columns=self.dec_feats))
        Xe, Xd, Y = (torch.from_numpy(x) for x in self._scale(a_tr))
        W = self._weights(a_tr)
        Ve, Vd, VY = (torch.from_numpy(x).to(self.device) for x in self._scale(a_va))

        bs = int(self.params["batch_size"])
        while True:
            try:
                self._train_loop(Xe, Xd, Y, W, Ve, Vd, VY, bs, verbose, trial)
                break
            except RuntimeError as e:
                if is_oom(e) and bs > 8:
                    bs //= 2
                    log.warning("CUDA OOM: retrying with batch_size=%d", bs)
                    self.model = None
                    free_gpu()
                    continue
                raise
        self.params["batch_size_used"] = bs
        self._calibrate(Ve, Vd, VY)
        if refit_all:
            # retrain on train+val for the early-stopped number of epochs, replaying the LR
            # schedule, so the most recent data (the validation slice) is not wasted
            n_best = int(np.argmin([h["val_loss"] for h in self.history])) + 1
            lrs = [h["lr"] for h in self.history[:n_best]]
            a_all = self._arrays(daily, fc, origins)
            Ae, Ad, AY = (torch.from_numpy(x) for x in self._scale(a_all))
            self._refit(Ae, Ad, AY, self._weights(a_all), bs, lrs)
        self.train_end = str(origins.max().date())
        return self

    def _refit(self, Xe, Xd, Y, W, bs, lrs):
        import torch

        from src.models.nets import pinball_loss

        set_seed(self.seed)
        self.model = self._build(Xe.shape[-1], Xd.shape[-1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=lrs[0], weight_decay=self.params["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        g = torch.Generator().manual_seed(self.seed)
        n = len(Xe)
        for epoch, lr in enumerate(lrs, 1):
            for grp in opt.param_groups:
                grp["lr"] = lr
            self.model.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                e, d, y, w = (x[idx].to(self.device, non_blocking=True) for x in (Xe, Xd, Y, W))
                opt.zero_grad(set_to_none=True)
                with autocast(self.device):
                    pred = self.model(e, d)
                loss = pinball_loss(pred.float(), y, QUANTILES, weight=w)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(idx)
            self.history[epoch - 1]["refit_loss"] = tot / n

    def _train_loop(self, Xe, Xd, Y, W, Ve, Vd, VY, bs, verbose, trial):
        import torch

        from src.models.nets import pinball_loss

        set_seed(self.seed)
        self.model = self._build(Xe.shape[-1], Xd.shape[-1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.params["lr"],
                                weight_decay=self.params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        self.history = []
        n = len(Xe)
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            self.model.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                e, d, y, w = (x[idx].to(self.device, non_blocking=True) for x in (Xe, Xd, Y, W))
                opt.zero_grad(set_to_none=True)
                with autocast(self.device):
                    pred = self.model(e, d)
                loss = pinball_loss(pred.float(), y, QUANTILES, weight=w)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(idx)
            val = self._val_loss(Ve, Vd, VY)
            sched.step(val)
            self.history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val,
                                 "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
            if verbose and epoch % 10 == 0:
                log.info("%s epoch %d train %.4f val %.4f", self.kind, epoch, tot / n, val)
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if trial is not None:
                trial.report(val, epoch)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()
            if bad >= self.patience:
                break
        self.model.load_state_dict(best_state)
        self.best_val = best

    def _val_loss(self, Ve, Vd, VY):
        import torch

        from src.models.nets import pinball_loss

        self.model.eval()
        with torch.no_grad(), autocast(self.device):
            p = self.model(Ve, Vd)
        return float(pinball_loss(p.float(), VY, QUANTILES))

    def _calibrate(self, Ve, Vd, VY, alpha: float = 0.2):
        """Conformalized quantile regression: widen/narrow P10-P90 per target so that the
        validation coverage of the interval is 1-alpha."""
        import torch

        self.model.eval()
        with torch.no_grad(), autocast(self.device):
            p = self.model(Ve, Vd).float().cpu().numpy()
        y = VY.cpu().numpy()
        self.cqr = np.zeros(len(TGT), np.float32)
        for k in range(len(TGT)):
            s = np.maximum(p[..., k, 0] - y[..., k], y[..., k] - p[..., k, 2]).ravel()
            s = s[~np.isnan(s)]
            n = len(s)
            if n > 10:
                self.cqr[k] = np.quantile(s, min(1.0, (1 - alpha) * (n + 1) / n))

    # ------------------------------------------------------------------ inference
    def predict_quantiles(self, daily, fc, origins, return_weights=False, **kw):
        """-> (levels [N, H, 3 targets, 3 q], arrays dict)."""
        import torch

        a = self._arrays(daily, fc, origins, **kw)
        e, d, _ = self._scale(a)
        self.model.eval()
        outs, weights = [], []
        with torch.no_grad():
            for i in range(0, len(e), 512):
                et = torch.from_numpy(e[i:i + 512]).to(self.device)
                dt_ = torch.from_numpy(d[i:i + 512]).to(self.device)
                with autocast(self.device):
                    if return_weights and self.kind == "tft":
                        p = self.model(et, dt_, return_weights=True)
                        weights.append({k: v.float().cpu().numpy() for k, v in self.model.last_weights.items()
                                        if v is not None})
                    else:
                        p = self.model(et, dt_)
                outs.append(p.float().cpu().numpy())
        p = np.concatenate(outs)                                   # [N, H, T, 3] standardized
        p[..., 0] -= self.cqr
        p[..., 2] += self.cqr
        p = np.sort(p, -1)
        ratio = 1 + p * self.y_sd[None, None, :, None]
        levels = ratio * a["ref"][:, None, :, None]
        if weights:
            a["weights"] = {k: np.concatenate([w[k] for w in weights]) for k in weights[0]}
        return levels, a

    def predict_long(self, daily, fc, origins, **kw) -> pd.DataFrame:
        levels, a = self.predict_quantiles(daily, fc, origins, **kw)
        o = pd.DatetimeIndex(a["origins"])
        N, H, T = levels.shape[:3]
        origin = np.repeat(o.to_numpy(), H * T)
        h = np.tile(np.repeat(np.arange(1, H + 1), T), N)
        date = origin + pd.to_timedelta(h, unit="D").to_numpy()
        target = np.tile(TGT, N * H)
        q = levels.reshape(-1, 3)
        df = pd.DataFrame({"model": self.name, "origin": origin, "date": date, "h": h, "target": target,
                           "p10": q[:, 0], "p50": q[:, 1], "p90": q[:, 2]})
        act = daily[TGT].stack().rename("actual")
        act.index.names = ["date", "target"]
        return df.join(act, on=["date", "target"])

    # ------------------------------------------------------------------ persistence
    def save(self, root: Path = MODELS_DIR, tag: str | None = None) -> Path:
        import torch

        ts = tag or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = root / self.kind / ts
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        np.savez(out / "scaler.npz", enc_mu=self.enc_mu, enc_sd=self.enc_sd, dec_mu=self.dec_mu,
                 dec_sd=self.dec_sd, y_sd=self.y_sd, cqr=self.cqr)
        cfg = {"kind": self.kind, "params": self.params, "enc_days": self.enc_days,
               "horizon": HORIZON, "targets": TGT, "quantiles": list(QUANTILES),
               "enc_features": self.enc_feats, "dec_features": self.dec_feats, "train_end": self.train_end,
               "feature_set": self.feature_set,
               "best_val_pinball_std": float(self.best_val), "epochs": len(self.history),
               "device": str(self.device), "seed": self.seed, "cqr": self.cqr.tolist(),
               "regime_boost": float(self.regime_boost),
               "recency_halflife": getattr(self, "recency_halflife", None)}
        (out / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg, default=float))))
        hist = pd.DataFrame(self.history)
        hist.to_csv(out / "history.csv", index=False)
        (REPORTS / "training").mkdir(parents=True, exist_ok=True)
        hist.to_csv(REPORTS / "training" / f"{self.kind}_{ts}.csv", index=False)
        (root / self.kind / "latest.txt").write_text(ts)
        return out

    @classmethod
    def load(cls, path: Path, device=None) -> "DeepForecaster":
        import torch

        cfg = yaml.safe_load((path / "config.yaml").read_text())
        obj = cls(cfg["kind"], cfg["params"], enc_days=cfg["enc_days"], device=device,
                  regime_boost=cfg.get("regime_boost", 0.0), feature_set=cfg.get("feature_set", "full"),
                  recency_halflife=cfg.get("recency_halflife"))
        if cfg["enc_features"] != obj.enc_feats or cfg["dec_features"] != obj.dec_feats:
            raise ValueError(f"{path}: feature set changed since this model was trained; retrain")
        s = np.load(path / "scaler.npz")
        for k in ["enc_mu", "enc_sd", "dec_mu", "dec_sd", "y_sd", "cqr"]:
            setattr(obj, k, s[k])
        obj.model = obj._build(len(obj.enc_feats), len(obj.dec_feats))
        obj.model.load_state_dict(torch.load(path / "model.pt", map_location=obj.device))
        obj.model.eval()
        obj.train_end = cfg["train_end"]
        obj.best_val = cfg.get("best_val_pinball_std", np.nan)
        hist = path / "history.csv"
        obj.history = pd.read_csv(hist).to_dict("records") if hist.exists() else []
        return obj
