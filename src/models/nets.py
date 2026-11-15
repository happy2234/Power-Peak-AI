"""PyTorch models: LSTM encoder-decoder and a compact Temporal Fusion Transformer.

Both map (encoder sequence [B, T_enc, F_enc], decoder known-future inputs [B, H, F_dec]) to
quantiles [B, H, n_targets, 3] in standardized ratio space. The quantile head predicts the
median plus two softplus offsets, so P10 <= P50 <= P90 holds by construction.

Why a hand-written TFT instead of pytorch-forecasting/neuralforecast: our samples come from a
leakage-tested window builder with ratio targets, 3 targets x 3 quantiles and a conformal
calibration step; the libraries bring their own dataset/scaling abstractions (and Lightning)
that would have to be bent around that. ~150 lines keep the exact inputs under our control
and expose the variable-selection weights we use for explanations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileHead(nn.Module):
    def __init__(self, d_in: int, n_targets: int):
        super().__init__()
        self.n_targets = n_targets
        self.lin = nn.Linear(d_in, n_targets * 3)

    def forward(self, x):  # [B, H, d] -> [B, H, T, 3]
        o = self.lin(x).view(*x.shape[:-1], self.n_targets, 3)
        mid = o[..., 1]
        lo = mid - F.softplus(o[..., 0]) - 1e-3
        hi = mid + F.softplus(o[..., 2]) + 1e-3
        return torch.stack([lo, mid, hi], -1)


# ============================================================================ LSTM seq2seq
class LSTMSeq2Seq(nn.Module):
    def __init__(self, f_enc: int, f_dec: int, n_targets: int = 3, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.enc_in = nn.Linear(f_enc, hidden)
        self.dec_in = nn.Linear(f_dec, hidden)
        drop = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(dropout)
        self.head = QuantileHead(2 * hidden, n_targets)

    def forward(self, enc, dec):
        _, (h, c) = self.encoder(self.drop(torch.relu(self.enc_in(enc))))
        out, _ = self.decoder(self.drop(torch.relu(self.dec_in(dec))), (h, c))
        ctx = h[-1].unsqueeze(1).expand(-1, dec.shape[1], -1)  # encoder summary at every step
        return self.head(self.drop(torch.cat([out, ctx], -1)))


# ============================================================================ TFT
class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.lin(x).chunk(2, -1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network (TFT building block)."""

    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        return self.norm(self.skip(x) + self.glu(h))


class VariableSelection(nn.Module):
    """Embeds each scalar input separately, then mixes them with softmax weights from a GRN.
    The weights (per time step) are the model's variable importances."""

    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_vars, d_model) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_vars, d_model))
        self.var_grn = GRN(d_model, d_model, dropout=dropout)
        self.weight_grn = GRN(n_vars, d_model, n_vars, dropout=dropout)

    def forward(self, x):  # [B, T, V]
        e = x.unsqueeze(-1) * self.embed + self.bias          # [B, T, V, d]
        e = self.var_grn(e)
        w = torch.softmax(self.weight_grn(x), -1)             # [B, T, V]
        return (w.unsqueeze(-1) * e).sum(-2), w


class TFTLite(nn.Module):
    def __init__(self, f_enc: int, f_dec: int, n_targets: int = 3, d_model: int = 32,
                 heads: int = 2, dropout: float = 0.1, horizon: int = 7):
        super().__init__()
        self.vsn_enc = VariableSelection(f_enc, d_model, dropout)
        self.vsn_dec = VariableSelection(f_dec, d_model, dropout)
        self.lstm_enc = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_dec = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_lstm = GLU(d_model, d_model)
        self.norm_lstm = nn.LayerNorm(d_model)
        self.enrich = GRN(d_model, d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.gate_attn = GLU(d_model, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = GRN(d_model, d_model, dropout=dropout)
        self.head = QuantileHead(d_model, n_targets)
        self.horizon = horizon
        self.last_weights: dict = {}

    def forward(self, enc, dec, return_weights: bool = False):
        xe, we = self.vsn_enc(enc)
        xd, wd = self.vsn_dec(dec)
        he, state = self.lstm_enc(xe)
        hd, _ = self.lstm_dec(xd, state)
        x_in = torch.cat([xe, xd], 1)
        h = self.norm_lstm(x_in + self.gate_lstm(torch.cat([he, hd], 1)))
        h = self.enrich(h)
        T, H = enc.shape[1], dec.shape[1]
        q = h[:, T:]
        # decoder step i may attend to all encoder steps and decoder steps <= i
        mask = torch.ones(H, T + H, dtype=torch.bool, device=enc.device)
        mask[:, :T] = False
        mask[:, T:] = torch.triu(torch.ones(H, H, dtype=torch.bool, device=enc.device), 1)
        a, att = self.attn(q, h, h, attn_mask=mask, need_weights=return_weights)
        z = self.norm_attn(q + self.gate_attn(a))
        out = self.head(self.ff(z))
        if return_weights:
            self.last_weights = {"enc_vars": we.detach(), "dec_vars": wd.detach(),
                                 "attention": att.detach() if att is not None else None}
        return out


def pinball_loss(pred, y, quantiles=(0.1, 0.5, 0.9), weight=None):
    """pred [B,H,T,3], y [B,H,T] with NaN for missing targets (masked out).
    weight [B,H] (optional): per-sample, per-horizon weight (rare weather regimes)."""
    mask = ~torch.isnan(y)
    if weight is not None:
        mask = mask * weight.unsqueeze(-1)
    y0 = torch.nan_to_num(y)
    losses = []
    for i, q in enumerate(quantiles):
        d = y0 - pred[..., i]
        losses.append(torch.maximum(q * d, (q - 1) * d))
    loss = torch.stack(losses, -1).mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


MODELS = {"lstm": LSTMSeq2Seq, "tft": TFTLite}
