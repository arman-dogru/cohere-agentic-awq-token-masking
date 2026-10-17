"""Minimal CPU AWQ (activation-aware weight quantization) with token masking.

W4 asymmetric group-wise fake quantization (group 128 along input features), an RTN baseline, and the
AWQ scale search of Lin et al. (2023) with AutoAWQ's grid (s = mean|x|^a, a in {0, 0.05, ..., 0.95}).
Both the activation statistic mean|x| and the reconstruction loss use only tokens whose mask is 1:
that is the token masking of Cohere's W4A8 post (long system/tool-template tokens should not dominate
calibration).

Scales are folded into the preceding RMSNorm (q/k/v, gate/up) or into up_proj rows (down_proj), so
the fp model's function is unchanged before rounding. o_proj is RTN only: with GQA (14 query heads,
2 kv heads) v_proj rows cannot absorb a per-input-channel scale of o_proj. No clipping search.
"""
from __future__ import annotations

import torch

GROUP = 128


def quantize_w(w: torch.Tensor, bits=4, group=GROUP, scale_dtype=None, channel_scale=False):
    """Fake-quantize w [out, in] group-wise asymmetric. scale_dtype simulates storing the group scales
    as e.g. torch.float8_e4m3fn: a naive cast, or with channel_scale=True fp8 ratios to an fp32
    per-output-channel scale (the post's channel-scale fix)."""
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group, group)
    wmax, wmin = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = (wmax - wmin).clamp(min=1e-8) / qmax
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
    if scale_dtype is not None:
        if channel_scale:
            row = scale.amax(1, keepdim=True)
            ratio = (scale / row * 448.0).to(scale_dtype).to(torch.float32) / 448.0
            scale = (ratio * row).clamp(min=1e-12)
        else:
            scale = scale.to(scale_dtype).to(torch.float32).clamp(min=1e-12)
    return ((q - zero) * scale).reshape(out_f, in_f)


def group_scales(w: torch.Tensor, bits=4, group=GROUP):
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group, group)
    return (g.amax(-1) - g.amin(-1)).clamp(min=1e-8) / (2 ** bits - 1)


def _linears(layer):
    a, m = layer.self_attn, layer.mlp
    return [a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj]


@torch.no_grad()
def rtn_model(model, **kw):
    for layer in model.model.layers:
        for lin in _linears(layer):
            lin.weight.data = quantize_w(lin.weight.data, **kw)


class Collector:
    """Forward pre-hooks that gather, per AWQ group input, (a) mean |x| over masked-in tokens and
    (b) a uniform random subsample of masked-in token rows for the reconstruction loss."""

    KEYS = ("qkv", "gateup", "down")

    def __init__(self, model, n_loss_tokens=1024, seed=0):
        self.n_loss, self.gen = n_loss_tokens, torch.Generator().manual_seed(seed)
        self.mask, self.sel = None, None
        self.stats, self.rows, self.handles = {}, {}, []
        for i, layer in enumerate(model.model.layers):
            mods = (layer.self_attn.q_proj, layer.mlp.gate_proj, layer.mlp.down_proj)
            for key, mod in zip(self.KEYS, mods):
                self.handles.append(mod.register_forward_pre_hook(self._hook((i, key))))

    def _hook(self, k):
        def fn(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            xs = x[self.mask.reshape(-1)]
            s, n = self.stats.get(k, (0.0, 0))
            self.stats[k] = (s + xs.abs().sum(0), n + xs.shape[0])
            self.rows.setdefault(k, []).append(xs[self.sel])
        return fn

    @torch.no_grad()
    def run(self, model, batches):
        """batches: list of (input_ids [1,L], keep-mask bool [1,L])."""
        total = sum(int(m.sum()) for _, m in batches)
        perm = torch.randperm(total, generator=self.gen)[: self.n_loss]
        off = 0
        for ids, m in batches:
            n = int(m.sum())
            self.sel = perm[(perm >= off) & (perm < off + n)] - off
            self.mask = m
            model(input_ids=ids)
            off += n
        for h in self.handles:
            h.remove()
        return {k: (s / max(n, 1), torch.cat(self.rows[k], 0)) for k, (s, n) in self.stats.items()}


def search_scale(xmean, x, linears, n_grid=20):
    """Best per-input-channel scale for linears sharing input x; returns (scale, losses per alpha)."""
    w_cat = torch.cat([l.weight.data for l in linears], 0)
    ref = x @ w_cat.T
    best, best_s, losses = float("inf"), None, []
    for i in range(n_grid):
        s = xmean.clamp(min=1e-4).pow(i / n_grid)
        s = s / (s.max() * s.min()).sqrt()
        wq = quantize_w(w_cat * s) / s
        loss = (x @ wq.T - ref).pow(2).mean().item()
        losses.append(loss)
        if loss < best:
            best, best_s = loss, s
    return best_s, losses


@torch.no_grad()
def awq_model(model, stats, n_grid=20, log=None, **qkw):
    """Fold AWQ scales into the model (function-preserving), then RTN-quantize every linear."""
    for i, layer in enumerate(model.model.layers):
        a, m = layer.self_attn, layer.mlp
        groups = [("qkv", layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj]),
                  ("gateup", layer.post_attention_layernorm, [m.gate_proj, m.up_proj]),
                  ("down", m.up_proj, [m.down_proj])]
        for key, prev, lins in groups:
            xmean, x = stats[(i, key)]
            s, losses = search_scale(xmean, x, lins, n_grid)
            if log is not None:
                log.append({"layer": i, "group": key, "best_alpha": losses.index(min(losses)) / n_grid,
                            "loss_rtn": losses[0], "loss_best": min(losses)})
            for l in lins:
                l.weight.data *= s
            if key == "down":
                prev.weight.data /= s[:, None]
            else:
                prev.weight.data /= s
    rtn_model(model, **qkw)


class ActQuant:
    """Dynamic per-token symmetric int8 fake quantization of every linear input (the A8 of W4A8)."""

    def __init__(self, model, bits=8):
        q = 2 ** (bits - 1) - 1

        def fn(mod, args):
            x = args[0]
            s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / q
            return (torch.round(x / s).clamp(-q, q) * s,)
        self.handles = [lin.register_forward_pre_hook(fn) for layer in model.model.layers for lin in _linears(layer)]

    def remove(self):
        for h in self.handles:
            h.remove()
