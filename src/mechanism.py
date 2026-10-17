"""Why masking matters: where AWQ's activation statistic mean|x| comes from, by token role.

One fp pass over the agentic calibration set (seed 0). For every layer and AWQ group input, record
per-role sums of |x| per channel. Reports (a) the share of tokens and of total |x| mass from each role,
(b) cosine similarity and top-1% salient-channel overlap between the unmasked statistic and the
masked one (user/assistant/tool_result only), and (c) the contribution of the first token (attention
sink). Writes results/mechanism.json and results/mechanism.md.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from . import data as D
from .run import ROOT, build_calib, load_model

KEEP = [D.ROLE_ID[r] for r in ("user", "assistant", "tool_result")]


def main():
    torch.set_num_threads(4)
    tok, model = load_model()
    role_list = []
    batches, summ = build_calib(tok, "agentic", None, 0, role_list=role_list)
    nR = len(D.ROLES)
    acc = {}
    cur = {}

    def hook(k):
        def fn(mod, args):
            x = args[0][0].abs().float()                          # [L, C]
            r = cur["roles"]
            s = acc.setdefault(k, {"sum": torch.zeros(nR, x.shape[1]), "first": torch.zeros(x.shape[1]), "n": torch.zeros(nR)})
            s["sum"].index_add_(0, r, x)
            s["n"] += torch.bincount(r, minlength=nR).float()
            s["first"] += x[0]
        return fn
    hs = []
    for i, layer in enumerate(model.model.layers):
        for key, mod in (("qkv", layer.self_attn.q_proj), ("gateup", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)):
            hs.append(mod.register_forward_pre_hook(hook((i, key))))
    with torch.no_grad():
        for (ids, _), roles in zip(batches, role_list):
            cur["roles"] = roles
            model(input_ids=ids)
    for h in hs:
        h.remove()
    rows = []
    for (i, key), s in sorted(acc.items()):
        tot = s["sum"].sum(0)
        all_mean = tot / s["n"].sum()
        msk_mean = s["sum"][KEEP].sum(0) / s["n"][KEEP].sum()
        cos = float(torch.nn.functional.cosine_similarity(all_mean, msk_mean, dim=0))
        k = max(1, int(0.01 * len(tot)))
        top_a, top_m = set(all_mean.topk(k).indices.tolist()), set(msk_mean.topk(k).indices.tolist())
        mass = (s["sum"].sum(1) / tot.sum()).tolist()
        rows.append({"layer": i, "group": key, "cos_all_vs_masked": cos, "top1pct_overlap": len(top_a & top_m) / k,
                     "mass_share": dict(zip(D.ROLES, mass)), "first_token_mass": float(s["first"].sum() / tot.sum()),
                     "ratio_max_channel": float((all_mean / msk_mean.clamp(min=1e-6)).max())})
    tok_share = dict(zip(D.ROLES, (np.array(list(summ["role_tokens"].values())) / summ["raw_tokens"]).round(4).tolist()))
    out = {"calib": summ, "token_share": tok_share, "rows": rows}
    (ROOT / "results/mechanism.json").write_text(json.dumps(out, indent=1))
    md = [f"Token share by role (agentic calibration, seed 0, {summ['raw_tokens']} tokens): " +
          ", ".join(f"{k} {v:.1%}" for k, v in tok_share.items()), "",
          "| layer | group | cos(all, masked) | top-1% channel overlap | |x| mass from system | from kept roles | first-token mass |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["layer"] in (0, 1, 2, 6, 12, 18, 22, 23):
            kept = sum(r["mass_share"][x] for x in ("user", "assistant", "tool_result"))
            md.append(f"| {r['layer']} | {r['group']} | {r['cos_all_vs_masked']:.3f} | {r['top1pct_overlap']:.2f} | "
                      f"{r['mass_share']['system']:.1%} | {kept:.1%} | {r['first_token_mass']:.1%} |")
    md.append("")
    md.append(f"Median over all {len(rows)} groups: cos {np.median([r['cos_all_vs_masked'] for r in rows]):.3f}, "
              f"top-1% overlap {np.median([r['top1pct_overlap'] for r in rows]):.2f}.")
    (ROOT / "results/mechanism.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
