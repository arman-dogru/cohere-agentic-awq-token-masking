"""Quantize (fp / RTN / AWQ with a calibration set and role mask) and evaluate one condition.

python -m src.run --cond awq_long_masked --seed 0
Writes results/raw/<tag>__s<seed>.json with per-item metrics (for paired bootstrap) and AWQ search logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data as D
from . import quant as Q

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

# condition -> (calibration source, roles kept in AWQ stats + loss (None = all), options)
CONDS = {
    "fp": None,
    "rtn": None,
    "awq_short": ("short", None, {}),
    "awq_long_nomask": ("agentic", None, {}),
    "awq_long_masked": ("agentic", {"user", "assistant", "tool_result"}, {}),
    "awq_long_masked_strict": ("agentic", {"user", "assistant"}, {}),
    "awq_long_masked_matched": ("agentic", {"user", "assistant", "tool_result"}, {"match_effective": True}),
    "awq_long_template_only": ("agentic", {"system", "template"}, {}),
}
CALIB_TOKENS = 16384
SEQ_LEN = 2048


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    model.eval()
    return tok, model


def alpaca_rows():
    t = pq.read_table(ROOT / "data/alpaca.parquet").to_pylist()
    return [r for r in t if 20 <= len(r["output"]) <= 1200]


def short_chat_text(tok, row):
    user = row["instruction"] + (("\n\n" + row["input"]) if row["input"] else "")
    return D.render(tok, [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}])


def build_calib(tok, source, keep_roles, seed, match_effective=False, budget=CALIB_TOKENS):
    """(ids[1,L], keep[1,L]) batches. The budget counts raw tokens, or kept tokens if match_effective.
    Calibration draws use seeds 1000+seed; the eval set uses 424242, and alpaca rows are split in half."""
    r = random.Random(1000 + seed)
    if source == "short":
        rows = alpaca_rows()
        idx = list(range(len(rows) // 2))
        r.shuffle(idx)
    batches, raw, eff, counts, i = [], 0, 0, np.zeros(len(D.ROLES), int), 0
    while (eff if match_effective else raw) < budget:
        if source == "short":
            text = short_chat_text(tok, rows[idx[i]])
        else:
            ep = D.sample_episode(r)
            text = D.render(tok, D.episode_messages(ep), tools=ep.tools)
        i += 1
        ids, roles = D.tokenize_with_roles(tok, text, SEQ_LEN)
        roles = np.array(roles)
        counts += np.bincount(roles, minlength=len(D.ROLES))
        keep = np.ones(len(ids), bool) if keep_roles is None else np.isin(roles, [D.ROLE_ID[k] for k in keep_roles])
        batches.append((torch.tensor([ids]), torch.tensor(keep)[None]))
        raw += len(ids)
        eff += int(keep.sum())
    return batches, {"n_seq": len(batches), "raw_tokens": raw, "kept_tokens": eff,
                     "role_tokens": {k: int(v) for k, v in zip(D.ROLES, counts)}}


def eval_sets(tok, n_agentic, n_chat):
    r = random.Random(424242)
    agentic = []
    for _ in range(n_agentic):
        ep = D.sample_episode(r)
        agentic.append(D.render(tok, D.episode_messages(ep), tools=ep.tools))
    rows = alpaca_rows()
    held = rows[len(rows) // 2:]
    rr = random.Random(7)
    chat = [short_chat_text(tok, held[j]) for j in rr.sample(range(len(held)), n_chat)]
    return agentic, chat


def spans(roles, want):
    out, cur = [], []
    for i, ro in enumerate(roles):
        if ro == want:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def score_text(model, tok, text):
    """Teacher-forced metrics on every assistant span. A span is greedy-exact iff the argmax equals the
    gold token at every position, i.e. greedy decoding from that prefix reproduces the gold exactly."""
    ids, roles = D.tokenize_with_roles(tok, text)
    x = torch.tensor([ids])
    logits = model(input_ids=x).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    pred = logits.argmax(-1)
    res = []
    for sp in spans(roles, D.ROLE_ID["assistant"]):
        pos = torch.tensor(sp)
        tgt, prev = x[0, pos], pos - 1
        ok = pred[prev] == tgt
        kind = "call" if "<tool_call>" in tok.decode(tgt) else "answer"
        res.append({"kind": kind, "n": len(sp), "nll": float(-logp[prev, tgt].sum()), "correct": int(ok.sum()), "exact": bool(ok.all())})
    return res


@torch.no_grad()
def mmlu(model, tok, n):
    t = pq.read_table(ROOT / "data/mmlu_test.parquet").to_pylist()
    items = [t[i] for i in random.Random(11).sample(range(len(t)), n)]
    letters = ["A", "B", "C", "D"]
    lid = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    out = []
    for it in items:
        q = it["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"])) + "\nAnswer with the letter only."
        p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        lg = model(input_ids=torch.tensor([tok.encode(p, add_special_tokens=False)])).logits[0, -1]
        out.append(int(int(lg[lid].argmax()) == it["answer"]))
    return out


def tag_of(cond, a8, scale_dtype):
    return cond + ("_a8" if a8 else "") + ("" if scale_dtype == "none" else "_" + scale_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-agentic", type=int, default=200)
    ap.add_argument("--n-chat", type=int, default=150)
    ap.add_argument("--n-mmlu", type=int, default=400)
    ap.add_argument("--n-loss-tokens", type=int, default=1024)
    ap.add_argument("--a8", action="store_true", help="also fake-quantize linear inputs to int8 per token")
    ap.add_argument("--scale-dtype", choices=["none", "fp8", "fp8_channel"], default="none")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tok, model = load_model()
    meta = {"cond": a.cond, "seed": a.seed, "a8": a.a8, "scale_dtype": a.scale_dtype, "model": MODEL,
            "revision": REVISION, "torch": torch.__version__, "threads": a.threads, "n_loss_tokens": a.n_loss_tokens}
    qkw = {} if a.scale_dtype == "none" else {"scale_dtype": torch.float8_e4m3fn, "channel_scale": a.scale_dtype == "fp8_channel"}
    if a.cond == "rtn":
        Q.rtn_model(model, **qkw)
    elif a.cond != "fp":
        src, keep, opt = CONDS[a.cond]
        batches, meta["calib"] = build_calib(tok, src, keep, a.seed, opt.get("match_effective", False))
        stats = Q.Collector(model, a.n_loss_tokens, seed=a.seed).run(model, batches)
        meta["awq_log"] = []
        Q.awq_model(model, stats, log=meta["awq_log"], **qkw)
        del stats
    meta["t_quant_s"] = round(time.time() - t0, 1)
    aq = Q.ActQuant(model) if a.a8 else None
    agentic, chat = eval_sets(tok, a.n_agentic, a.n_chat)
    meta["eval_sha"] = hashlib.sha256(("".join(agentic) + "".join(chat)).encode()).hexdigest()[:16]
    res = {"agentic": [score_text(model, tok, t) for t in agentic],
           "chat": [score_text(model, tok, t) for t in chat],
           "mmlu": mmlu(model, tok, a.n_mmlu) if a.n_mmlu else []}
    if aq:
        aq.remove()
    meta["t_total_s"] = round(time.time() - t0, 1)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = tag_of(a.cond, a.a8, a.scale_dtype)
    (out / f"{tag}__s{a.seed}.json").write_text(json.dumps({"meta": meta, "items": res}))
    calls = [s for ep in res["agentic"] for s in ep if s["kind"] == "call"]
    chat_tok = sum(s["n"] for ep in res["chat"] for s in ep)
    print(tag, a.seed, "call_exact", round(float(np.mean([s["exact"] for s in calls])), 3),
          "chat_nll/tok", round(sum(s["nll"] for ep in res["chat"] for s in ep) / chat_tok, 4),
          "mmlu", round(float(np.mean(res["mmlu"])), 3) if res["mmlu"] else None, "t", meta["t_total_s"], flush=True)


if __name__ == "__main__":
    main()
