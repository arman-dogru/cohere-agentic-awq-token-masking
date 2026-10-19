"""Tables from results/raw: per-condition metrics (mean over calibration seeds, 95% bootstrap CI over
eval episodes) and paired deltas (same eval items; resample episodes, average over seeds).
Writes results/summary.md and results/summary.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 2000


def item_metrics(items):
    """Per-item arrays. Agentic items are episodes (clusters of spans); metrics are per-episode sums so
    ratios are computed after resampling (ratio of sums, cluster bootstrap)."""
    def agg(eps, kind):
        out = []
        for ep in eps:
            sp = [s for s in ep if kind is None or s["kind"] == kind]
            out.append([sum(s["exact"] for s in sp), len(sp), sum(s["nll"] for s in sp), sum(s["n"] for s in sp)])
        return np.array(out, float)
    return {"call": agg(items["agentic"], "call"), "answer": agg(items["agentic"], "answer"),
            "chat": agg(items["chat"], None), "mmlu": np.array(items["mmlu"], float)}


METRICS = {  # name -> (block, function of summed columns)
    "call_exact": ("call", lambda s: s[0] / s[1]),
    "call_nll_tok": ("call", lambda s: s[2] / s[3]),
    "answer_nll_tok": ("answer", lambda s: s[2] / s[3]),
    "chat_nll_tok": ("chat", lambda s: s[2] / s[3]),
    "mmlu_acc": ("mmlu", lambda s: s),
}


def stat(block_arr, fn, idx=None):
    a = block_arr if idx is None else block_arr[idx]
    return float(fn(a.sum(0)) if a.ndim == 2 else a.mean())


def load():
    runs = defaultdict(dict)
    for f in sorted((ROOT / "results/raw").glob("*.json")):
        tag, s = f.stem.rsplit("__s", 1)
        runs[tag][int(s)] = item_metrics(json.loads(f.read_text())["items"])
    return runs


def boot(runs_a, runs_b=None, metric="call_exact", seed=0, metrics=None):
    """Point estimate and 95% CI. Value per resample = mean over calibration seeds of the metric
    (minus the same for runs_b if given). Seeds of fp/rtn are deterministic (single run)."""
    block, fn = (metrics or METRICS)[metric]
    n = len(next(iter(runs_a.values()))[block])
    rng = np.random.default_rng(seed)

    def val(idx):
        v = np.mean([stat(r[block], fn, idx) for r in runs_a.values()])
        if runs_b is not None:
            v -= np.mean([stat(r[block], fn, idx) for r in runs_b.values()])
        return v
    point = val(np.arange(n))
    bs = np.array([val(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    seed_vals = [stat(r[block], fn) for r in runs_a.values()]
    return {"est": point, "lo": float(lo), "hi": float(hi), "p_le0": float((bs <= 0).mean()),
            "seed_sd": float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0, "n_seeds": len(seed_vals)}


ORDER = ["fp", "rtn", "awq_short", "awq_long_nomask", "awq_long_masked", "awq_long_masked_strict",
         "awq_long_masked_matched", "awq_long_template_only"]
PAIRS = [("awq_long_masked", "awq_short"), ("awq_long_masked", "awq_long_nomask"), ("awq_long_nomask", "awq_short"),
         ("awq_short", "rtn"), ("awq_long_nomask", "rtn"), ("awq_long_masked", "rtn"), ("awq_long_masked_strict", "awq_long_masked"),
         ("awq_long_masked_matched", "awq_long_masked"), ("awq_long_template_only", "awq_long_nomask"),
         ("awq_long_masked_a8", "awq_long_nomask_a8"), ("awq_long_masked_a8", "awq_short_a8"),
         ("awq_long_nomask_a8", "awq_short_a8"), ("rtn_a8", "rtn"), ("fp_a8", "fp"),
         ("rtn_fp8", "rtn"), ("rtn_fp8_channel", "rtn"), ("rtn_fp8_channel", "rtn_fp8")]


def main():
    runs = load()
    tags = [t for t in ORDER if t in runs] + sorted(t for t in runs if t not in ORDER)
    out = {"conditions": {}, "pairs": {}}
    lines = ["| condition | seeds | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for t in tags:
        row = {m: boot(runs[t], metric=m) for m in METRICS}
        out["conditions"][t] = row
        cells = [f"{row[m]['est']:.3f} [{row[m]['lo']:.3f}, {row[m]['hi']:.3f}]" for m in METRICS]
        lines.append(f"| {t} | {row['call_exact']['n_seeds']} | " + " | ".join(cells) + " |")
    lines += ["", "Paired deltas (A − B; same eval items; 95% cluster-bootstrap CI; mean over calibration seeds):", "",
              "| A − B | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for a, b in PAIRS + [(t, "fp") for t in tags if t != "fp"]:
        if a in runs and b in runs:
            row = {m: boot(runs[a], runs[b], m) for m in METRICS}
            out["pairs"][f"{a} - {b}"] = row
            lines.append(f"| {a} − {b} | " + " | ".join(f"{row[m]['est']:+.3f} [{row[m]['lo']:+.3f}, {row[m]['hi']:+.3f}]" for m in METRICS) + " |")
    lines += ["", "Per calibration seed (point estimates on the full eval set; the CIs above resample eval items only,",
              "so seed-to-seed spread is listed here separately):", "",
              "| condition | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for t in tags:
        if len(runs[t]) > 1:
            cells = []
            for m in METRICS:
                block, fn = METRICS[m]
                v = [stat(runs[t][s][block], fn) for s in sorted(runs[t])]
                cells.append(" / ".join(f"{x:.3f}" for x in v) + f" (sd {np.std(v, ddof=1):.3f})")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    (ROOT / "results/summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results/summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
