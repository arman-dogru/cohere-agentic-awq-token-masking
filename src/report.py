"""Held-out-tools tables, free-generation error analysis, spot-check examples and figures.

Reads results/heldout/*.json (src/heldout.py), results/raw/*.json and results/mechanism.json. Writes
results/heldout_summary.{md,json}, results/error_analysis.md, results/heldout_examples.md and
results/figures/{awq_scales,error_categories,main_deltas}.png. Every condition that is present is
reported; conditions with fewer seeds are labelled with their seed count.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from . import summarize as S
from .heldout import CATEGORIES
from .run import ROOT

HO = ROOT / "results/heldout"
FIG = ROOT / "results/figures"
ORDER = S.ORDER
PAIRS = [("awq_long_masked", "awq_long_nomask"), ("awq_long_masked", "awq_short"), ("awq_long_nomask", "awq_short"),
         ("rtn", "fp"), ("awq_short", "rtn"), ("awq_long_nomask", "rtn"), ("awq_long_masked", "rtn")]
GEN_METRICS = {"gen_correct_heldout": ("gen_heldout", lambda s: s), "gen_correct_indist": ("gen_indist", lambda s: s)}
TF_METRICS = {k: S.METRICS[k] for k in ("call_exact", "call_nll_tok", "answer_nll_tok")}
ALL_METRICS = TF_METRICS | GEN_METRICS


def load_heldout():
    runs, raw = defaultdict(dict), defaultdict(dict)
    for f in sorted(HO.glob("*.json")):
        tag, s = f.stem.rsplit("__s", 1)
        d = json.loads(f.read_text())
        m = S.item_metrics({"agentic": d["teacher_forced"], "chat": [], "mmlu": []})
        for split in ("heldout", "indist"):
            m[f"gen_{split}"] = np.array([p["category"] == "correct" for p in d["generation"][split]], float)
        runs[tag][int(s)] = m
        raw[tag][int(s)] = d
    return runs, raw


def fmt(r, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(r['est'])} [{f.format(r['lo'])}, {f.format(r['hi'])}]"


def tables(runs, raw):
    tags = [t for t in ORDER if t in runs]
    out = {"conditions": {}, "pairs": {}, "alpha_agreement": {}}
    n_ep = len(next(iter(next(iter(runs.values())).values()))["call"])
    n_gen = len(next(iter(next(iter(runs.values())).values()))["gen_heldout"])
    L = [f"Held-out tools: {n_ep} teacher-forced episodes over 10 tools never seen in calibration; free generation on "
         f"{n_gen} held-out and {n_gen} in-distribution prompts (greedy, first call after a user request). "
         "95% cluster bootstrap over episodes/prompts, mean over calibration seeds.", "",
         "| condition | seeds | held-out call exact (TF) | held-out call NLL/tok | held-out answer NLL/tok | gen correct, held-out | gen correct, in-dist |",
         "|---|---|---|---|---|---|---|"]
    for t in tags:
        row = {m: S.boot(runs[t], metric=m, metrics=ALL_METRICS) for m in ALL_METRICS}
        out["conditions"][t] = row
        L.append(f"| {t} | {len(runs[t])} | " + " | ".join(fmt(row[m]) for m in ALL_METRICS) + " |")
        out["alpha_agreement"][t] = {s: d["meta"].get("alpha_agreement_with_main_grid") for s, d in raw[t].items()}
    L += ["", "Paired deltas (A − B, same items; only seeds present for both are used):", "",
          "| A − B | " + " | ".join(ALL_METRICS) + " |", "|---|" + "---|" * len(ALL_METRICS)]
    for a, b in PAIRS:
        if a in runs and b in runs:
            sa = {s: v for s, v in runs[a].items() if b in ("fp", "rtn") or s in runs[b]}
            sb = runs[b] if b in ("fp", "rtn") else {s: v for s, v in runs[b].items() if s in sa}
            row = {m: S.boot(sa, sb, m, metrics=ALL_METRICS) for m in ALL_METRICS}
            out["pairs"][f"{a} - {b}"] = row
            L.append(f"| {a} − {b} ({len(sa)} seed{'s' * (len(sa) > 1)}) | " + " | ".join(fmt(row[m], True) for m in ALL_METRICS) + " |")
    L += ["", "AWQ alpha agreement with the main-grid run of the same condition/seed (1.0 = identical scales): " +
          ", ".join(f"{t} s{s}: {v}" for t, d in out["alpha_agreement"].items() for s, v in d.items() if v is not None)]
    return L, out


def error_analysis(raw):
    tags = [t for t in ORDER if t in raw]
    L = ["# Tool-call error analysis (free generation)", "",
         "Greedy generation of the next assistant turn right after a user request (history from earlier rounds included), "
         "parsed and assigned to exactly one category, checked in this order: `no_call` (prose, no `<tool_call>`), "
         "`malformed_json` (tag present, body not a `{name, arguments}` JSON object, incl. truncation), `hallucinated_tool` "
         "(name not offered), `wrong_tool` (another offered tool), `wrong_arg_keys` (missing/extra argument names), "
         "`wrong_arg_type` (value right only as a string, e.g. `\"50\"` vs `50`), `wrong_arg_value`, else `correct`. "
         "Counts are summed over the calibration seeds present (seed count in brackets).", ""]
    counts = {}
    for split, title in (("heldout", "Held-out tools (never in calibration)"), ("indist", "In-distribution tools (main-grid eval episodes)")):
        L += [f"## {title}", "", "| condition | n | " + " | ".join(CATEGORIES) + " | case-only value errors |",
              "|---|---|" + "---|" * (len(CATEGORIES) + 1)]
        for t in tags:
            ps = [p for d in raw[t].values() for p in d["generation"][split]]
            c = Counter(p["category"] for p in ps)
            counts[(split, t)] = {k: c.get(k, 0) / len(ps) for k in CATEGORIES}
            case = sum(bool(p.get("case_only")) for p in ps)
            L.append(f"| {t} [{len(raw[t])}] | {len(ps)} | " + " | ".join(f"{c.get(k, 0)} ({c.get(k, 0) / len(ps):.0%})" for k in CATEGORIES) + f" | {case} |")
        L.append("")
        # transitions: prompts fp gets right that a quantized condition gets wrong, by category (seed 0)
        if "fp" in raw:
            fp = raw["fp"][0]["generation"][split]
            L += ["Where quantization breaks calls the fp model gets right (seed 0; count of fp-correct prompts by the quantized category):", "",
                  "| condition | fp-correct prompts broken | " + " | ".join(CATEGORIES[1:]) + " |", "|---|---|" + "---|" * (len(CATEGORIES) - 1)]
            for t in tags:
                if t == "fp" or 0 not in raw[t]:
                    continue
                q = raw[t][0]["generation"][split]
                br = Counter(b["category"] for a, b in zip(fp, q) if a["category"] == "correct" and b["category"] != "correct")
                fixed = sum(a["category"] != "correct" and b["category"] == "correct" for a, b in zip(fp, q))
                L.append(f"| {t} | {sum(br.values())} (and {fixed} fixed) | " + " | ".join(str(br.get(k, 0)) for k in CATEGORIES[1:]) + " |")
            L.append("")
        L += ["Examples (seed 0, first two per condition and failure category):", ""]
        for t in tags:
            ps = raw[t][min(raw[t])]["generation"][split]
            shown = Counter()
            ex = []
            for p in ps:
                k = p["category"]
                if k != "correct" and shown[k] < 2:
                    shown[k] += 1
                    gold = json.dumps(p["gold"])
                    got = p["output"].strip().replace("\n", " ")[:220]
                    ex.append(f"- **{k}** · request: “{p['request']}” · gold `{gold}` · got `{got}`" +
                              (f" · {p['detail']}" if p.get("detail") else ""))
            if ex:
                L += [f"**{t}**", ""] + ex + [""]
    return L, counts


def examples(raw, n=6):
    tags = [t for t in ORDER if t in raw]
    base = raw[tags[0]][min(raw[tags[0]])]["generation"]["heldout"]
    L = ["# Spot-check: generated answers on held-out-tool prompts", "",
         f"The first {n} held-out prompts, greedy, seed 0 of every condition. Tools offered are listed; none occurs in calibration data.", ""]
    for i in range(min(n, len(base))):
        p = base[i]
        L += [f"## {i + 1}. “{p['request']}”", "", f"offered: {', '.join(p['tools'])} · round {p['round'] + 1} of {p['n_rounds']}", "",
              f"gold: `{json.dumps(p['gold'])}`", "", "| condition | output | category |", "|---|---|---|"]
        for t in tags:
            q = raw[t][min(raw[t])]["generation"]["heldout"][i]
            o = q["output"].strip().replace("\n", " ").replace("|", "\\|")[:260]
            L.append(f"| {t} | `{o}` | {q['category']} |")
        L.append("")
    return L


def figures(counts, raw_tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    # 1) AWQ scale figure: chosen alpha, loss reduction, and how much masking changes the statistic, per layer.
    logs = defaultdict(list)
    for f in sorted((ROOT / "results/raw").glob("*.json")):
        tag, s = f.stem.rsplit("__s", 1)
        if tag.startswith("awq_") and not tag.endswith(("_a8", "_fp8", "_fp8_channel")):
            logs[tag].append(json.loads(f.read_text())["meta"]["awq_log"])
    if logs:
        mech = json.loads((ROOT / "results/mechanism.json").read_text())["rows"] if (ROOT / "results/mechanism.json").exists() else []
        groups = ["qkv", "gateup", "down"]
        fig, ax = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True)
        for j, g in enumerate(groups):
            for tag in [t for t in ORDER if t in logs]:
                A = np.array([[e["best_alpha"] for e in lg if e["group"] == g] for lg in logs[tag]])
                R = np.array([[1 - e["loss_best"] / e["loss_rtn"] for e in lg if e["group"] == g] for lg in logs[tag]])
                x = np.arange(A.shape[1])
                for k, M in ((0, A), (1, R)):
                    ln, = ax[k, j].plot(x, M.mean(0), marker=".", label=f"{tag} ({len(M)} seeds)")
                    ax[k, j].fill_between(x, M.min(0), M.max(0), alpha=0.15, color=ln.get_color())
            ax[0, j].set_title(f"{g}: chosen AWQ exponent α (s = mean|x|^α)")
            ax[1, j].set_title(f"{g}: output-MSE reduction vs RTN (1 − best/α=0)")
            rows = [r for r in mech if r["group"] == g]
            if rows:
                x = [r["layer"] for r in rows]
                ax[2, j].plot(x, [r["cos_all_vs_masked"] for r in rows], marker=".", label="cos(mean|x| all, masked)")
                ax[2, j].plot(x, [r["top1pct_overlap"] for r in rows], marker=".", label="top-1% salient-channel overlap")
                ax[2, j].plot(x, [r["mass_share"]["system"] for r in rows], marker=".", label="|x| mass from system/tool schemas")
                ax[2, j].set_ylim(0, 1.05)
            ax[2, j].set_title(f"{g}: masked vs unmasked statistic (seed 0)")
            ax[2, j].set_xlabel("layer")
        ax[0, 0].legend(fontsize=8)
        ax[2, 0].legend(fontsize=8)
        fig.suptitle("AWQ scales on Qwen2.5-0.5B-Instruct, W4 g128: per layer, mean over calibration seeds (band = min–max)")
        fig.tight_layout()
        fig.savefig(FIG / "awq_scales.png", dpi=120)
        plt.close(fig)
        made.append("awq_scales.png")
    # 2) error categories (stacked bars), held-out vs in-distribution
    if counts:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        cmap = plt.get_cmap("tab10")
        for i, split in enumerate(("heldout", "indist")):
            tags = [t for t in raw_tags if (split, t) in counts]
            bottom = np.zeros(len(tags))
            for k, cat in enumerate(CATEGORIES):
                v = np.array([counts[(split, t)][cat] for t in tags])
                ax[i].bar(tags, v, bottom=bottom, label=cat, color="#4c9a2a" if cat == "correct" else cmap(k))
                bottom += v
            ax[i].set_title({"heldout": "held-out tools", "indist": "in-distribution tools"}[split] + " — free generation")
            ax[i].tick_params(axis="x", rotation=25)
        ax[0].set_ylabel("share of prompts")
        ax[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "error_categories.png", dpi=120)
        plt.close(fig)
        made.append("error_categories.png")
    # 3) main-grid paired deltas on tool-call exact match (forest plot)
    sj = ROOT / "results/summary.json"
    if sj.exists():
        pairs = json.loads(sj.read_text())["pairs"]
        names = [k for k in pairs if not k.endswith(" - fp")] + [k for k in pairs if k.endswith(" - fp")]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1))
        for i, k in enumerate(names):
            r = pairs[k]["call_exact"]
            ax.errorbar(r["est"], i, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", color="k" if r["lo"] > 0 or r["hi"] < 0 else "grey")
        ax.axvline(0, color="r", lw=0.8)
        ax.set_yticks(range(len(names)), [k.replace(" - ", " − ") for k in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ tool-call exact match (paired, 95% cluster-bootstrap CI)")
        fig.tight_layout()
        fig.savefig(FIG / "main_deltas.png", dpi=120)
        plt.close(fig)
        made.append("main_deltas.png")
    return made


def main():
    out = {}
    counts, tags = {}, []
    if HO.exists() and any(HO.glob("*.json")):
        runs, raw = load_heldout()
        tags = [t for t in ORDER if t in raw]
        L, out = tables(runs, raw)
        (ROOT / "results/heldout_summary.md").write_text("\n".join(L) + "\n")
        (ROOT / "results/heldout_summary.json").write_text(json.dumps(out, indent=1))
        E, counts = error_analysis(raw)
        (ROOT / "results/error_analysis.md").write_text("\n".join(E) + "\n")
        (ROOT / "results/heldout_examples.md").write_text("\n".join(examples(raw)) + "\n")
        print("\n".join(L))
    print("figures:", figures(counts, tags))


if __name__ == "__main__":
    main()
