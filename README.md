# Agentic AWQ token masking: a CPU quality-recovery study

> **Status: work in progress (study-v1 branch).** Method, harness, tests and the reproduction script
> are in place; the main grid is running. Numbers below are filled in from `results/summary.md` as cells finish.

Cohere's W4A8 post ([vLLM integration and quality-recovery techniques](https://cohere.com/blog/vllm-integration-and-quality-recovery-techniques-explained))
lists *AWQ token masking* as one of the fixes that made 4-bit weights work on long agentic workloads:
when you calibrate on agentic traces, the long system prompt and tool-schema tokens dominate the
activation statistics, so the scales AWQ picks fit the template rather than the content the model
actually has to produce. This repo reproduces that **diagnosis** on a small open model on CPU, with
no Hopper kernels: W4 is simulated by fake quantization (dequantized weights), which preserves the
numerics of the rounding while skipping the speed-up.

## Question

On Qwen2.5-0.5B-Instruct quantized to W4 (group 128), does calibrating AWQ on long agentic traces
**with system/tool-template tokens masked out** recover more tool-calling quality than (a) the usual
short-chat calibration set and (b) the same agentic traces without masking?

## Setup

| piece | choice |
|---|---|
| model | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576`, fp32 on CPU (4 threads) |
| weights | W4 asymmetric, group 128, round-to-nearest after AWQ scaling (`src/quant.py`) |
| AWQ | minimal reimplementation: per group input, `s = mean|x|^α`, α ∈ {0, 0.05, …, 0.95}, pick α minimising output MSE on 1,024 calibration token rows; scales folded into RMSNorm (q/k/v, gate/up) or `up_proj` rows (`down_proj`). `o_proj` is RTN only (GQA prevents folding into `v_proj`). No clipping search. |
| token masking | the statistic `mean|x|` **and** the reconstruction rows use only tokens whose role is kept |
| calibration budget | 16,384 raw tokens per condition (sequences ≤ 2,048), 3 calibration seeds |
| agentic traces | synthetic, `src/data.py`: 12 tools with JSON schemas, 6–12 offered per episode, 1–3 rounds of user request → `<tool_call>` → `<tool_response>` → answer, rendered with Qwen's own tool chat template. In the calibration set ~80% of tokens are system/tool-schema tokens. |
| short-chat calibration | Alpaca rows (first half), rendered with the chat template |

Every token carries a role label (`template`, `system`, `user`, `tool_result`, `assistant`), derived
from character offsets of the rendered chat (tested in `tests/test_core.py`).

### Conditions

| condition | calibration | roles in AWQ statistics |
|---|---|---|
| `fp` | – | – (fp32 reference) |
| `rtn` | – | – (W4 round-to-nearest, no AWQ) |
| `awq_short` | short chat | all |
| `awq_long_nomask` | agentic traces | all |
| `awq_long_masked` | agentic traces | user + assistant + tool_result (system and template masked) |
| ablation `awq_long_masked_strict` | agentic traces | user + assistant only |
| ablation `awq_long_masked_matched` | agentic traces | as masked, but 16,384 **kept** tokens (more episodes) |
| ablation `awq_long_template_only` | agentic traces | system + template only (the inverse mask) |
| ablation `*_a8` | as above | plus dynamic per-token int8 activations (the A8 of W4A8) |
| ablation `rtn_fp8`, `rtn_fp8_channel` | – | group scales stored as FP8 e4m3: naive cast vs fp8 ratio to an fp32 per-channel scale (the post's channel-scale fix) |

### Metrics (all paired: every condition sees the same evaluation items)

* **Tool-call exact match** (primary, agentic): 200 held-out synthetic episodes (seed disjoint from
  calibration). For every assistant `<tool_call>` span, teacher-forced: the call is exact iff the
  argmax equals the gold token at every position, which is the same as greedy decoding reproducing
  the gold call from that prefix. Later calls see the earlier rounds, so prompts are 1–3k tokens.
* Tool-call NLL per token, and final-answer NLL per token (answers copy values from tool results).
* Short-chat NLL per token: 150 held-out Alpaca answers (second half of Alpaca).
* MMLU-lite: 400 random MMLU test questions, zero-shot, letter argmax.

Uncertainty: 95% cluster bootstrap over evaluation episodes (2,000 resamples; ratio of sums per
resample), averaged over the 3 calibration seeds; paired deltas resample the same items for both
conditions. Seed-to-seed SD is reported separately.

## Results

*Pending: the main grid (fp, rtn, 3 AWQ conditions × 3 seeds) is running.* See `results/summary.md`.

## Reproduce

```bash
./reproduce.sh            # main grid  (~1 h on 8 CPU cores with two workers; resumable)
./reproduce.sh ablations  # ablations
uv run pytest -q          # unit tests
uv run python -m src.mechanism   # where the activation statistic comes from, by role
```

`reproduce.sh` does `uv sync --frozen` (CPU torch from the PyTorch index, everything pinned in
`uv.lock`), downloads MMLU and Alpaca parquet files and checks their sha256, then runs every cell
that has no result yet and writes the tables.

## Limitations (so far)

* Fake quantization: no INT4/FP8 kernels, so no latency numbers; only quality.
* A 0.5B model and a synthetic, templated agentic distribution; calibration and evaluation share the
  12 tools and request templates (different seeds and values). A held-out-tools split is planned.
* Minimal AWQ (no clipping search, `o_proj` not scaled); absolute recovery will differ from AutoAWQ /
  llm-compressor, but every condition uses the same quantizer, so the comparisons are like for like.

## Layout

`src/data.py` traces and role labels · `src/quant.py` quantizer, AWQ, masking, A8, FP8 scales ·
`src/run.py` one condition end to end · `src/summarize.py` tables and paired bootstrap ·
`src/mechanism.py` activation-statistic decomposition · `tests/` unit tests · `reproduce.sh`.

## License

MIT
