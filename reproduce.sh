#!/usr/bin/env bash
# Cold-install reproduction: deps, data (sha256-checked), grid (resumable: finished cells are skipped), tables.
# Usage: ./reproduce.sh [main|ablations|all]   (THREADS=4 by default)
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:-main}
uv sync --frozen
mkdir -p data results/raw logs
[ -f data/mmlu_test.parquet ] || curl -sfL -o data/mmlu_test.parquet https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet
[ -f data/alpaca.parquet ] || curl -sfL -o data/alpaca.parquet https://huggingface.co/api/datasets/tatsu-lab/alpaca/parquet/default/train/0.parquet
(cd data && sha256sum -c SHA256SUMS)
EVAL="--n-agentic 200 --n-chat 150 --n-mmlu 400 --threads ${THREADS:-4}"
run() {  # run <cond> <seed> [extra args]
  local c=$1 s=$2; shift 2
  local tag=$c; for x in "$@"; do case $x in --a8) tag+=_a8;; fp8|fp8_channel) tag+=_$x;; esac; done
  [ -f "results/raw/${tag}__s${s}.json" ] && { echo "skip $tag s$s"; return; }
  uv run python -m src.run --cond "$c" --seed "$s" $EVAL "$@" > "logs/${tag}__s${s}.log" 2>&1
  tail -1 "logs/${tag}__s${s}.log"
}
jobs_main() {
  run fp 0; run rtn 0
  for s in 0 1 2; do run awq_short $s; run awq_long_nomask $s; run awq_long_masked $s; done
}
jobs_ablations() {
  for s in 0 1 2; do run awq_long_masked_strict $s; run awq_long_masked_matched $s; run awq_long_template_only $s; done
  for c in fp rtn; do run $c 0 --a8; done
  for s in 0 1 2; do run awq_short $s --a8; run awq_long_nomask $s --a8; run awq_long_masked $s --a8; done
  run rtn 0 --scale-dtype fp8; run rtn 0 --scale-dtype fp8_channel
}
[ "${SOURCE_ONLY:-0}" = 1 ] && return 0
case $STAGE in
  main) jobs_main;; ablations) jobs_ablations;; all) jobs_main; jobs_ablations;;
esac
uv run python -m src.summarize
