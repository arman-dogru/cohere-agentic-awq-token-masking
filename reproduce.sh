#!/usr/bin/env bash
# Cold-install reproduction: deps, data (sha256-checked), grid (resumable: finished cells are skipped), tables, figures.
# Usage: ./reproduce.sh [main|ablations|heldout|all|smoke]   (THREADS=4 by default)
#   heldout: held-out-tools split + free-generation error analysis (src/heldout.py), then src/report.py
#   smoke:   unit tests + a few-item run of src.run and src.heldout into results/smoke (minutes, cold-install check)
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:-main}
uv sync --frozen
mkdir -p data results/raw logs
fetch() {  # fetch <file> <url>: retried, and written to .part first so an interrupted download never looks finished
  [ -f "data/$1" ] && return
  curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "data/$1.part" "$2" && mv "data/$1.part" "data/$1"
}
fetch mmlu_test.parquet https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet
fetch alpaca.parquet https://huggingface.co/api/datasets/tatsu-lab/alpaca/parquet/default/train/0.parquet
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
ho() {  # ho <cond> <seed>: held-out tools (teacher-forced) + free generation on held-out and in-distribution prompts
  local c=$1 s=$2
  [ -f "results/heldout/${c}__s${s}.json" ] && { echo "skip heldout $c s$s"; return; }
  uv run python -m src.heldout --cond "$c" --seed "$s" --threads "${THREADS:-4}" > "logs/heldout_${c}__s${s}.log" 2>&1
  tail -1 "logs/heldout_${c}__s${s}.log"
}
jobs_heldout() {
  ho fp 0; ho rtn 0
  for s in 0 1 2; do ho awq_short $s; ho awq_long_nomask $s; ho awq_long_masked $s; done
}
jobs_smoke() {
  uv run pytest -q
  uv run python -m src.run --cond rtn --seed 0 --n-agentic 3 --n-chat 3 --n-mmlu 5 --threads "${THREADS:-4}" --out results/smoke
  uv run python -m src.heldout --cond rtn --seed 0 --n-episodes 3 --n-gen 2 --threads "${THREADS:-4}" --out results/smoke
}
[ "${SOURCE_ONLY:-0}" = 1 ] && return 0
case $STAGE in
  main) jobs_main;; ablations) jobs_ablations;; heldout) jobs_heldout;; all) jobs_main; jobs_ablations; jobs_heldout;;
  smoke) jobs_smoke
         # regenerate README figures from whatever results exist (smoke itself writes to results/smoke)
         if [ -f results/summary.json ]; then uv run python -m src.figures; fi
         exit 0;;
  *) echo "unknown stage $STAGE" >&2; exit 2;;
esac
uv run python -m src.summarize
uv run python -m src.figures
if [ "$STAGE" = heldout ] || [ "$STAGE" = all ]; then uv run python -m src.report; fi
