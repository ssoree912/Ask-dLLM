#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILY="${1:?usage: run_oc_mc.sh <llada|dream> <keep_ratio> [checkpoint] [dataset]}"
export FUTURE_DLLM_KEEP_RATIO="${2:?specify keep_ratio}"
export FUTURE_DLLM_STUDENT="${3:-}"
DATASET="${4:-all}"
OC_PYTHON="${OC_PYTHON:-${PY:-python}}"
case "$FAMILY" in
  llada) DEFAULT_MODEL="$REPO/model/LLaDA-8B-Instruct" ;;
  dream) DEFAULT_MODEL="$REPO/model/Dream-v0-Instruct-7B" ;;
  *) echo "unknown model family: $FAMILY" >&2; exit 2 ;;
esac
case "$DATASET" in
  all) ;;
  arc_c|piqa|gpqa) ;;
  *) echo "unknown OpenCompass dataset: $DATASET" >&2; exit 2 ;;
esac
if [[ ! "$FUTURE_DLLM_KEEP_RATIO" =~ ^1([.]0+)?$ ]] && [ -z "$FUTURE_DLLM_STUDENT" ]; then
  echo "keep_ratio=$FUTURE_DLLM_KEEP_RATIO requires a student checkpoint" >&2
  exit 2
fi
export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$DEFAULT_MODEL}"
export FUTURE_DLLM_LLADA_MODEL="$FUTURE_DLLM_MODEL"
export FUTURE_DLLM_LLADA_STUDENT="$FUTURE_DLLM_STUDENT"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-$REPO/.oc_cache}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
source "$REPO/scripts/runtime_env.sh"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WORK_DIR="${WORK_DIR:-$REPO/results/oc/${FAMILY}/${DATASET}/keep${FUTURE_DLLM_KEEP_RATIO}/$STAMP}"
EXTRA=()
if [ -n "${LIMIT:-}" ]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [ "${DRY_RUN:-0}" = 1 ]; then EXTRA+=(--dry-run); fi
mkdir -p "$REPO/logs/oc"
cd "$REPO"
"$OC_PYTHON" eval_oc/run.py "$FAMILY" --dataset "$DATASET" \
  --work-dir "$WORK_DIR" "${EXTRA[@]}" \
  2>&1 | tee "$REPO/logs/oc/${FAMILY}_${DATASET}_${STAMP}.log"
