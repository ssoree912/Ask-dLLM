#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_per_head}"
MAX_SEQ_LEN=4096
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_default_teacher_${RUN_TAG}.log}"

read -r -a DATASETS <<< "${DATASETS:-math5s mbpp_full gov_report multi_news musique}"
read -r -a LIMITS <<< "${LIMITS:-500 371 150 100 500}"
PER_HEAD="${PER_HEAD:-1}"
case "$PER_HEAD" in
  1) PER_HEAD_ARGS=(--per-head) ;;
  0) PER_HEAD_ARGS=(--no-per-head) ;;
  *) echo "PER_HEAD must be 0 or 1" >&2; exit 2 ;;
esac
REDUCE_ARGS=(--label-row-reduce "${LABEL_ROW_REDUCE:-max}"
             --label-group-reduce "${LABEL_GROUP_REDUCE:-mean}")

export FUTURE_DLLM_DATA="$DATA_ROOT"
source "$REPO/scripts/runtime_env.sh"

mkdir -p "$(dirname "$LOG_FILE")" "$PROMPT_ROOT" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default teacher extraction\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nper_head=%s\nreduce=row:%s group:%s\ndatasets=%s\nlimits=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" \
  "$PER_HEAD" "${LABEL_ROW_REDUCE:-max}" "${LABEL_GROUP_REDUCE:-mean}" \
  "${DATASETS[*]}" "${LIMITS[*]}" "$LOG_FILE"

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  limit="${LIMITS[$index]}"
  "$PY" "$REPO/teacher/build_prompt_shards.py" \
    --dataset "$dataset" \
    --limit "$limit" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --out-root "$PROMPT_ROOT"
done

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  limit="${LIMITS[$index]}"
  "$PY" "$REPO/teacher/extract_teacher_llada.py" \
    --dataset "$dataset" \
    --n-samples "$limit" \
    --block-length 32 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT" \
    "${PER_HEAD_ARGS[@]}" \
    "${REDUCE_ARGS[@]}"
done

echo "default teacher extraction complete"
