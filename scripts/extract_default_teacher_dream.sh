#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
source "$REPO/scripts/dream_decoding_env.sh"
MAX_SEQ_LEN=2048
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards_dream_${MAX_SEQ_LEN}}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_dream_${MAX_SEQ_LEN}_per_head_${DREAM_DECODER_TAG}}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_default_teacher_dream_${RUN_TAG}.log}"

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

printf 'default teacher extraction (dream)\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nper_head=%s\nreduce=row:%s group:%s\ndatasets=%s\nlimits=%s\ngpu=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" \
  "$PER_HEAD" "${LABEL_ROW_REDUCE:-max}" "${LABEL_GROUP_REDUCE:-mean}" \
  "${DATASETS[*]}" "${LIMITS[*]}" \
  "$CUDA_VISIBLE_DEVICES" "$LOG_FILE"

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
  "$PY" "$REPO/teacher/extract_teacher_dream.py" \
    --dataset "$dataset" \
    --n-samples "$limit" \
    --block-length 32 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT" \
    --seed "$DREAM_SEED" "${DREAM_ARGS[@]}" \
    "${PER_HEAD_ARGS[@]}" "${REDUCE_ARGS[@]}"
done

echo "default teacher extraction (dream) complete"
