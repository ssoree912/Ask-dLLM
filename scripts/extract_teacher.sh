#!/usr/bin/env bash
# Build prompt shards and extract teacher labels for every training set.
#   scripts/extract_teacher.sh <llada|dream>
#   DATASETS="math5s" LIMITS="500" scripts/extract_teacher.sh llada
set -euo pipefail
FAMILY="${1:?usage: extract_teacher.sh <llada|dream>}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="$REPO/logs/teacher/${FAMILY}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
printf 'model=%s\ndatasets=%s\nlimits=%s\nprompts=%s\nteacher=%s\n' \
  "$MODEL" "${TRAIN_DATASETS[*]}" "${TRAIN_LIMITS[*]}" "$PROMPT_ROOT" "$TEACHER_ROOT"

for i in "${!TRAIN_DATASETS[@]}"; do
  "$PY" "$REPO/teacher/build_prompt_shards.py" \
    --dataset "${TRAIN_DATASETS[$i]}" --limit "${TRAIN_LIMITS[$i]}" \
    --model "$MODEL" --max-seq-len "$MAX_SEQ_LEN" \
    --data-root "$DATA_ROOT" --out-root "$PROMPT_ROOT"
done

for i in "${!TRAIN_DATASETS[@]}"; do
  "$PY" "$REPO/teacher/extract_teacher.py" \
    --dataset "${TRAIN_DATASETS[$i]}" --n-samples "${TRAIN_LIMITS[$i]}" \
    --model "$MODEL" --max-seq-len "$MAX_SEQ_LEN" --block-length "$BLOCK_LENGTH" \
    --shard-root "$PROMPT_ROOT" --output-root "$TEACHER_ROOT" \
    "${SEED_ARGS[@]}" "${DECODING_ARGS[@]}"
done
