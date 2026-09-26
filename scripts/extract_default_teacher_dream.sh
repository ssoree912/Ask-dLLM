#!/usr/bin/env bash
# Build prompts and extract the default five-domain teacher labels, on Dream.
#
# Separate artifact roots from the LLaDA run on purpose. Dream tokenises with a
# Qwen2 chat template, so its prompt shards have different ids and lengths for
# the same sample, and the resume check in build_prompt_shards only compares
# lengths -- pointed at the LLaDA root it would silently accept LLaDA shards.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
source "$REPO/scripts/dream_decoding_env.sh"
# Dream uses a 2048-token total budget, including generation.
#
# DATASETS / LIMITS / PER_HEAD / TEACHER_ROOT are overridable, so a per-head run
# on a subset does not need a second copy of this script:
#
#   PER_HEAD=1 LIMITS="250 185 75 50 250" \
#     TEACHER_ROOT=$PWD/artifacts/teacher_dream_perhead \
#     scripts/extract_default_teacher_dream.sh
MAX_SEQ_LEN=2048
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards_dream_${MAX_SEQ_LEN}}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_dream_${MAX_SEQ_LEN}_${DREAM_DECODER_TAG}}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_default_teacher_dream_${RUN_TAG}.log}"

read -r -a DATASETS <<< "${DATASETS:-math5s mbpp_full gov_report multi_news musique}"
read -r -a LIMITS <<< "${LIMITS:-500 371 150 100 500}"
# Head-averaged labels force one kept set per layer; --per-head keeps the axis
# so each head can keep its own. On Dream that axis is the KV head axis (4, not
# the 28 query heads): the cache holds one entry per KV head, so that is the
# finest granularity eviction can act on. 4x the storage, hence its own root.
PER_HEAD_ARGS=()
[ -n "${PER_HEAD:-}" ] && PER_HEAD_ARGS=(--per-head)
# How the block's rows, and the query heads sharing a KV entry, are folded into
# the label. max is what the label shipped with; mean is the alternative the
# row/group diagnostic argues for. Each writes its own teacher_kind, so the
# resume check refuses to mix them in one root.
REDUCE_ARGS=(--label-row-reduce "${LABEL_ROW_REDUCE:-max}"
             --label-group-reduce "${LABEL_GROUP_REDUCE:-max}")

export FUTURE_DLLM_DATA="$DATA_ROOT"
source "$REPO/scripts/runtime_env.sh"

mkdir -p "$(dirname "$LOG_FILE")" "$PROMPT_ROOT" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default teacher extraction (dream)\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nper_head=%s\nreduce=row:%s group:%s\ndatasets=%s\nlimits=%s\ngpu=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" \
  "${PER_HEAD:-0}" "${LABEL_ROW_REDUCE:-max}" "${LABEL_GROUP_REDUCE:-max}" \
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
