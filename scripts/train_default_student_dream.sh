#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
source "$REPO/scripts/dream_decoding_env.sh"
MAX_SEQ_LEN=2048
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_dream_${MAX_SEQ_LEN}_per_head_${DREAM_DECODER_TAG}}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROJ_DIM="${PROJ_DIM:-256}"
MLP_DIM="${MLP_DIM:-512}"
PAIRS="${PAIRS:-4096}"
BLOCK_LENGTH=32
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MAX_SHARDS="${MAX_SHARDS-500,371,150,100,500}"
SAMPLE_TAG="${SAMPLE_TAG:-${MAX_SHARDS//,/-}}"
RUN_NAME="${RUN_NAME:-dream_${DREAM_DECODER_TAG}_5ds${SAMPLE_TAG:+_$SAMPLE_TAG}_e${EPOCHS}_lr${LR}_len${MAX_SEQ_LEN}_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-$REPO/logs/train/train_${RUN_NAME}.log}"

ROOTS=(
  "$TEACHER_ROOT/math5s"
  "$TEACHER_ROOT/mbpp_full"
  "$TEACHER_ROOT/gov_report"
  "$TEACHER_ROOT/multi_news"
  "$TEACHER_ROOT/musique"
)
TEACHER_ROOTS="$(IFS=,; echo "${ROOTS[*]}")"

source "$REPO/scripts/runtime_env.sh"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

MAX_SHARDS_ARGS=()
[ -n "$MAX_SHARDS" ] && MAX_SHARDS_ARGS=(--max-shards "$MAX_SHARDS")

printf 'default student training (dream)\nmodel=%s\nteacher=%s\nmax_seq_len=%s\nmax_shards=%s\ngpu=%s\nrun=%s\nlog=%s\n' \
  "$MODEL" "$TEACHER_ROOT" "$MAX_SEQ_LEN" "${MAX_SHARDS:-all}" \
  "$CUDA_VISIBLE_DEVICES" "$RUN_NAME" "$LOG_FILE"

"$PY" "$REPO/student/train_student.py" \
  --model "$MODEL" \
  --teacher-root "$TEACHER_ROOTS" \
  "${MAX_SHARDS_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --val-ratio "$VAL_RATIO" \
  --proj-dim "$PROJ_DIM" \
  --mlp-dim "$MLP_DIM" \
  --pairs "$PAIRS" \
  --block-length "$BLOCK_LENGTH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --name "$RUN_NAME" \
  "${DREAM_ARGS[@]}"

echo "default student training (dream) complete"
