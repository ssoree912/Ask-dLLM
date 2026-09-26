#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_per_head}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROJ_DIM="${PROJ_DIM:-256}"
MLP_DIM="${MLP_DIM:-512}"
PAIRS="${PAIRS:-4096}"
MAX_SEQ_LEN=4096
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MAX_SHARDS="${MAX_SHARDS:-500,371,150,100,500}"
LAMBDA_LIST="${LAMBDA_LIST:-1.0}"
RUN_NAME="${RUN_NAME:-default_5ds_$(echo "$MAX_SHARDS" | tr , -)_e${EPOCHS}_lr${LR}_lam${LAMBDA_LIST}_${RUN_TAG}}"
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

printf 'default student training\nmodel=%s\nteacher=%s\nmax_seq_len=%s\nrun=%s\nlog=%s\n' \
  "$MODEL" "$TEACHER_ROOT" "$MAX_SEQ_LEN" "$RUN_NAME" "$LOG_FILE"
echo "lambda_list=$LAMBDA_LIST"

"$PY" "$REPO/student/train_student.py" \
  --model "$MODEL" \
  --teacher-root "$TEACHER_ROOTS" \
  --max-shards "$MAX_SHARDS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --val-ratio "$VAL_RATIO" \
  --proj-dim "$PROJ_DIM" \
  --mlp-dim "$MLP_DIM" \
  --pairs "$PAIRS" \
  --lambda-list "$LAMBDA_LIST" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --block-length 32 \
  --name "$RUN_NAME"

echo "default student training complete"
