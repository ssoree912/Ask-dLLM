#!/usr/bin/env bash
# Train the student scorer on the teacher labels of every training set.
#   scripts/train_student.sh <llada|dream>
#   EPOCHS=10 LR=2e-4 RUN_NAME=my_run scripts/train_student.sh dream
set -euo pipefail
FAMILY="${1:?usage: train_student.sh <llada|dream>}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$ARTIFACTS/$FAMILY/ckpts/$RUN_NAME"
ROOTS=()
for dataset in "${TRAIN_DATASETS[@]}"; do ROOTS+=("$TEACHER_ROOT/$dataset"); done

LOG="$REPO/logs/train/${FAMILY}_${RUN_NAME}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

"$PY" "$REPO/student/train_student.py" \
  --model "$MODEL" \
  --teacher-root "$(IFS=,; echo "${ROOTS[*]}")" \
  --max-shards "$(IFS=,; echo "${TRAIN_LIMITS[*]}")" \
  --epochs "$EPOCHS" --lr "$LR" \
  --seed "${SEED:-0}" --val-ratio "${VAL_RATIO:-0.1}" \
  --proj-dim "${PROJ_DIM:-256}" --mlp-dim "${MLP_DIM:-512}" \
  --pairs "${PAIRS:-4096}" --lambda-list "${LAMBDA_LIST:-1.0}" \
  --max-seq-len "$MAX_SEQ_LEN" --block-length "$BLOCK_LENGTH" \
  --output-dir "$OUTPUT_DIR" "${DECODING_ARGS[@]}"
