# Shared settings, sourced by the scripts in this directory.
#   FAMILY  llada | dream (first positional argument of every script)
#   MODEL   checkpoint path            (default: model/<family checkpoint>)
#   PY      python for lm-eval, extraction and training (default: python)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
ARTIFACTS="${ARTIFACTS:-$REPO/artifacts}"
BLOCK_LENGTH=32
export TOKENIZERS_PARALLELISM=false

case "${FAMILY:-}" in
  llada)
    MODEL="${MODEL:-$REPO/model/LLaDA-8B-Instruct}"
    MAX_SEQ_LEN=4096
    DECODING_ARGS=()
    SEED_ARGS=()
    ;;
  dream)
    MODEL="${MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
    MAX_SEQ_LEN=2048
    DREAM_ALG="${DREAM_ALG:-entropy}"
    DREAM_TEMPERATURE="${DREAM_TEMPERATURE:-0.2}"
    DREAM_TOP_P="${DREAM_TOP_P:-0.95}"
    DREAM_STEPS="${DREAM_STEPS:-512}"
    DREAM_SEED="${DREAM_SEED:-0}"
    DECODING_ARGS=(--dream-alg "$DREAM_ALG" --dream-temperature "$DREAM_TEMPERATURE"
                   --dream-top-p "$DREAM_TOP_P" --dream-steps "$DREAM_STEPS")
    SEED_ARGS=(--seed "$DREAM_SEED")
    ;;
  *)
    echo "model family must be llada or dream, got '${FAMILY:-}'" >&2
    exit 2
    ;;
esac

# Training sets and prompts per set used for the released students.
read -r -a TRAIN_DATASETS <<< "${DATASETS:-math5s mbpp_full gov_report multi_news musique}"
read -r -a TRAIN_LIMITS <<< "${LIMITS:-500 371 150 100 500}"
if [ "${#TRAIN_DATASETS[@]}" -ne "${#TRAIN_LIMITS[@]}" ]; then
  echo "DATASETS and LIMITS need the same number of entries" >&2
  exit 2
fi
PROMPT_ROOT="${PROMPT_ROOT:-$ARTIFACTS/$FAMILY/prompts}"
TEACHER_ROOT="${TEACHER_ROOT:-$ARTIFACTS/$FAMILY/teacher}"
