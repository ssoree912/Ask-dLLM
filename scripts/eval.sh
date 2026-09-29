#!/usr/bin/env bash
# Evaluate one task with the student-driven cache at a given keep ratio.
#   scripts/eval.sh <llada|dream> <task> <keep_ratio> [student_checkpoint]
#
# arc_c / piqa / gpqa / mmlu run through OpenCompass (OC_PY, default: PY);
# every other task runs through lm-eval. keep_ratio=1.0 disables eviction.
#   LIMIT=N        evaluate the first N examples (per MMLU subject for mmlu)
#   LOG_SAMPLES=1  keep per-sample lm-eval outputs
set -euo pipefail
FAMILY="${1:?usage: eval.sh <llada|dream> <task> <keep_ratio> [checkpoint]}"
TASK="${2:?specify a task}"
KEEP="${3:?specify keep_ratio}"
CKPT="${4:-}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ! "$KEEP" =~ ^1([.]0*)?$ ]] && [ -z "$CKPT" ]; then
  echo "keep_ratio=$KEEP requires a student checkpoint" >&2
  exit 2
fi
if [ -n "$CKPT" ]; then CKPT="$(cd "$CKPT" && pwd)"; fi
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$REPO/results/$FAMILY/$TASK/keep$KEEP/$STAMP"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/eval.log") 2>&1
echo "model=$MODEL task=$TASK keep_ratio=$KEEP student=${CKPT:--} out=$RUN_DIR"

case "$TASK" in
  arc_c|piqa|gpqa|mmlu)
    ARGS=(--family "$FAMILY" --model "$MODEL" --tasks "$TASK" --keep-ratio "$KEEP"
          --data-root "$DATA_ROOT" --work-dir "$RUN_DIR")
    [ -n "$CKPT" ] && ARGS+=(--student "$CKPT")
    [ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
    [ "$FAMILY" = dream ] && ARGS+=("${DECODING_ARGS[@]}" --dream-seed "$DREAM_SEED")
    # OpenCompass writes scratch files to the working directory; keep them in the run.
    cd "$RUN_DIR"
    PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "${OC_PY:-$PY}" "$REPO/eval_oc/run.py" "${ARGS[@]}"
    exit
    ;;
esac

FEWSHOT=()
EXTRA=()
case "$TASK" in
  narrativeqa|qasper|multifieldqa_en|hotpotqa|2wikimqa|musique|gov_report|qmsum|multi_news|trec|triviaqa|samsum|passage_count|passage_retrieval_en|lcc|repobench-p)
    LM_TASK="longbench_$TASK" ;;
  gsm8k) LM_TASK=local_gsm8k; FEWSHOT=(--num_fewshot 5) ;;
  math|math500) LM_TASK="local_$TASK" ;;
  humaneval) LM_TASK=local_humaneval; EXTRA=(--confirm_run_unsafe_code) ;;
  mbpp) LM_TASK=local_mbpp; FEWSHOT=(--num_fewshot 3)
        EXTRA=(--apply_chat_template --fewshot_as_multiturn --confirm_run_unsafe_code) ;;
  bbh) LM_TASK=local_bbh; FEWSHOT=(--num_fewshot 3)
       EXTRA=(--apply_chat_template --fewshot_as_multiturn) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_ARGS="pretrained=$MODEL,block_len=$BLOCK_LENGTH,keep_ratio=$KEEP,max_seq_len=$MAX_SEQ_LEN"
[ -n "$CKPT" ] && MODEL_ARGS="$MODEL_ARGS,student_path=$CKPT"
if [ "$FAMILY" = dream ]; then
  MODEL_ARGS="$MODEL_ARGS,dream_alg=$DREAM_ALG,dream_temperature=$DREAM_TEMPERATURE"
  MODEL_ARGS="$MODEL_ARGS,dream_top_p=$DREAM_TOP_P,dream_steps=$DREAM_STEPS,dream_seed=$DREAM_SEED"
fi

# lm-eval resolves data_files literally, so the task YAMLs are copied with DATA_DIR filled in.
TASK_DIR="$RUN_DIR/tasks"
mkdir -p "$TASK_DIR"
cp "$REPO"/eval/tasks/*.py "$TASK_DIR/"
for f in "$REPO"/eval/tasks/{local,local_bbh,longbench}/*; do
  sed "s#DATA_DIR#$DATA_ROOT#g" "$f" > "$TASK_DIR/$(basename "$f")"
done

[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ "${LOG_SAMPLES:-0}" != 0 ] && EXTRA+=(--log_samples)
cd "$REPO"
"$PY" eval/run.py --model ask_dllm --model_args "$MODEL_ARGS" \
  --tasks "$LM_TASK" "${FEWSHOT[@]}" --include_path "$TASK_DIR" \
  --batch_size 1 "${EXTRA[@]}" --output_path "$RUN_DIR"
