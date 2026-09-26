#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
KEEP="${2:?specify keep_ratio}"
CKPT="${3:-}"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
FAMILY="$("$PY" - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path
kind = json.loads((Path(sys.argv[1]) / "config.json").read_text())["model_type"].lower()
if "dream" in kind:
    print("dream")
elif "llada" in kind:
    print("llada")
else:
    raise SystemExit(f"unsupported model_type: {kind}")
PY
)"
case "$DATASET" in
  arc_c|piqa|gpqa)
    exec bash "$REPO/scripts/run_oc_mc.sh" "$FAMILY" "$KEEP" "$CKPT" "$DATASET" ;;
esac
SHOTS=()
EXTRA=()
case "$DATASET" in
  samsum|trec|triviaqa|2wikimqa|hotpotqa|musique|qasper|narrativeqa|multifieldqa_en|gov_report|qmsum|multi_news|lcc|repobench-p|passage_retrieval_en|passage_count)
    TASK="longbench_$DATASET" ;;
  gsm8k) TASK=local_gsm8k; SHOTS=(--num_fewshot 5) ;;
  math|math500) TASK="local_$DATASET" ;;
  humaneval) TASK=local_humaneval; EXTRA=(--confirm_run_unsafe_code); export HF_ALLOW_CODE_EVAL=1 ;;
  bbh) TASK=local_bbh; SHOTS=(--num_fewshot 3); EXTRA=(--apply_chat_template --fewshot_as_multiturn) ;;
  mbpp) TASK=local_mbpp; SHOTS=(--num_fewshot 3); EXTRA=(--apply_chat_template --fewshot_as_multiturn --confirm_run_unsafe_code); export HF_ALLOW_CODE_EVAL=1 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac
if [[ ! "$KEEP" =~ ^1([.]0+)?$ ]] && [ -z "$CKPT" ]; then
  echo "keep_ratio=$KEEP requires a student checkpoint" >&2
  exit 2
fi
source "$REPO/scripts/runtime_env.sh"
if [ "$FAMILY" = dream ]; then
  MODEL_NAME=Dream_future
  MAX_SEQ_LEN=2048
  source "$REPO/scripts/dream_decoding_env.sh"
else
  MODEL_NAME=LLaDA_future
  MAX_SEQ_LEN=4096
fi
ARGS="pretrained=$MODEL,block_len=32,keep_ratio=$KEEP,max_seq_len=$MAX_SEQ_LEN"
if [ "$FAMILY" = dream ]; then
  ARGS="$ARGS,dream_alg=$DREAM_ALG,dream_temperature=$DREAM_TEMPERATURE,dream_top_p=$DREAM_TOP_P,dream_steps=$DREAM_STEPS,dream_seed=$DREAM_SEED"
fi
if [ -n "$CKPT" ]; then
  ARGS="$ARGS,student_path=$(cd "$CKPT" && pwd)"
fi
export FUTURE_DLLM_DATA="${FUTURE_DLLM_DATA:-$REPO/data}"
export HF_HOME="$REPO/.hf_cache"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$REPO/results/$FAMILY/$DATASET/keep$KEEP/$STAMP"
mkdir -p "$RUN_DIR/tasks" "$REPO/logs/eval"
export FUTURE_DLLM_RESUME="${FUTURE_DLLM_RESUME:-$RUN_DIR/resume.jsonl}"
"$PY" - "$REPO" "$RUN_DIR/tasks" <<'PY'
import os
import shutil
import sys
from pathlib import Path
source = Path(sys.argv[1]) / "eval/tasks"
target = Path(sys.argv[2])
data = os.environ["FUTURE_DLLM_DATA"]
for path in source.glob("*.py"):
    shutil.copy2(path, target / path.name)
for folder in ("local", "local_bbh", "longbench"):
    for path in (source / folder).iterdir():
        if path.is_file():
            content = path.read_text().replace("LONGBENCH_DATA_DIR", f"{data}/longbench/data")
            (target / path.name).write_text(content.replace("DATA_DIR", data))
PY
if [ -n "${LIMIT:-}" ]; then EXTRA+=(--limit "$LIMIT"); fi
if [ "${LOG_SAMPLES:-0}" != 0 ]; then EXTRA+=(--log_samples); fi
cd "$REPO"
"$PY" eval/run.py --model "$MODEL_NAME" --model_args "$ARGS" \
  --tasks "$TASK" "${SHOTS[@]}" --include_path "$RUN_DIR/tasks" \
  --batch_size 1 "${EXTRA[@]}" --output_path "$RUN_DIR" \
  2>&1 | tee "$REPO/logs/eval/${FAMILY}_${DATASET}_${STAMP}.log"
