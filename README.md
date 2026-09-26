# Ask-dLLM

Teacher extraction, cache-scorer training, and inference for LLaDA and Dream.
The teacher measures attention from completed answer blocks. The student learns
to rank cache entries using the hidden states available at eviction time.

This development branch is based on
[Future_dLLM, commit 890cce0](https://github.com/ssoree912/Future_dLLM/commit/890cce01c70c06ffa6d7b22deeaff669bfdb8c7f).

## Configuration

| Model | Total tokens (prompt + generation) | Block length |
| --- | ---: | ---: |
| LLaDA-8B-Instruct | 4096 | 32 |
| Dream-v0-Instruct-7B | 2048 | 32 |

The launch scripts use these budgets for extraction, training, and inference.
Each prompt is left-truncated after reserving the task's generation budget.
Dream defaults to entropy decoding, temperature 0.2, top-p 0.95, and a
256-step cap. Its scorer checkpoint must match the extraction decoding settings.
Scripts expose only physical GPU 2 as `cuda:0`; model loading uses one GPU.

## Installation and data

```bash
conda env create -f environment.yml
conda activate future-dllm
bash scripts/download_model.sh
MODEL_REPO=Dream-org/Dream-v0-Instruct-7B \
  MODEL_DIR="$PWD/model/Dream-v0-Instruct-7B" bash scripts/download_model.sh
python scripts/download_data.py --parts train eval longbench bbh
```

Use local model directories containing `config.json`, tokenizer files, and
weights. Set `FUTURE_DLLM_MODEL` to use an existing model directory and
`FUTURE_DLLM_DATA` to use an existing training/lm-eval data directory.
Weights, datasets, teacher labels, checkpoints, and results are not committed.

OpenCompass runs in a separate environment because it requires older NumPy and
Pandas versions than the training environment:

```bash
python -m venv --system-site-packages .venv-oc
.venv-oc/bin/python -m pip install -r requirements-oc.txt
export OC_PYTHON="$PWD/.venv-oc/bin/python"
bash scripts/fetch_oc_data.sh
```

GPQA requires access to the gated `Idavidrein/gpqa` dataset and a Hugging Face
token (`HF_TOKEN`). OpenCompass downloads ARC-Challenge on first use; PIQA is
prepared by `fetch_oc_data.sh`. The default cache is `.oc_cache/`, configurable
with `COMPASS_DATA_CACHE`. GPQA CSV files live in `data/gpqa/`.

## Teacher extraction

```bash
# LLaDA: 4096 total tokens, 32-token blocks.
bash scripts/extract_default_teacher.sh

# Dream: 2048 total tokens, 32-token blocks.
bash scripts/extract_default_teacher_dream.sh
```

| Training dataset | Samples | Generation tokens | Blocks |
| --- | ---: | ---: | ---: |
| MATH, five subjects (`math5s`) | 500 | 256 | 8 |
| MBPP full (`mbpp_full`) | 371 | 256 | 8 |
| GovReport (`gov_report`) | 150 | 512 | 16 |
| Multi-News (`multi_news`) | 100 | 512 | 16 |
| MuSiQue (`musique`) | 500 | 32 | 1 |

Prompt limits are the model's total budget minus the generation tokens above.
LLaDA and Dream use separate prompt and teacher directories under `artifacts/`.
Set `PROMPT_ROOT` or `TEACHER_ROOT` to relocate them. `DATASETS` and `LIMITS`
override the extraction datasets and sample counts. `PER_HEAD=1` produces
per-KV-head labels; use a separate teacher directory for each label variant.

## Student training

```bash
bash scripts/train_default_student.sh
bash scripts/train_default_student_dream.sh
```

Training freezes the base model and fits the cache scorer with listwise and
pairwise ranking losses. Validation selects `artifacts/ckpts/<run>/checkpoint-best`.
The default scripts use 10 epochs, learning rate `2e-4`, and the five domains
listed above. Override `EPOCHS`, `LR`, `RUN_NAME`, `TEACHER_ROOT`, and
`MAX_SHARDS` as needed. Always train against the same base model and labels.

## Inference

Generative tasks use lm-eval: GSM8K, MATH, MATH-500, HumanEval, MBPP, BBH, and
the 16 LongBench tasks in `eval/tasks/longbench/`.

```bash
# Trained scorer; keep 10% of the external cache.
bash scripts/run_eval.sh gsm8k 0.1 artifacts/ckpts/<run>/checkpoint-best

# Full cache; no scorer checkpoint required.
bash scripts/run_eval.sh gsm8k 1.0

# Dream uses its own model, scorer, and 2048-token total budget.
FUTURE_DLLM_MODEL="$PWD/model/Dream-v0-Instruct-7B" \
  bash scripts/run_eval.sh gov_report 0.1 artifacts/ckpts/<dream-run>/checkpoint-best
```

**ARC-Challenge, PIQA, and GPQA use OpenCompass generation and accuracy scoring.**
They never use lm-eval loglikelihood in this repository. ARC-Challenge uses the
test split (OpenCompass retains four-choice questions), PIQA uses validation,
and GPQA uses diamond with five fixed demonstrations.

```bash
# All three multiple-choice tasks, one model at a time.
bash scripts/run_oc_mc.sh llada 0.1 artifacts/ckpts/<llada-run>/checkpoint-best
bash scripts/run_oc_mc.sh dream 0.1 artifacts/ckpts/<dream-run>/checkpoint-best

# Full cache, or a single task.
bash scripts/run_oc_mc.sh llada 1.0
bash scripts/run_oc_mc.sh dream 1.0 "" arc_c

# This entry point also routes the three tasks to OpenCompass.
bash scripts/run_eval.sh piqa 0.1 artifacts/ckpts/<llada-run>/checkpoint-best
```

`LIMIT=2` runs a small subset. `DRY_RUN=1` on `run_oc_mc.sh` writes the resolved
configuration without loading a model. Results are stored under `results/` and
logs under `logs/`. `LOG_SAMPLES=1` saves lm-eval generations. For interrupted
lm-eval runs, set `FUTURE_DLLM_RESUME` to the previous run's `resume.jsonl`.
HumanEval and MBPP execute generated Python during scoring.

## Code layout

- `teacher/`: prompt preparation and teacher-label extraction.
- `student/`: scorer training and checkpoint selection.
- `future_dllm/`: model backends, cache scorer, and block decoding.
- `eval/`: generative lm-eval integration and local task definitions.
- `eval_oc/`: OpenCompass integration for the three multiple-choice tasks.
- `scripts/`: data preparation and extraction, training, and inference launchers.
