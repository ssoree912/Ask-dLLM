# Ask-dLLM

KV cache eviction for diffusion LLMs, ranking cache entries by what the completed answer needs.

Models: `GSAI-ML/LLaDA-8B-Instruct`, `Dream-org/Dream-v0-Instruct-7B`.

## Installation

```bash
conda env create -f environment.yml
conda activate future-dllm
export CUDA_VISIBLE_DEVICES=2
```

- [Model download](scripts/download_model.sh)
- [Dataset download](scripts/download_data.py)
- Teacher extraction: [LLaDA](scripts/extract_default_teacher.sh) / [Dream](scripts/extract_default_teacher_dream.sh)
- Student training: [LLaDA](scripts/train_default_student.sh) / [Dream](scripts/train_default_student_dream.sh)

```text
Ask-dLLM/
├── model/
│   ├── LLaDA-8B-Instruct/
│   └── Dream-v0-Instruct-7B/
├── data/
│   ├── eval/<name>/           # Evaluation parquet files
│   ├── train/<name>/          # Training parquet/JSONL files
│   ├── longbench/data/       # LongBench JSONL files
│   └── gpqa/                 # OpenCompass GPQA CSV files
├── artifacts/                # Prompt shards, teacher labels, checkpoints
└── results/                  # Evaluation results
```

```bash
scripts/download_model.sh
MODEL_REPO=Dream-org/Dream-v0-Instruct-7B \
  MODEL_DIR="$PWD/model/Dream-v0-Instruct-7B" scripts/download_model.sh
python scripts/download_data.py --parts eval train longbench bbh
```

OpenCompass setup:

```bash
python -m venv --system-site-packages .venv-oc
.venv-oc/bin/python -m pip install -r requirements-oc.txt
export OC_PYTHON="$PWD/.venv-oc/bin/python"
scripts/fetch_oc_data.sh
```

GPQA requires approved dataset access and `HF_TOKEN`.

## Evaluation datasets

Total length (prompt + generation): **LLaDA 4096**, **Dream 2048**. Block length: **32**.

| Dataset | Generation length |
|---|---:|
| `gov_report` / `multi_news` / `qmsum` | 512 |
| `samsum` / `qasper` / `narrativeqa` | 128 |
| `trec` / `lcc` / `repobench-p` / `multifieldqa_en` | 64 |
| `triviaqa` / `2wikimqa` / `hotpotqa` / `musique` / `passage_retrieval_en` / `passage_count` | 32 |
| `gsm8k` (5-shot) | 256 |
| `math` / `math500` (4-shot) | 256 |
| `humaneval` / `mbpp` (3-shot for MBPP) | 512 |
| `bbh` (3-shot) | 256 |
| `arc_c` / `piqa` / `gpqa` (5-shot for GPQA) | 256 |

ARC-Challenge, PIQA, and GPQA use **OpenCompass generative evaluation**. Other tasks use lm-eval.

## Training datasets

| Dataset | Samples | Generation length | LLaDA prompt limit | Dream prompt limit | Teacher blocks |
|---|---:|---:|---:|---:|---:|
| `math5s` | 500 | 256 | 3840 | 1792 | 8 |
| `mbpp_full` | 371 | 256 | 3840 | 1792 | 8 |
| `gov_report` | 150 | 512 | 3584 | 1536 | 16 |
| `multi_news` | 100 | 512 | 3584 | 1536 | 16 |
| `musique` | 500 | 32 | 4064 | 2016 | 1 |

Teacher blocks = generation length / 32. Prompt limit = total length − generation length.

## Teacher labels

Default extraction:

```bash
scripts/extract_default_teacher.sh          # LLaDA
scripts/extract_default_teacher_dream.sh    # Dream
```

Extract one dataset:

```bash
DATASETS=math5s LIMITS=500 scripts/extract_default_teacher.sh
DATASETS=math5s LIMITS=500 scripts/extract_default_teacher_dream.sh
```

Extraction and training use total lengths of 4096 for LLaDA and 2048 for Dream, with separate artifact directories.

## Training

```bash
scripts/train_default_student.sh          # LLaDA
scripts/train_default_student_dream.sh    # Dream
```

Train on one dataset:

```bash
python student/train_student.py \
  --model model/LLaDA-8B-Instruct \
  --teacher-root artifacts/teacher/math5s \
  --max-seq-len 4096 --block-length 32
```

## Inference

```bash
scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]

scripts/run_eval.sh samsum 0.1 artifacts/ckpts/<run>/checkpoint-best
scripts/run_eval.sh gsm8k 1.0    # Full cache; no checkpoint required

FUTURE_DLLM_MODEL="$PWD/model/Dream-v0-Instruct-7B" \
  scripts/run_eval.sh gsm8k 0.1 artifacts/ckpts/<dream-run>/checkpoint-best

LIMIT=200 scripts/run_eval.sh math 0.1 artifacts/ckpts/<run>/checkpoint-best
```

OpenCompass — ARC-Challenge, PIQA, and GPQA:

```bash
scripts/run_oc_mc.sh llada 0.1 artifacts/ckpts/<llada-run>/checkpoint-best
scripts/run_oc_mc.sh dream 0.1 artifacts/ckpts/<dream-run>/checkpoint-best
scripts/run_oc_mc.sh llada 1.0    # Full cache
```
