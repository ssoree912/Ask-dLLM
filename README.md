# Ask-dLLM

KV cache eviction for diffusion LLMs, ranking cache entries by what the completed answer needs.

A lightweight student reads each layer's hidden states and predicts, per KV head, how much
attention the finished block will pay to every cached token. At decode time the cache keeps
only the top `keep_ratio` entries the student ranks highest. The student is trained on
teacher labels: the attention the fully decoded block actually paid to each candidate.

Models: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) /
[Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B).

## Installation

```bash
conda env create -f environment.yml
conda activate ask-dllm
```

The multiple-choice suite runs through [OpenCompass](https://github.com/open-compass/opencompass),
which pins different `numpy`/`pandas` versions, so it lives in its own environment that
reuses the base packages:

```bash
python -m venv --system-site-packages .venv-oc
.venv-oc/bin/python -m pip install -r requirements-oc.txt
export OC_PY="$PWD/.venv-oc/bin/python"
```

## Layout

```text
Ask-dLLM/
├── ask_dllm/        model backends, student-driven KV cache, student network
├── teacher/         prompt shards and teacher label extraction
├── student/         student training
├── eval/            lm-eval model wrapper and task definitions
├── eval_oc/         OpenCompass model wrappers and configs
├── scripts/         extract_teacher.sh / train_student.sh / eval.sh
├── model/           LLaDA-8B-Instruct/  Dream-v0-Instruct-7B/
├── data/
│   ├── train/       gov_report/ hendrycks_math/ mbpp/ multi_news/ musique/ (+ gsm8k/ for few-shot)
│   └── eval/        longbench/ gsm8k/ math500/ hendrycks_math/ humaneval/ mbpp/ bbh/
│                    arc_c/ piqa/ gpqa/ mmlu/
├── artifacts/<family>/{prompts,teacher,ckpts}/
└── results/<family>/<task>/keep<ratio>/<run>/
```

`model/`, `data/`, `artifacts/` and `results/` are not tracked. Place (or symlink) the
checkpoints and datasets as above; every script also accepts `MODEL=...` and
`DATA_ROOT=...` to point elsewhere.

### Data

| Path | Contents | Source |
|---|---|---|
| `data/train/hendrycks_math/<subject>/train-*.parquet` | MATH train | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| `data/train/mbpp/full/train-*.parquet` | MBPP train | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| `data/train/gov_report/train.parquet` | GovReport train | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) |
| `data/train/multi_news/train.parquet` | Multi-News train (`document`, `summary`) | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) |
| `data/train/musique/musique_ans_v1.0_train.jsonl` | MuSiQue train | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) |
| `data/train/gsm8k/train.parquet` | GSM8K few-shot pool | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `data/eval/longbench/<task>.jsonl` | 16 LongBench tasks (`data.zip`) | [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench) |
| `data/eval/gsm8k/test.parquet` | GSM8K test | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `data/eval/hendrycks_math/<subject>-test.parquet` | MATH test | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| `data/eval/math500/test.parquet` | MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| `data/eval/humaneval/test.parquet` | HumanEval | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| `data/eval/mbpp/full/test-*.parquet` | MBPP test | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| `data/eval/bbh/<task>.parquet` | 27 BBH tasks | [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh) |
| `data/eval/arc_c/ARC-Challenge-Test.jsonl` | ARC-Challenge | [OpenCompass ARC.zip](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ARC.zip) (`ARC-c/`) |
| `data/eval/piqa/dev.jsonl`, `dev-labels.lst` | PIQA | [OpenCompass piqa.zip](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/piqa.zip) |
| `data/eval/gpqa/gpqa_diamond.csv` | GPQA-Diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) |
| `data/eval/mmlu/{dev,test}/<subject>_*.csv` | MMLU | [OpenCompass mmlu.zip](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip) |

## Settings

Total length (prompt + generation): **LLaDA 4096**, **Dream 2048**. Block length: **32**.
Denoising steps: **LLaDA = generation length**, **Dream = min(512, generation length)**
(Dream: entropy, temperature 0.2, top-p 0.95).
Scoring is per KV head: **LLaDA 32**, **Dream 4**; GQA query heads are averaged within each KV group.

## 1. Teacher extraction

```bash
scripts/extract_teacher.sh llada
scripts/extract_teacher.sh dream
```

Builds prompt shards under `artifacts/<family>/prompts/` and teacher labels under
`artifacts/<family>/teacher/`. Default training sets:

| Dataset | Samples | Generation length | Teacher blocks |
|---|---:|---:|---:|
| `math5s` (MATH, 5 subjects, no Asymptote) | 500 | 256 | 8 |
| `mbpp_full` | 371 | 256 | 8 |
| `gov_report` | 150 | 512 | 16 |
| `multi_news` | 100 | 512 | 16 |
| `musique` | 500 | 32 | 1 |

One dataset: `DATASETS=math5s LIMITS=500 scripts/extract_teacher.sh llada`.

## 2. Student training

```bash
scripts/train_student.sh llada
scripts/train_student.sh dream
```

Trains on all five teacher sets (10 epochs, lr 2e-4) and writes
`artifacts/<family>/ckpts/<RUN_NAME>/checkpoint-best`. `EPOCHS`, `LR`, `RUN_NAME`,
`DATASETS`/`LIMITS` override the defaults.

## 3. Evaluation

```bash
scripts/eval.sh <llada|dream> <task> <keep_ratio> [student_checkpoint]

scripts/eval.sh llada gsm8k 0.1 artifacts/llada/ckpts/<run>/checkpoint-best
scripts/eval.sh dream mmlu  0.1 artifacts/dream/ckpts/<run>/checkpoint-best
scripts/eval.sh llada gsm8k 1.0          # full cache, no student
LIMIT=50 scripts/eval.sh llada samsum 0.1 artifacts/llada/ckpts/<run>/checkpoint-best
```

| Task | Harness | Generation length |
|---|---|---:|
| `gov_report` / `multi_news` / `qmsum` | lm-eval | 512 |
| `samsum` / `qasper` / `narrativeqa` | lm-eval | 128 |
| `trec` / `lcc` / `repobench-p` / `multifieldqa_en` | lm-eval | 64 |
| `triviaqa` / `2wikimqa` / `hotpotqa` / `musique` / `passage_retrieval_en` / `passage_count` | lm-eval | 32 |
| `gsm8k` (5-shot) | lm-eval | 256 |
| `math` / `math500` (4-shot) | lm-eval | 256 |
| `humaneval` / `mbpp` (3-shot for MBPP) | lm-eval | 512 |
| `bbh` (3-shot) | lm-eval | 256 |
| `arc_c` / `piqa` / `gpqa` (5-shot) / `mmlu` (5-shot) | OpenCompass (generative) | 256 |

Outputs go to `results/<family>/<task>/keep<ratio>/<timestamp>/`. Set `CUDA_VISIBLE_DEVICES`
to choose the GPU; `LIMIT=N` evaluates the first N examples (per subject for MMLU).
