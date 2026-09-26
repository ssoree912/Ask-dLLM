# Ask-dLLM

KV cache eviction for diffusion LLMs, ranking cache entries by what the completed answer needs.

Models: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) / [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B).

Compact PyTorch reference algorithms for teacher extraction, student training, and cache eviction. Full model integration and benchmark runners are on [dev](https://github.com/ssoree912/Ask-dLLM/tree/dev).

## Installation

```bash
pip install -r requirements.txt
```

## Core algorithms

```text
Ask-dLLM/
├── teacher/extract_teacher.py   # Completed-block attention targets
├── student/train_student.py     # Block-conditioned scorer and ranking loss
└── inference/generate.py        # Block-wise top-K cache selection
```

Model-specific forward passes and token-reveal schedules are supplied through the `Decoder` interface in `inference/generate.py`.

## Evaluation datasets

Total length (prompt + generation): **LLaDA 4096**, **Dream 2048**. Block length: **32**.

[LongBench](https://huggingface.co/datasets/zai-org/LongBench/tree/main) provides the first four rows below.

| Dataset | Generation length |
|---|---:|
| `gov_report` / `multi_news` / `qmsum` | 512 |
| `samsum` / `qasper` / `narrativeqa` | 128 |
| `trec` / `lcc` / `repobench-p` / `multifieldqa_en` | 64 |
| `triviaqa` / `2wikimqa` / `hotpotqa` / `musique` / `passage_retrieval_en` / `passage_count` | 32 |
| [gsm8k](https://huggingface.co/datasets/openai/gsm8k) (5-shot) | 256 |
| [math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) / [math500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (4-shot) | 256 |
| [humaneval](https://huggingface.co/datasets/openai/openai_humaneval) / [mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (3-shot for MBPP) | 512 |
| [bbh](https://huggingface.co/datasets/lukaemon/bbh) (3-shot) | 256 |
| [arc_c](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ARC.zip) / [piqa](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/piqa.zip) / [gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (5-shot for GPQA) | 256 |

ARC-Challenge, PIQA, and GPQA use **OpenCompass generative evaluation**. Other tasks use lm-eval.

## Training datasets

| Dataset | Samples | Generation length | LLaDA prompt limit | Dream prompt limit | Teacher blocks |
|---|---:|---:|---:|---:|---:|
| [math5s](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | 500 | 256 | 3840 | 1792 | 8 |
| [mbpp_full](https://huggingface.co/datasets/google-research-datasets/mbpp) | 371 | 256 | 3840 | 1792 | 8 |
| [gov_report](https://huggingface.co/datasets/ccdv/govreport-summarization) | 150 | 512 | 3584 | 1536 | 16 |
| [multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) | 100 | 512 | 3584 | 1536 | 16 |
| [musique](https://huggingface.co/datasets/dgslibisey/MuSiQue) | 500 | 32 | 4064 | 2016 | 1 |

Teacher blocks = generation length / 32. Prompt limit = total length − generation length.

## Teacher labels

[extract_teacher.py](teacher/extract_teacher.py) decodes with the full cache, saves the state before step 1, and computes attention targets from the completed block. Labels take the maximum attention over block rows after averaging heads, with an optional per-KV-head reduction.

## Training

[train_student.py](student/train_student.py) replays the frozen model at selection time. Each layer scores `[token; mean block; token × mean block]` projections using an MLP, trained with listwise KL and pairwise ranking losses.

## Inference

[generate.py](inference/generate.py) selects the top-K external cache entries during step 1 and uses them for the remaining block steps. The current 32-token block stays available; `keep_ratio=1.0` retains the full cache. A fresh cache is built for each block.
