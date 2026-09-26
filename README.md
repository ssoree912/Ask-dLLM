# Ask-dLLM

Diffusion large language models (dLLMs) incur substantial computation and memory costs through repeated bidirectional attention. Temporal caching reduces redundant KV recomputation but retains full-sequence KV states, while current-attention-based eviction may discard entries needed after denoising. We introduce Ask-dLLM, a KV cache eviction framework that predicts denoised-state relevance conditioned on the current masked block. A lightweight scorer learns from the same frozen dLLM’s denoised-state relevance and ranks external cache candidates once per block using masked-state representations. Across 23 reasoning, code, and long-context benchmarks, retaining only 10% of external KV entries improves the average score over the strongest prior eviction baseline by 7.75 points on LLaDA and 2.39 points on Dream, while remaining within approximately 0.92 points of the unmodified Origin model on both backbones. On four LongBench datasets at an 8K context length, Ask-dLLM achieves approximately 9.7× and 11.8× the throughput of Origin on LLaDA and Dream, respectively, while matching or exceeding its average task score.

Models: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) / [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B).

## Installation

```bash
pip install -r requirements.txt
```

## Code

- [Teacher extraction](teacher/extract_teacher.py)
- [Training](student/train_student.py)
- [Inference](inference/generate.py)

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
