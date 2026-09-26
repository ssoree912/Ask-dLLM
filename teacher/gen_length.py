from __future__ import annotations

from pathlib import Path

import yaml


TASKS = Path(__file__).resolve().parent.parent / "eval" / "tasks"

DEFAULT_MAX_GEN_TOKS = 256

DATASET_TASK = {
    "musique": "longbench/musique.yaml",
    "gov_report": "longbench/gov_report.yaml",
    "multi_news": "longbench/multi_news.yaml",
    "gsm8k": "local/gsm8k.yaml",
    "math": "local/math.yaml",
    "math5s": "local/math.yaml",
}

NO_TASK_BUDGET = {
    "mbpp_full": 256,
}


def _load_yaml(path: Path) -> dict:
    loader = yaml.SafeLoader
    loader.add_constructor("!function", lambda l, n: l.construct_scalar(n))
    return yaml.load(path.read_text(), Loader=loader)


def resolve(dataset: str) -> tuple[int, str]:
    if dataset in NO_TASK_BUDGET:
        return NO_TASK_BUDGET[dataset], "no eval task generates; repo default"

    try:
        rel = DATASET_TASK[dataset]
    except KeyError:
        raise SystemExit(
            f"no eval task known for --dataset {dataset}; add it to "
            f"teacher/gen_length.py or pass --gen-length"
        ) from None

    cfg = _load_yaml(TASKS / rel)
    budget = (cfg.get("generation_kwargs") or {}).get("max_gen_toks")
    if budget is None:
        return DEFAULT_MAX_GEN_TOKS, f"{rel} sets none; lm-eval default"
    return int(budget), f"{rel} max_gen_toks"


if __name__ == "__main__":
    for name in sorted(set(DATASET_TASK) | set(NO_TASK_BUDGET)):
        value, source = resolve(name)
        print(f"{name:14s} {value:5d}   {source}")
