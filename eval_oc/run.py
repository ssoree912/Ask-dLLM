'Run the multiple-choice suite (ARC-C, PIQA, GPQA, MMLU) through OpenCompass.'
import argparse
import sys
from pathlib import Path

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TASKS = ("arc_c", "piqa", "gpqa", "mmlu")
DATA_PATH = {
    "arc_c": "arc_c/ARC-Challenge-Test.jsonl",
    "piqa": "piqa",
    "gpqa": "gpqa",
    "mmlu": "mmlu",
}


def task_of(abbr):
    if abbr == "ARC-c-test":
        return "arc_c"
    if abbr == "piqa":
        return "piqa"
    if abbr.startswith("GPQA_"):
        return "gpqa"
    if abbr.startswith("lukaemon_mmlu_"):
        return "mmlu"
    raise ValueError(f"unexpected OpenCompass dataset {abbr}")


def build_config(args):
    cfg = Config.fromfile(str(REPO_ROOT / f"eval_oc/configs/eval_{args.family}.py"))
    data_root = Path(args.data_root).resolve() / "eval"
    datasets = []
    for data in cfg.datasets:
        task = task_of(data["abbr"])
        if task not in args.tasks:
            continue
        data["path"] = str(data_root / DATA_PATH[task])
        if args.limit is not None:
            data["reader_cfg"]["test_range"] = f"[0:{args.limit}]"
        datasets.append(data)
    cfg.datasets = datasets

    model = cfg.models[0]
    model.update(path=str(Path(args.model).resolve()), keep_ratio=args.keep_ratio,
                 student_path=str(Path(args.student).resolve()) if args.student else "")
    model["abbr"] = f"{args.family}-keep{args.keep_ratio:g}"
    if args.family == "dream":
        model.update(alg=args.dream_alg, temperature=args.dream_temperature,
                     top_p=args.dream_top_p, steps=args.dream_steps, seed=args.dream_seed)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("llada", "dream"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--keep-ratio", type=float, default=1.0)
    p.add_argument("--student", default="", help="student checkpoint directory")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--work-dir", required=True)
    p.add_argument("--limit", type=int, help="first N questions of each dataset "
                                             "(each MMLU subject counts as one)")
    p.add_argument("--dream-alg", default="entropy")
    p.add_argument("--dream-temperature", type=float, default=0.2)
    p.add_argument("--dream-top-p", type=float, default=0.95)
    p.add_argument("--dream-steps", type=int, default=512)
    p.add_argument("--dream-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="write the config and stop")
    args = p.parse_args()
    if not 0.0 < args.keep_ratio <= 1.0:
        raise SystemExit("--keep-ratio must be in (0, 1]")
    if args.keep_ratio < 1.0 and not args.student:
        raise SystemExit("--keep-ratio < 1 requires --student")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    cfg = build_config(args)
    cfg.work_dir = str(Path(args.work_dir).resolve())
    config_path = Path(cfg.work_dir) / "run_config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(config_path))
    print(f"OpenCompass config: {config_path}", flush=True)
    if not args.dry_run:
        from opencompass.cli.main import main as run_opencompass
        sys.argv = [sys.argv[0], str(config_path), "--debug"]
        run_opencompass()


if __name__ == "__main__":
    main()
