'Select the OpenCompass datasets and run one model in a single process.'
import argparse
import os
from pathlib import Path
import sys

from mmengine.config import Config

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def build_config(family, dataset="all", limit=None):
    cfg = Config.fromfile(str(REPO / f"eval_oc/configs/eval_{family}_mc.py"))
    names = {"arc_c": "ARC-c-test", "piqa": "piqa", "gpqa": "GPQA_diamond_5shot"}
    if dataset != "all":
        cfg.datasets = [d for d in cfg.datasets if d["abbr"] == names[dataset]]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        for data in cfg.datasets:
            data["reader_cfg"]["test_range"] = f"[0:{limit}]"
    model = cfg.models[0]
    model["keep_ratio"] = float(os.environ.get("FUTURE_DLLM_KEEP_RATIO", "1.0"))
    if not 0.0 < model["keep_ratio"] <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    model["abbr"] = f"{family}-ask-k{model['keep_ratio']:g}"
    model["path"] = os.environ.get("FUTURE_DLLM_MODEL", "")
    model["student_path"] = os.environ.get("FUTURE_DLLM_STUDENT", "")
    if family == "dream":
        model.update(
            dream_alg=os.environ.get("DREAM_ALG", "entropy"),
            dream_temperature=float(os.environ.get("DREAM_TEMPERATURE", "0.2")),
            dream_top_p=float(os.environ.get("DREAM_TOP_P", "0.95")),
            dream_steps=int(os.environ.get("DREAM_STEPS", "512")),
            dream_seed=int(os.environ.get("DREAM_SEED", "0")),
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("llada", "dream"))
    parser.add_argument("--dataset", choices=("all", "arc_c", "piqa", "gpqa"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="write the config without loading a model")
    args = parser.parse_args()
    cfg = build_config(args.family, args.dataset, args.limit)
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
