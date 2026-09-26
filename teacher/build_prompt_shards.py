from __future__ import annotations

import argparse, glob, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FUTURE_DLLM_DATA", REPO_ROOT / "data"))

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_length import resolve as resolve_gen_length

MATH_INSTRUCTION = ("Please reason step by step, and put your final answer within "
                    "\\boxed{}.")


def _drop_test_overlap(table, column, test_glob):
    import pandas as pd, re
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(test_glob)))]
    if not frames:
        return table
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    seen = {norm(v) for v in pd.concat(frames)[column]}
    keep = [norm(v) not in seen for v in table[column]]
    dropped = len(keep) - sum(keep)
    if dropped:
        print(f"  dropped {dropped} rows that also appear in test", flush=True)
    return table[keep]


def _balanced(table, columns, limit, seed=0):
    import pandas as pd
    if not columns or not all(c in table.columns for c in columns):
        return table.sample(frac=1.0, random_state=seed).head(limit)
    groups = [g.sample(frac=1.0, random_state=seed).to_dict("records")
              for _, g in table.groupby(list(columns), sort=True)]
    out, i = [], 0
    while len(out) < limit and any(groups):
        g = groups[i % len(groups)]
        if g:
            out.append(g.pop())
        i += 1
        if i % max(1, len(groups)) == 0:
            groups = [g for g in groups if g]
            i = 0
            if not groups:
                break
    return pd.DataFrame(out[:limit])


def gsm8k(limit):
    import pandas as pd
    table = _balanced(pd.read_parquet(DATA / "train/gsm8k/train.parquet"), [], limit)
    return [(f"gsm8k-{i}", f"{row.question}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def musique(limit):
    import random
    rows = [json.loads(l) for l in open(DATA / "train/musique/musique_ans_v1.0_train.jsonl")]
    random.Random(0).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:limit]):
        ctx = "\n".join(f"Passage {j+1}:\n{p['title']}\n{p['paragraph_text']}"
                        for j, p in enumerate(r["paragraphs"]))
        out.append((f"musique-{i}",
                    "Answer the question based on the given passages. Only give me the "
                    "answer and do not output any other words.\n\nThe following are given "
                    f"passages.\n{ctx}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\n"
                    f"Question: {r['question']}\nAnswer:"))
    return out


def gov_report(limit):
    import pandas as pd
    table = pd.read_parquet(DATA / "train/gov_report/train.parquet")
    table = table.sample(frac=1.0, random_state=0).head(limit)
    return [(f"gov_report-{i}",
             "You are given a report by a government agency. Write a one-page summary "
             f"of the report.\n\nReport:\n{row.report}\n\nNow, write a one-page summary "
             "of the report.\n\nSummary:")
            for i, row in enumerate(table.itertuples())]


def multi_news(limit):
    import pandas as pd
    table = pd.read_parquet(DATA / "train/multi_news/train.parquet")
    table = table.sample(frac=1.0, random_state=0).head(limit)
    return [(f"multinews-{i}",
             "You are given several news passages. Write a one-page summary of all news. "
             f"\n\nNews:\n{row.document}\n\nNow, write a one-page summary of all the news."
             "\n\nSummary:")
            for i, row in enumerate(table.itertuples())]


def math(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob(str(DATA / "train/hendrycks_math/*/train-*.parquet")))]
    table = _balanced(pd.concat(frames), ["type", "level"], limit)
    return [(f"math-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


MATH5S_TRAIN = ["Prealgebra", "Algebra", "Geometry", "Number Theory", "Precalculus"]


def _math_subjects(subjects, limit, prefix):
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob(str(DATA / "train/hendrycks_math/*/train-*.parquet")))]
    table = pd.concat(frames)
    table = table[~table["problem"].str.contains(r"\[asy\]", regex=True)]
    table = table[table["type"].isin(subjects)]
    table = _balanced(table, ["type", "level"], limit)
    return [(f"{prefix}-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def math5s(limit):
    return _math_subjects(MATH5S_TRAIN, limit, "math5s")


def mbpp_full(limit):
    import pandas as pd
    table = _drop_test_overlap(
        pd.read_parquet(DATA / "train/mbpp/full/train-00000-of-00001.parquet"),
        "text", DATA / "eval/mbpp/full/test-*.parquet")
    table = _balanced(table, [], limit)
    return [(f"mbppfull-{i}",
             f"You are an expert Python programmer. {row.text}\n"
             f"Your code should pass these tests:\n" + "\n".join(row.test_list) + "\n")
            for i, row in enumerate(table.itertuples())]


BUILDERS = {
    "math5s": math5s, "mbpp_full": mbpp_full, "gov_report": gov_report,
    "multi_news": multi_news, "musique": musique, "gsm8k": gsm8k, "math": math,
}

RAW_TEXT = {"musique", "gov_report", "multi_news"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--model", required=True,
                   help="tokenizer source; required so a run always says which "
                        "model it tokenised for. Shards are NOT interchangeable "
                        "between families: Dream's Qwen2 chat template and "
                        "vocabulary give different ids and lengths than LLaDA's, "
                        "and the resume check only compares lengths, so give each "
                        "family its own --out-root")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="total token budget: LLaDA 4096, Dream 2048")
    p.add_argument("--gen-length", type=int, default=None,
                   help="default: the matching eval task's generation budget")
    p.add_argument("--chat-template", type=int, default=-1,
                   help="-1 selects the dataset default (disabled for LongBench)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    args = p.parse_args()

    if args.max_seq_len is None:
        sys.path.insert(0, str(REPO_ROOT))
        from future_dllm import detect_family
        args.max_seq_len = 2048 if detect_family(args.model) == "dream" else 4096
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    if args.gen_length is None:
        args.gen_length, gen_source = resolve_gen_length(args.dataset)
    else:
        gen_source = "--gen-length"
    if args.gen_length < 1:
        raise SystemExit("--gen-length must be positive")
    prompt_limit = args.max_seq_len - args.gen_length
    if prompt_limit < 1:
        raise SystemExit(
            f"generation length {args.gen_length} leaves no prompt space within "
            f"--max-seq-len {args.max_seq_len}"
        )

    chat = (args.dataset not in RAW_TEXT) if args.chat_template < 0 else bool(args.chat_template)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.truncation_side = "left"
    out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    added = rebuilt = 0
    for sid, text in BUILDERS[args.dataset](args.limit):
        target = out / f"{sid}.pt"
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            if (saved.get("max_seq_len") == args.max_seq_len
                    and saved.get("gen_length") == args.gen_length
                    and saved.get("prompt_limit") == prompt_limit):
                continue
            rebuilt += 1
        else:
            added += 1
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=not chat,
                  truncation=True, max_length=prompt_limit).input_ids[0]
        payload = {"sample_id": sid, "dataset": args.dataset,
                   "prompt_input_ids": ids.to(torch.long),
                   "prompt_limit": prompt_limit,
                   "gen_length": args.gen_length,
                   "max_seq_len": args.max_seq_len}
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
    n = len(list(out.glob("*.pt")))
    print(f"{args.dataset}: {n} shards total, {added} new, {rebuilt} rebuilt "
          f"(prompt_limit={prompt_limit}, gen_length={args.gen_length} from "
          f"{gen_source}, max_seq_len={args.max_seq_len}, chat_template={chat}) "
          f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
