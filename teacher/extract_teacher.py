'Extract teacher labels: the attention each finished block pays to its cached candidates.'

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ask_dllm import CustomCache, detect_family, load_model  # noqa: E402
from ask_dllm.dream_decoding import (  # noqa: E402
    DreamDecoding, add_dream_arguments, require_matching_decoding, sample_seed)
from ask_dllm.llada_generate import add_gumbel_noise, get_num_transfer_tokens  # noqa: E402
from teacher.build_prompt_shards import GEN_LENGTH  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="LLaDA or Dream checkpoint to label with; teacher labels are "
                        "only meaningful for the model that produced them")
    p.add_argument("--dataset", required=True,
                   help="prompt shard directory name, e.g. math5s / mbpp_full / musique")
    p.add_argument("--shard-root", required=True,
                   help="directory written by teacher/build_prompt_shards.py")
    p.add_argument("--output-root", required=True)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--gen-length", type=int, default=None,
                   help="default: GEN_LENGTH[dataset] in teacher/build_prompt_shards.py")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="total token budget; default LLaDA 4096, Dream 2048")
    p.add_argument("--max-prompt-len", type=int, default=None,
                   help="optional stricter prompt-only cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-row-reduce", choices=("max", "mean"), default="max",
                   help="how the block's rows become one number per candidate. "
                        "max is what the label shipped with: a candidate "
                        "survives if any finished token needed it")
    p.add_argument("--label-group-reduce", choices=("max", "mean"), default="mean",
                   help="how the query heads sharing a KV entry are folded, on "
                        "a GQA backend. Ignored when --per-head is off")
    p.add_argument("--per-head", action=argparse.BooleanOptionalAction, default=True,
                   help="label every attention head separately instead of "
                        "averaging them. Needed for per-head eviction, where "
                        "each head keeps its own top-k; costs H times the "
                        "storage, so it writes its own teacher_kind and is "
                        "meant for a separate --output-root")
    add_dream_arguments(p)
    args = p.parse_args()

    args.family = detect_family(args.model)
    if args.max_seq_len is None:
        args.max_seq_len = 2048 if args.family == "dream" else 4096
    if args.gen_length is None:
        if args.dataset not in GEN_LENGTH:
            raise SystemExit(f"no default generation length for {args.dataset}; "
                             "pass --gen-length")
        args.gen_length = GEN_LENGTH[args.dataset]
    if args.gen_length < 1:
        raise SystemExit("--gen-length must be positive")
    if args.block_length < 1:
        raise SystemExit("--block-length must be positive")
    if args.gen_length % args.block_length:
        raise SystemExit(f"gen_length {args.gen_length} is not a multiple of "
                         f"block_length {args.block_length}")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    available = args.max_seq_len - args.gen_length
    if available < 1:
        raise SystemExit(
            f"generation length {args.gen_length} leaves no prompt space within "
            f"--max-seq-len {args.max_seq_len}"
        )
    if args.max_prompt_len is not None and args.max_prompt_len < 1:
        raise SystemExit("--max-prompt-len must be positive")
    args.prompt_limit = min(available, args.max_prompt_len or available)
    return args


@torch.no_grad()
def collect(model, prompt_ids, args, backend):
    if backend.name == "dream":
        return collect_dream(model, prompt_ids, args, backend)

    device = model.device
    prompt_ids = prompt_ids[-args.prompt_limit:].to(device).unsqueeze(0)
    P = prompt_ids.shape[1]
    G, B = args.gen_length, args.block_length
    n_blocks = G // B
    S = args.gen_length // n_blocks
    L, MASK_ID = backend.n_layers, backend.mask_id

    x = torch.full((1, P + G), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    records = []

    for block in range(n_blocks):
        cache = CustomCache(n_layers=L, device=device, keep_ratio=1.0)
        cache.collect_pool = True
        bs, be = P + block * B, P + (block + 1) * B
        ntt = get_num_transfer_tokens(x[:, bs:be] == MASK_ID, S)

        def step(i):
            state = 2 if i > 1 else i
            inp = x if state != 2 else x[:, bs:be]
            m = (inp == MASK_ID)
            logits = model(inp, bs, state, cache).logits
            if backend.logit_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            x0 = torch.argmax(add_gumbel_noise(logits, 0.0), dim=-1)
            conf = torch.squeeze(torch.gather(F.softmax(logits, -1), -1,
                                              x0.unsqueeze(-1)), -1)
            tgt = x if state != 2 else x[:, bs:be]
            if state != 2:
                conf[:, be:] = -float("inf")
            x0 = torch.where(m, x0, tgt)
            conf = torch.where(m, conf, torch.full_like(conf, -float("inf")))
            if state == 0 and backend.seed_block_start:
                conf[:, bs] = float("inf")
            keep = torch.topk(conf[0], k=ntt[0, i]).indices
            tgt[0, keep] = x0[0, keep]

        step(0)
        x_at_block_start = x.clone()
        step(1)
        for i in range(2, S):
            step(i)

        cache.capture_rows = True
        cache.capture_per_head = args.per_head
        cache.group_reduce = args.label_group_reduce
        step(S - 1)
        row_axis = 1 if cache.capture_per_head else 0
        row_reduce = args.label_row_reduce
        label = torch.stack([
            reduce_rows(cache.pending_rows[layer], row_axis, row_reduce)
            for layer in range(L)
        ])
        cache.pending_rows.clear()
        cache.capture_rows = False

        candidates = torch.cat([torch.arange(bs, device=device),
                                torch.arange(be, x.shape[1], device=device)])
        record = {
            "block_index": block,
            "block_start": int(bs),
            "block_length": B,
            "window_start": int(bs),
            "window_length": int(B),
            "seed_block_start": bool(backend.seed_block_start),
            "backend": backend.name,
            "prompt_length": int(P),
            "gen_length": G,
            "steps_per_block": S,
            "x_at_block_start": x_at_block_start[0].cpu(),
            "candidate_indices": candidates.cpu(),
            "label_final_rowmax": label.to(torch.float16).cpu(),
        }
        records.append(record)
    return records


@torch.no_grad()
def reduce_rows(rows: torch.Tensor, axis: int, how: str) -> torch.Tensor:
    if how == "max":
        return rows.max(dim=axis).values
    if how == "mean":
        return rows.mean(dim=axis)
    raise ValueError(f"unknown row reduction: {how}")


def collect_dream(model, prompt_ids, args, backend):
    from ask_dllm.dream_generate import generate

    settings = DreamDecoding.from_args(args)
    prompt = prompt_ids[-args.prompt_limit:].to(model.device).unsqueeze(0)
    records = []

    per_head = args.per_head
    row_reduce = args.label_row_reduce
    group_reduce = args.label_group_reduce

    def completed(x, cache, block_index, bs, selection_input):
        be = bs + args.block_length
        cache.capture_rows = True
        cache.capture_per_head = per_head
        cache.group_reduce = group_reduce
        model(input_ids=x[:, bs:be], position_offset=bs, cache_state=2,
              customcache=cache, attention_mask="full")
        cache.capture_rows = False
        rows = [cache.pending_rows[layer] for layer in range(backend.n_layers)]
        row_axis = 1 if per_head else 0
        label = torch.stack([reduce_rows(row, row_axis, row_reduce) for row in rows])
        candidates = torch.cat([torch.arange(bs), torch.arange(be, x.shape[1])])
        record = {
            "block_index": block_index, "block_start": bs,
            "block_length": args.block_length, "window_start": bs,
            "window_length": args.block_length, "seed_block_start": True,
            "backend": "dream", "prompt_length": prompt.shape[1],
            "gen_length": args.gen_length,
            "steps_per_block": settings.steps_for_length(args.gen_length)
                               // (args.gen_length // args.block_length),
            "x_at_block_start": selection_input[0].cpu(),
            "candidate_indices": candidates,
            "completed_block_ids": x[0, bs:be].cpu().clone(),
            "label_final_rowmax": label.to(torch.float16).cpu(),
        }
        cache.pending_rows.clear()
        records.append(record)

    generate(model, prompt, gen_length=args.gen_length, block_length=args.block_length,
             mask_id=backend.mask_id, on_block_complete=completed,
             **settings.generation_kwargs(args.gen_length))
    return records


def main():
    args = parse_args()

    model, backend = load_model(args.model, max_seq_len=args.max_seq_len,
                                block_length=args.block_length, keep_ratio=1.0)
    if args.max_seq_len > backend.native_max_seq_len:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds the checkpoint's "
              f"trained context {backend.native_max_seq_len}", flush=True)
    print(f"backend={backend.name} layers={backend.n_layers} "
          f"mask_id={backend.mask_id} logit_shift={backend.logit_shift} "
          f"seed_block_start={backend.seed_block_start}", flush=True)
    decoding = None
    if args.family == "dream":
        decoding = DreamDecoding.from_args(args).metadata()
        print(f"decoding={decoding} seed={args.seed}", flush=True)

    per_head = args.per_head
    row_reduce = args.label_row_reduce
    group_reduce = args.label_group_reduce
    teacher_kind = f"final_row{row_reduce}" + ("_per_head" if per_head else "")
    if per_head and group_reduce != "max":
        teacher_kind += f"_group{group_reduce}"

    out = Path(args.output_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"{args.shard_root}/{args.dataset}/*.pt"))[: args.n_samples]
    if not shards:
        raise SystemExit(f"no prompt shards under {args.shard_root}/{args.dataset} "
                         f"- run teacher/build_prompt_shards.py first")
    added = 0
    for i, path in enumerate(shards):
        target = out / Path(path).name
        src = torch.load(path, map_location="cpu", weights_only=False)
        prompt_ids = src["prompt_input_ids"].to(torch.long)
        expected_prompt_len = min(prompt_ids.numel(), args.prompt_limit)

        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            if decoding is not None:
                require_matching_decoding(saved.get("decoding"), decoding, target)
                if saved.get("seed") != args.seed:
                    raise ValueError(f"{target}: teacher seed differs; use a new output root")
            if saved.get("teacher_kind") != teacher_kind:
                print(f"rebuilding teacher shard from "
                      f"{saved.get('teacher_kind')!r} to {teacher_kind!r}: "
                      f"{target.name}", flush=True)
                saved = None
            blocks = (saved.get("blocks") or []) if saved is not None else []
            if (blocks
                    and saved.get("backend", backend.name) == backend.name
                    and all(int(r.get("prompt_length", -1)) == expected_prompt_len
                            and int(r.get("gen_length", -1)) == args.gen_length
                            and r["x_at_block_start"].numel()
                            == expected_prompt_len + args.gen_length
                            for r in blocks)):
                continue
            print(f"rebuilding mismatched teacher shard: {target.name}", flush=True)
        added += 1
        if decoding is not None:
            item_seed = sample_seed(args.seed, f"{args.dataset}/{Path(path).stem}")
            with torch.random.fork_rng():
                torch.manual_seed(item_seed)
                records = collect(model, prompt_ids, args, backend)
        else:
            records = collect(model, prompt_ids, args, backend)
        payload = {"sample_id": src.get("sample_id"),
                   "dataset": args.dataset,
                   "backend": backend.name,
                   "model": str(args.model),
                   "prompt_input_ids": prompt_ids,
                   "prompt_limit": args.prompt_limit,
                   "gen_length": args.gen_length,
                   "max_seq_len": args.max_seq_len,
                   "teacher_kind": teacher_kind,
                   "blocks": records}
        if per_head and records:
            payload.update(
                per_head_axis="kv_heads",
                per_head_group_reduce=group_reduce,
                num_label_heads=int(records[0]["label_final_rowmax"].shape[1]),
            )
        if decoding is not None:
            payload.update(decoding=decoding, seed=args.seed, sample_seed=item_seed)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(shards)}",
                  flush=True)
    print(f"done: {len(list(out.glob('*.pt')))} shards total, {added} new -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
