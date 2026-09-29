'Train a scorer to predict the final x row-max label.'

from __future__ import annotations

import argparse, glob, json, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ask_dllm import (CustomCache, PromptUtilityStudent, StudentConfig,  # noqa: E402
                      detect_family, load_model)
from ask_dllm.dream_decoding import (  # noqa: E402
    DreamDecoding, add_dream_arguments, require_matching_decoding)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_dream_arguments(p)
    p.add_argument("--model", required=True,
                   help="the checkpoint the teacher labels were extracted with. "
                        "Required, and checked against the shards: the replay "
                        "forward has to reproduce the hidden states the selection "
                        "was made from, so a different model trains on states "
                        "deployment never sees")
    p.add_argument("--teacher-root", required=True,
                   help="comma-separated for mixed-domain training: val is split "
                        "per domain and the checkpoint is chosen on the domain "
                        "macro average, so a block-heavy domain cannot own it")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--block-length", type=int, default=32,
                   help="must match the teacher run; only used to size the "
                        "replay forward's cache window")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="total token budget: LLaDA 4096, Dream 2048")
    p.add_argument("--lambda-list", type=float, default=1.0,
                   help="weight on the listwise KL against the pairwise term")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-shards", default="",
                   help="comma list aligned with --teacher-root: use only the first "
                        "N prompts of each domain, 0 = all. The cut is taken before "
                        "the val split, "
                        "so val stays the same fraction of what is used.")
    return p.parse_args()


def recall_grid(pred, target, ratios=(0.05, 0.1, 0.2, 0.3, 0.5)):
    p = pred if pred.dim() == 2 else pred.unsqueeze(0)
    t = target if target.dim() == 2 else target.unsqueeze(0)
    n = t.shape[-1]
    out = []
    for r in ratios:
        k = max(1, int(n * r))
        chosen = p.topk(k, dim=-1).indices
        mark = torch.zeros_like(t, dtype=torch.bool)
        mark.scatter_(-1, t.topk(k, dim=-1).indices, True)
        out.append(mark.gather(-1, chosen).sum(-1).float().mean() / k)
    return float(sum(out) / len(out))


def window_start(record):
    return int(record.get("window_start", record["block_start"]))


def window_length(record):
    return int(record.get("window_length", record["block_length"]))


def load_shard(path, attempts=3):
    for i in range(attempts):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    args = parse_args()
    if args.max_seq_len is None:
        args.max_seq_len = 2048 if detect_family(args.model) == "dream" else 4096
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    torch.manual_seed(args.seed); random.seed(args.seed)

    model, backend = load_model(args.model, max_seq_len=args.max_seq_len,
                                block_length=args.block_length, keep_ratio=1.0)
    if args.max_seq_len > backend.native_max_seq_len:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds the checkpoint's "
              f"trained context {backend.native_max_seq_len}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    device, L, H = model.device, backend.n_layers, backend.hidden_dim
    print(f"backend={backend.name} layers={L} hidden={H}", flush=True)
    decoding = None
    if backend.name == "dream":
        decoding = DreamDecoding.from_args(args).metadata()

    def read_teacher(path):
        shard = load_shard(path)
        if decoding is not None:
            require_matching_decoding(shard.get("decoding"), decoding, path)
        return shard

    roots = [r for r in args.teacher_root.split(",") if r]
    shard_caps = [int(c) for c in args.max_shards.split(",")] if args.max_shards else []
    if shard_caps and len(shard_caps) != len(roots):
        raise SystemExit("--max-shards needs one entry per teacher root")
    train_shards, val_shards, datasets = [], [], []
    kinds, label_heads = set(), set()
    for index, root in enumerate(roots):
        name = Path(root).name
        found = sorted(glob.glob(f"{root}/*.pt"))
        if not found:
            raise SystemExit(f"no teacher shards under {root}")
        cap = shard_caps[index] if shard_caps else 0
        if cap:
            if cap > len(found):
                raise SystemExit(f"{name}: asked for {cap} prompts, only {len(found)} exist")
            found = found[:cap]
        head = read_teacher(found[0])
        shard_backend = head.get("backend")
        if shard_backend is not None and shard_backend != backend.name:
            raise SystemExit(
                f"{name}: teacher labels were extracted with {shard_backend} "
                f"({head.get('model', 'unknown checkpoint')}), but --model is a "
                f"{backend.name} checkpoint. Pass the model the labels came from."
            )
        kinds.add(head.get("teacher_kind", "final_rowmax"))
        if "num_label_heads" in head:
            label_heads.add(int(head["num_label_heads"]))
        split = max(1, int(len(found) * args.val_ratio))
        val_shards += [(name, p) for p in found[:split]]
        train_shards += [(name, p) for p in found[split:]]
        datasets.append(name)
        print(f"  {name}: train {len(found)-split} / val {split} shards"
              f"{'' if shard_backend is None else f' [{shard_backend}]'}", flush=True)
    print(f"train {len(train_shards)} / val {len(val_shards)} shards "
          f"over {len(datasets)} domain(s)", flush=True)

    if len(kinds) != 1 or len(label_heads) > 1:
        raise SystemExit(f"teacher roots disagree on the label: kinds={sorted(kinds)} "
                         f"heads={sorted(label_heads)}")
    teacher_kind = sorted(kinds)[0]

    counts = [sum(1 for n, _ in train_shards + val_shards if n == d) for d in datasets]
    out_dir = Path(args.output_dir)
    print(f"checkpoint -> {out_dir}", flush=True)

    probe = load_shard(train_shards[0][1])["blocks"][0]["label_final_rowmax"]
    if probe.dim() != 3 or probe.shape[1] != backend.kv_heads:
        raise SystemExit(f"teacher labels have shape {tuple(probe.shape)}; expected "
                         f"(layers, {backend.kv_heads} KV heads, candidates)")
    attn_heads = K = int(probe.shape[1])
    print(f"teacher_kind={teacher_kind}: {K} scores per candidate (one per KV head)",
          flush=True)

    student_cfg = StudentConfig(attn_heads=attn_heads, layer_count=L, hidden_dim=H,
                                proj_dim=args.proj_dim, mlp_dim=args.mlp_dim)
    student = PromptUtilityStudent(student_cfg).to(device).float()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"datasets": datasets, "samples": dict(zip(datasets, counts)),
               "backend": backend.name, "model": str(args.model),
               "decoding": decoding,
               "block_length": args.block_length,
               "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
               "proj_dim": args.proj_dim, "mlp_dim": args.mlp_dim,
               "attn_heads": attn_heads,
               "val_ratio": args.val_ratio, "pairs": args.pairs,
               "max_seq_len": args.max_seq_len,
               "lambda_list": args.lambda_list,
               "teacher_roots": roots,
               "teacher_kind": teacher_kind,
               "max_shards": dict(zip(datasets, shard_caps)) if shard_caps else {}},
              open(out_dir / "meta.json", "w"), indent=2)

    best = -1.0

    @torch.no_grad()
    def features(record):
        sequence_length = int(record["x_at_block_start"].numel())
        if sequence_length > args.max_seq_len:
            raise RuntimeError(
                f"teacher record total length {sequence_length} exceeds "
                f"--max-seq-len {args.max_seq_len}; use a matching student limit"
            )
        x = record["x_at_block_start"].unsqueeze(0).to(device)
        cache = CustomCache(n_layers=L, device=device, keep_ratio=1.0,
                            capture_hidden_states=True)
        model(x, window_start(record), 1, cache)
        return cache.layer_hidden_states

    def step(record, train: bool):
        hidden = features(record)
        cand = record["candidate_indices"].to(device)
        ws = window_start(record)
        blk = torch.arange(ws, ws + window_length(record), device=device)
        label = record["label_final_rowmax"].float().to(device)
        total, recalls = 0.0, []
        for l in range(L):
            h = hidden[l].float()
            pred = student.forward_layer(l, h, cand, blk).squeeze(0)
            tgt = label[l]
            rows_t = tgt if tgt.dim() == 2 else tgt.unsqueeze(0)
            rows_p = pred if pred.dim() == 2 else pred.unsqueeze(0)
            mass = rows_t.sum(-1)
            usable = torch.isfinite(rows_t).all(-1) & (mass > 0)
            if not bool(usable.any()):
                continue
            rows_t, rows_p = rows_t[usable], rows_p[usable]
            kl = F.kl_div(F.log_softmax(rows_p, -1),
                          rows_t / rows_t.sum(-1, keepdim=True),
                          reduction="none").sum(-1)
            loss = args.lambda_list * kl.mean()
            i = torch.randint(0, rows_t.shape[-1], (args.pairs,), device=device)
            j = torch.randint(0, rows_t.shape[-1], (args.pairs,), device=device)
            sign = torch.sign(rows_t[..., i] - rows_t[..., j])
            keep = sign != 0
            if keep.any():
                diff = rows_p[..., i] - rows_p[..., j]
                loss = loss + F.softplus(-sign[keep] * diff[keep]).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += float(loss.detach())
            recalls.append(recall_grid(rows_p.detach(), rows_t))
        return total / max(1, L), sum(recalls) / max(1, len(recalls))

    for epoch in range(args.epochs):
        student.train(); random.shuffle(train_shards)
        losses = []
        for n, (_, path) in enumerate(train_shards):
            for record in read_teacher(path)["blocks"]:
                losses.append(step(record, True)[0])
            if (n + 1) % 30 == 0:
                print(f"  epoch {epoch} {n+1}/{len(train_shards)} "
                      f"loss {sum(losses[-120:])/max(1,len(losses[-120:])):.4f}", flush=True)
        student.eval()
        per_ds = {}
        with torch.no_grad():
            for name, p in val_shards:
                for r in read_teacher(p)["blocks"]:
                    per_ds.setdefault(name, []).append(step(r, False)[1])
        means = {k: sum(v) / max(1, len(v)) for k, v in per_ds.items()}
        score = sum(means.values()) / max(1, len(means))
        detail = "  ".join(f"{k} {v:.3f}" for k, v in sorted(means.items()))
        print(f"epoch {epoch}: loss {sum(losses)/len(losses):.4f} | "
              f"val recall macro {score:.4f} [{detail}]", flush=True)
        if score > best:
            best = score
            ckpt = out_dir / "checkpoint-best"
            student.save(ckpt)
            if decoding is not None:
                with open(ckpt / "decoding.json", "w") as fh:
                    json.dump(decoding, fh, indent=2)
            json.dump({"val_recall": score, "val_recall_per_dataset": means,
                       "epoch": epoch, "datasets": datasets, "kv_heads": K},
                      open(out_dir / "best.json", "w"))
            print(f"  saved (best {best:.4f})", flush=True)
    print(f"done. best val recall {best:.4f} -> {out_dir}/checkpoint-best", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
