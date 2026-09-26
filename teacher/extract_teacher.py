"""Teacher targets from attention queries of the completed answer block."""
import torch

from inference.generate import BLOCK_LENGTH, candidate_indices, generate


@torch.no_grad()
def attention_targets(query, keys, candidates, per_head=False):
    """Q: [query_heads, block, dim]; K: [kv_heads, sequence, dim].

    Softmax includes the block itself; only then are candidate columns kept.
    Default: average query heads, then take the maximum over block rows.
    Per-head: max over query heads sharing a KV head, then max over rows.
    """
    query_heads, kv_heads = query.shape[0], keys.shape[0]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    repeated_keys = keys.repeat_interleave(groups, dim=0)
    logits = query.float() @ repeated_keys.float().transpose(-2, -1)
    attention = (logits / query.shape[-1] ** 0.5).softmax(dim=-1)
    rows = attention.index_select(-1, candidates)
    if per_head:
        rows = rows.reshape(kv_heads, groups, query.shape[1], -1).amax(dim=1)
    else:
        rows = rows.mean(dim=0, keepdim=True)
    return rows.amax(dim=-2)  # [1 or kv_heads, candidates]


@torch.no_grad()
def extract_teacher(decoder, prompt, family, gen_length, per_head=False):
    """Decode with the full cache and save one training record per block."""
    records = []

    def completed(tokens, selection_input, start, cache):
        candidates = candidate_indices(tokens.numel(), start, tokens.device)
        labels = [attention_targets(q, k, candidates, per_head)
                  for q, k in decoder.completed_qk(tokens, start, cache)]
        records.append({
            "x_at_block_start": selection_input.cpu().clone(),
            "block_start": start,
            "block_length": BLOCK_LENGTH,
            "candidate_indices": candidates.cpu(),
            "label_final_rowmax": torch.stack(labels).half().cpu(),
        })

    generate(decoder, prompt, family, gen_length,
             keep_ratio=1.0, on_block_complete=completed)
    return records
