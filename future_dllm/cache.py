from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sparse_dllm_current_score(
    q_block: torch.Tensor,
    candidate_k: torch.Tensor,
    pool_kernel_size: Optional[int] = 3,
) -> torch.Tensor:
    if q_block.ndim != 4 or candidate_k.ndim != 4:
        raise ValueError("q_block and candidate_k must both be rank-4 tensors")
    if q_block.size(0) != candidate_k.size(0):
        raise ValueError("q_block and candidate_k batch sizes do not match")
    if q_block.size(-1) != candidate_k.size(-1):
        raise ValueError("q_block and candidate_k head dimensions do not match")
    if q_block.size(1) != candidate_k.size(1):
        if q_block.size(1) % candidate_k.size(1):
            raise ValueError("query heads must be divisible by KV heads")
        candidate_k = candidate_k.repeat_interleave(
            q_block.size(1) // candidate_k.size(1), dim=1
        )

    average_query = q_block.mean(dim=-2)
    scores = torch.matmul(
        average_query.unsqueeze(-2), candidate_k.transpose(-2, -1)
    ).squeeze(-2)
    importance = scores.mean(dim=1)

    if pool_kernel_size is not None:
        if pool_kernel_size < 1 or pool_kernel_size % 2 == 0:
            raise ValueError("pool_kernel_size must be a positive odd integer or None")
        importance = F.max_pool1d(
            importance.unsqueeze(1),
            kernel_size=pool_kernel_size,
            stride=1,
            padding=pool_kernel_size // 2,
        ).squeeze(1)
    return importance


def teacher_current_attention_score(
    q_block: torch.Tensor,
    candidate_k: torch.Tensor,
    block_k: torch.Tensor,
    *,
    row_reduce: str = "max",
    group_reduce: str = "mean",
    per_head: bool = True,
) -> torch.Tensor:
    if q_block.ndim != 4 or candidate_k.ndim != 4 or block_k.ndim != 4:
        raise ValueError("q_block, candidate_k and block_k must be rank-4 tensors")
    if row_reduce not in ("max", "mean"):
        raise ValueError("row_reduce must be max or mean")
    if group_reduce not in ("max", "mean"):
        raise ValueError("group_reduce must be max or mean")
    if candidate_k.shape[:2] != block_k.shape[:2]:
        raise ValueError("candidate and block keys must have matching batch/KV-head axes")
    if q_block.size(0) != candidate_k.size(0):
        raise ValueError("query and key batch sizes do not match")
    if q_block.size(-1) != candidate_k.size(-1) or block_k.size(-1) != candidate_k.size(-1):
        raise ValueError("query and key head dimensions do not match")
    if q_block.size(-2) != block_k.size(-2):
        raise ValueError("q_block and block_k row counts do not match")

    kv_heads = candidate_k.size(1)
    if q_block.size(1) % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group = q_block.size(1) // kv_heads
    all_k = torch.cat([candidate_k, block_k], dim=-2)
    if group != 1:
        all_k = all_k.repeat_interleave(group, dim=1)

    logits = torch.matmul(q_block.float(), all_k.float().transpose(-2, -1))
    weights = torch.softmax(logits / (q_block.size(-1) ** 0.5), dim=-1)
    rows = weights[..., :candidate_k.size(-2)]

    if per_head:
        if group != 1:
            batch, _, n_rows, n_cols = rows.shape
            rows = rows.view(batch, kv_heads, group, n_rows, n_cols)
            rows = (rows.amax(dim=2) if group_reduce == "max"
                    else rows.mean(dim=2))
        return rows.amax(dim=-2) if row_reduce == "max" else rows.mean(dim=-2)

    rows = rows.mean(dim=1)
    return rows.amax(dim=-2) if row_reduce == "max" else rows.mean(dim=-2)


class CustomCache:

    def __init__(
        self,
        n_layers: int,
        device: torch.device,
        keep_ratio: float = 1.0,
        cache_scorer=None,
        prompt_length: int = 0,
        generation_length: int = 0,
        capture_current_scores: bool = False,
        current_score_pool_kernel: Optional[int] = 3,
        eviction_method: str = "student",
        current_reduce: Optional[tuple[str, str, bool]] = None,
        baseline_order: bool = False,
        accum_state: Optional[dict] = None,
        accum_decay: float = 1.0,
    ) -> None:
        if eviction_method not in ("student", "current", "sparse", "oracle"):
            raise ValueError("eviction_method must be student, current, sparse or oracle")
        if eviction_method == "current":
            current_reduce = current_reduce or ("max", "mean", True)
            row_reduce, group_reduce, _ = current_reduce
            if row_reduce not in ("max", "mean") or group_reduce not in ("max", "mean"):
                raise ValueError("current reductions must be max or mean")
        elif current_reduce is not None:
            raise ValueError("current_reduce requires eviction_method='current'")
        self.cache = {}
        self.accum_state = accum_state
        self.accum_decay = float(accum_decay)
        self.eviction_method = eviction_method
        self.current_reduce = current_reduce
        self.baseline_order = baseline_order
        self.candidate_order = {}
        self.keep_ratios = [keep_ratio for _ in range(n_layers)]
        self.cache_scorer = cache_scorer
        self.oracle_label = {}
        self.prompt_length = prompt_length
        self.generation_length = generation_length

        self.capture_current_scores = capture_current_scores
        self.current_score_pool_kernel = current_score_pool_kernel
        self.current_scores = {}
        self.current_teacher_scores = {}
        self.capture_current_reduce = ("max", "mean", True)

        self.layer_hidden_states = {}

        self.collect_pool = False
        self.capture_rows = False
        self.capture_per_head = True
        self.group_reduce = "mean"
        self.pending_rows = {}
        self.row_mask = None


    def set_row_mask(self, row_mask: Optional[torch.Tensor]) -> None:
        self.row_mask = row_mask

    def record_attention(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if not self.capture_rows:
            return
        kv_heads = k.size(1)
        group = q.size(1) // kv_heads
        if q.size(1) % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if group != 1:
            k = k.repeat_interleave(group, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        n_cand = k.size(-2) - q.size(-2)
        rows = weights[..., :n_cand]
        if self.capture_per_head:
            if group != 1:
                batch, _, n_rows, n_cols = rows.shape
                grouped = rows.view(batch, kv_heads, group, n_rows, n_cols)
                rows = (grouped.amax(dim=2) if self.group_reduce == "max"
                        else grouped.mean(dim=2))
            rows = rows.squeeze(0)
        else:
            rows = rows.mean(dim=1).squeeze(0)
        self.pending_rows[layer_id] = self._natural_order(layer_id, rows)

    def _natural_order(self, layer_id: int, rows: torch.Tensor) -> torch.Tensor:
        if layer_id not in self.candidate_order:
            return rows
        natural_rows = torch.empty_like(rows)
        natural_rows[..., self.candidate_order[layer_id]] = rows
        return natural_rows

    def capture_layer_hidden_states(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        if self.cache_scorer is not None:
            self.layer_hidden_states[layer_id] = hidden_states

    def get_cache(self, layer_id: int):
        return self.cache.get(layer_id, {"k": None, "v": None})

    def update_cache(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.cache[layer_id] = {"k": k.clone(), "v": v.clone()}

    def filter_cache(self, layer_id: int, q_block: torch.Tensor,
                     cur_filtered_len: int, block_len: int):
        cached = self.get_cache(layer_id)
        cached_k, cached_v = cached["k"], cached["v"]

        keep_k = torch.cat([cached_k[:, :, :cur_filtered_len, :],
                            cached_k[:, :, cur_filtered_len + block_len:, :]], dim=2)
        keep_v = torch.cat([cached_v[:, :, :cur_filtered_len, :],
                            cached_v[:, :, cur_filtered_len + block_len:, :]], dim=2)

        if self.capture_current_scores:
            self.current_scores[layer_id] = sparse_dllm_current_score(
                q_block, keep_k, self.current_score_pool_kernel
            ).detach()
            row_reduce, group_reduce, per_head = self.capture_current_reduce
            self.current_teacher_scores[layer_id] = teacher_current_attention_score(
                q_block,
                keep_k,
                cached_k[:, :, cur_filtered_len:cur_filtered_len + block_len, :],
                row_reduce=row_reduce, group_reduce=group_reduce, per_head=per_head,
            ).detach()

        full_pool = self.collect_pool or self.keep_ratios[layer_id] >= 1.0
        if self.eviction_method == "current" and not full_pool:
            block_k = cached_k[:, :, cur_filtered_len:cur_filtered_len + block_len, :]
            row_reduce, group_reduce, per_head = self.current_reduce
            scores = teacher_current_attention_score(
                q_block, keep_k, block_k, row_reduce=row_reduce,
                group_reduce=group_reduce, per_head=per_head)
            keep_num = int(keep_k.size(-2) * self.keep_ratios[layer_id])
            keep_indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0)
            if not self.baseline_order:
                keep_indices = keep_indices.sort(dim=-1).values
            head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
            self.cache[layer_id] = {
                "k": keep_k[:, head_index, keep_indices],
                "v": keep_v[:, head_index, keep_indices],
            }
            return

        if self.eviction_method == "sparse" or (full_pool and self.baseline_order):
            scores = sparse_dllm_current_score(q_block, keep_k, self.current_score_pool_kernel)
            keep_num = keep_k.size(-2) if full_pool else int(
                keep_k.size(-2) * self.keep_ratios[layer_id])
            indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0)
            if self.collect_pool:
                self.candidate_order[layer_id] = indices
            head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
            self.cache[layer_id] = {"k": keep_k[:, head_index, indices],
                                    "v": keep_v[:, head_index, indices]}
            return

        if full_pool:
            self.cache[layer_id] = {"k": keep_k, "v": keep_v}
            return

        if self.eviction_method == "oracle":
            scores = self.oracle_label.get(layer_id)
            if scores is None:
                raise RuntimeError(f"no oracle label for layer {layer_id}")
            if scores.shape[-1] != keep_k.size(-2):
                raise RuntimeError(
                    f"oracle label has {scores.shape[-1]} candidates but the "
                    f"cache holds {keep_k.size(-2)} at layer {layer_id}")
            scores = scores.float().unsqueeze(0)
        else:
            scores = None

        if scores is None and self.cache_scorer is None:
            raise RuntimeError(
                "future_dllm evicts with a trained scorer; pass a student "
                "checkpoint, or run with keep_ratio=1.0 to disable eviction"
            )

        if scores is not None:
            keep_num = int(keep_k.size(-2) * self.keep_ratios[layer_id])
            keep_indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0)
            if not self.baseline_order:
                keep_indices = keep_indices.sort(dim=-1).values
            head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
            self.cache[layer_id] = {"k": keep_k[:, head_index, keep_indices],
                                    "v": keep_v[:, head_index, keep_indices]}
            return

        hidden_states = self.layer_hidden_states.pop(layer_id, None)
        if hidden_states is None:
            raise RuntimeError(f"missing hidden states for scorer layer {layer_id}")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("scorer selection requires batch_size=1")
        sequence_length = int(hidden_states.shape[1])

        candidate_indices = torch.cat([
            torch.arange(cur_filtered_len, device=hidden_states.device),
            torch.arange(cur_filtered_len + block_len, sequence_length,
                         device=hidden_states.device),
        ])
        if candidate_indices.numel() != keep_k.size(-2):
            raise RuntimeError("scorer candidates do not match cached K/V")

        block_indices = torch.arange(cur_filtered_len, cur_filtered_len + block_len,
                                     device=hidden_states.device)

        scores = self.cache_scorer.forward_layer(
            layer_id, hidden_states.float(), candidate_indices,
            head="score", block_indices=block_indices).float()
        if scores.dim() == 3 and scores.shape[1] != keep_k.size(1):
            raise RuntimeError(
                f"scorer emits {scores.shape[1]} head scores but the cache has "
                f"{keep_k.size(1)} heads")
        if self.accum_state is not None:
            prior = self.accum_state.get(layer_id)
            if prior is None:
                prior = torch.zeros((*scores.shape[1:-1], sequence_length),
                                    device=scores.device, dtype=scores.dtype)
            prior = prior * self.accum_decay
            prior = prior.index_add(-1, candidate_indices,
                                    scores.softmax(-1).squeeze(0))
            self.accum_state[layer_id] = prior
            scores = prior.index_select(-1, candidate_indices).unsqueeze(0)

        keep_num = int(candidate_indices.numel() * self.keep_ratios[layer_id])
        keep_indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0)
        if not self.baseline_order:
            keep_indices = keep_indices.sort(dim=-1).values

        head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
        self.cache[layer_id] = {"k": keep_k[:, head_index, keep_indices],
                                "v": keep_v[:, head_index, keep_indices]}

    def clear(self):
        self.cache.clear()
