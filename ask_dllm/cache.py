from __future__ import annotations

import torch
import torch.nn.functional as F


def block_attention_score(q_block: torch.Tensor, candidate_k: torch.Tensor,
                          pool_kernel_size: int = 3) -> torch.Tensor:
    """Score each cached key by the decoding block's mean query (Sparse-dLLM order)."""
    if q_block.size(1) != candidate_k.size(1):
        if q_block.size(1) % candidate_k.size(1):
            raise ValueError("query heads must be divisible by KV heads")
        candidate_k = candidate_k.repeat_interleave(
            q_block.size(1) // candidate_k.size(1), dim=1)
    average_query = q_block.mean(dim=-2)
    scores = torch.matmul(average_query.unsqueeze(-2),
                          candidate_k.transpose(-2, -1)).squeeze(-2)
    importance = scores.mean(dim=1)
    return F.max_pool1d(importance.unsqueeze(1), kernel_size=pool_kernel_size,
                        stride=1, padding=pool_kernel_size // 2).squeeze(1)


class CustomCache:
    """Per-block K/V cache that keeps the top `keep_ratio` entries chosen by the student.

    At cache_state 1 every layer stores its K/V and calls `filter_cache`, which drops
    the current block and evicts among the remaining candidates. At cache_state 2
    the layer attends to what is left. `capture_rows` records the block's attention
    over the candidates, which is the teacher label.
    """

    def __init__(self, n_layers: int, device: torch.device, keep_ratio: float = 1.0,
                 cache_scorer=None, *, order_full_cache: bool = False,
                 capture_hidden_states: bool = False) -> None:
        self.n_layers = n_layers
        self.device = device
        self.keep_ratio = float(keep_ratio)
        self.cache_scorer = cache_scorer
        self.order_full_cache = order_full_cache
        self.capture_hidden_states = capture_hidden_states or cache_scorer is not None
        self.cache = {}
        self.layer_hidden_states = {}
        self.candidate_order = {}

        self.collect_pool = False
        self.capture_rows = False
        self.capture_per_head = True
        self.group_reduce = "mean"
        self.pending_rows = {}

    def record_attention(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if not self.capture_rows:
            return
        kv_heads = k.size(1)
        if q.size(1) % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        group = q.size(1) // kv_heads
        if group != 1:
            k = k.repeat_interleave(group, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        rows = weights[..., :k.size(-2) - q.size(-2)]
        if self.capture_per_head:
            if group != 1:
                batch, _, n_rows, n_cols = rows.shape
                grouped = rows.view(batch, kv_heads, group, n_rows, n_cols)
                rows = (grouped.amax(dim=2) if self.group_reduce == "max"
                        else grouped.mean(dim=2))
            rows = rows.squeeze(0)
        else:
            rows = rows.mean(dim=1).squeeze(0)
        if layer_id in self.candidate_order:
            natural = torch.empty_like(rows)
            natural[..., self.candidate_order[layer_id]] = rows
            rows = natural
        self.pending_rows[layer_id] = rows

    def capture_layer_hidden_states(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        if self.capture_hidden_states:
            self.layer_hidden_states[layer_id] = hidden_states

    def get_cache(self, layer_id: int):
        return self.cache.get(layer_id, {"k": None, "v": None})

    def update_cache(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.cache[layer_id] = {"k": k.clone(), "v": v.clone()}

    def _keep(self, layer_id, keep_k, keep_v, indices):
        head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
        self.cache[layer_id] = {"k": keep_k[:, head_index, indices],
                                "v": keep_v[:, head_index, indices]}

    def filter_cache(self, layer_id: int, q_block: torch.Tensor,
                     cur_filtered_len: int, block_len: int):
        cached = self.get_cache(layer_id)
        cached_k, cached_v = cached["k"], cached["v"]
        keep_k = torch.cat([cached_k[:, :, :cur_filtered_len, :],
                            cached_k[:, :, cur_filtered_len + block_len:, :]], dim=2)
        keep_v = torch.cat([cached_v[:, :, :cur_filtered_len, :],
                            cached_v[:, :, cur_filtered_len + block_len:, :]], dim=2)

        if self.collect_pool or self.keep_ratio >= 1.0:
            if not self.order_full_cache:
                self.cache[layer_id] = {"k": keep_k, "v": keep_v}
                return
            scores = block_attention_score(q_block, keep_k)
            indices = torch.topk(scores, k=keep_k.size(-2), dim=-1).indices.squeeze(0)
            if self.collect_pool:
                self.candidate_order[layer_id] = indices
            self._keep(layer_id, keep_k, keep_v, indices)
            return

        if self.cache_scorer is None:
            raise RuntimeError("cache eviction needs a trained student; pass a "
                               "checkpoint, or run with keep_ratio=1.0")
        hidden_states = self.layer_hidden_states.pop(layer_id, None)
        if hidden_states is None:
            raise RuntimeError(f"missing hidden states for scorer layer {layer_id}")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("student eviction requires batch_size=1")
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
            block_indices=block_indices).float()
        if scores.dim() == 3 and scores.shape[1] != keep_k.size(1):
            raise RuntimeError(f"scorer emits {scores.shape[1]} head scores but the "
                               f"cache has {keep_k.size(1)} heads")
        keep_num = int(candidate_indices.numel() * self.keep_ratio)
        indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0)
        self._keep(layer_id, keep_k, keep_v, indices)

    def clear(self):
        self.cache.clear()
