"""Block-wise cache selection; model-specific decoding is supplied by an adapter."""
from typing import Callable, Protocol

import torch

BLOCK_LENGTH = 32
TOTAL_LENGTH = {"llada": 4096, "dream": 2048}


class Decoder(Protocol):
    """Contract for a frozen LLaDA or Dream model in evaluation mode.

    Forward runs on the full sequence at steps 0 and 1, and on the current
    block thereafter. At step 1, call select(layer, hidden, keys, values) in
    each layer and store its returned external K/V and absolute positions.
    The current block's K/V are computed separately on every later step.

    reveal updates tokens in place using the native schedule: LLaDA uses
    greedy low-confidence remasking; Dream uses shifted logits, first-token
    seeding, entropy selection, temperature 0.2, and top-p 0.95.
    """

    mask_id: int

    def steps_for_length(self, gen_length: int) -> int:
        """LLaDA: gen_length. Dream: min(256, gen_length)."""
        ...

    def new_cache(self): ...

    def forward(self, tokens, block_start, step, cache, select: Callable): ...

    def reveal(self, tokens, output, block_start, step, steps_per_block): ...

    def completed_qk(self, tokens, block_start, cache):
        """One extra forward on the finished block, without sampling.

        Return one (Q_block, K_all) pair per layer: [query_heads, 32, dim]
        and [kv_heads, sequence_length, dim]. K_all is in absolute position
        order and combines the unchanged full external cache with fresh
        completed-block keys. Q and K already include rotary embeddings.
        """
        ...


def prepare_sequence(prompt, family, gen_length, mask_id):
    """Reserve the generation budget and left-truncate one tokenized prompt."""
    if prompt.ndim != 1 or prompt.numel() == 0:
        raise ValueError("prompt must be a nonempty one-dimensional token tensor")
    if gen_length <= 0 or gen_length % BLOCK_LENGTH:
        raise ValueError("gen_length must be a positive multiple of 32")
    prompt_limit = TOTAL_LENGTH[family] - gen_length
    if prompt_limit < 1:
        raise ValueError("generation leaves no room for a prompt")
    prompt = prompt[-prompt_limit:]
    masked = prompt.new_full((gen_length,), mask_id)
    return torch.cat((prompt, masked)), prompt.numel()


def candidate_indices(length, block_start, device):
    """Prompt, completed blocks, and future blocks compete in the same pool."""
    return torch.cat((torch.arange(block_start, device=device),
                      torch.arange(block_start + BLOCK_LENGTH, length, device=device)))


def select_cache(student, layer, hidden, keys, values, block_start, keep_ratio):
    """Select external K/V; hidden is [sequence, width], K/V are [heads, sequence, dim]."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    candidates = candidate_indices(hidden.shape[0], block_start, hidden.device)
    heads = keys.shape[0]
    if keep_ratio == 1:
        positions = candidates.expand(heads, -1)
    else:
        if student is None:
            raise ValueError("eviction requires a trained student")
        block = torch.arange(block_start, block_start + BLOCK_LENGTH, device=hidden.device)
        scores = student.score(layer, hidden.float(), candidates, block)
        if scores.shape not in ((1, candidates.numel()), (heads, candidates.numel())):
            raise ValueError("student must emit one score row or one row per KV head")
        k = int(candidates.numel() * keep_ratio)
        chosen = scores.topk(k, dim=-1).indices
        positions = candidates[chosen].expand(heads, -1)
    index = positions.unsqueeze(-1).expand(-1, -1, keys.shape[-1])
    return keys.gather(1, index), values.gather(1, index), positions


@torch.no_grad()
def generate(decoder: Decoder, prompt, family, gen_length, student=None,
             keep_ratio=1.0, on_block_complete=None):
    """Return prompt + answer; callbacks observe teacher states without modifying them."""
    if not 0 < keep_ratio <= 1 or (keep_ratio < 1 and student is None):
        raise ValueError("use keep_ratio in (0, 1] and a student when evicting")
    if on_block_complete is not None and keep_ratio != 1:
        raise ValueError("teacher collection requires the full cache")
    if student is not None:
        student.eval()
    tokens, prompt_length = prepare_sequence(prompt, family, gen_length, decoder.mask_id)
    blocks = gen_length // BLOCK_LENGTH
    steps = decoder.steps_for_length(gen_length)
    if steps % blocks or steps // blocks < 2:
        raise ValueError("each block requires at least two decoding steps")
    steps_per_block = steps // blocks

    for start in range(prompt_length, tokens.numel(), BLOCK_LENGTH):
        cache = decoder.new_cache()
        selection_input = None

        def select(layer, hidden, keys, values):
            return select_cache(student, layer, hidden, keys, values, start, keep_ratio)

        for step in range(steps_per_block):
            # Deployment scores the state after step 0, before step 1 reveals tokens.
            if step == 1 and on_block_complete is not None:
                selection_input = tokens.clone()
            output = decoder.forward(tokens, start, step, cache, select)
            decoder.reveal(tokens, output, start, step, steps_per_block)
        if on_block_complete is not None:
            on_block_complete(tokens, selection_input, start, cache)
    return tokens
