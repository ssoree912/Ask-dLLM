import numpy as np
import torch
import torch.nn.functional as F

from .modeling_llada import CustomCache

MASK_ID = 126336


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    counts = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                         dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        counts[i, :remainder[i]] += 1
    return counts


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32,
             temperature=0., cfg_scale=0., remasking='low_confidence',
             mask_id=MASK_ID, cache_scorer=None, *, eviction_method="student",
             eviction_accum="none", eviction_accum_decay=1.0, oracle_reduce=None,
             current_reduce=None):
    if eviction_method not in ("student", "current", "sparse", "oracle"):
        raise ValueError("eviction_method must be student, current, sparse or oracle")
    if (oracle_reduce is not None) != (eviction_method == "oracle"):
        raise ValueError("oracle_reduce and eviction_method='oracle' go together")
    if (current_reduce is not None) != (eviction_method == "current"):
        raise ValueError("current_reduce and eviction_method='current' go together")
    if eviction_accum not in ("none", "across_blocks"):
        raise ValueError("eviction_accum must be none or across_blocks")
    accum_state = {} if eviction_accum == "across_blocks" else None
    if model.config.keep_ratio < 1 and eviction_method == "student" and cache_scorer is None:
        raise ValueError("student eviction requires a scorer")
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        num_transfer = get_num_transfer_tokens(
            x[:, block_start:block_end] == mask_id, steps_per_block)

        def step_block(cache):
            for i in range(steps_per_block):
                cache_state = 2 if i > 1 else i
                model_input = x if cache_state != 2 else x[:, block_start:block_end]
                mask_index = (model_input == mask_id)

                logits = model(model_input, block_start, cache_state, cache).logits
                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, -1, torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                target = x if cache_state != 2 else x[:, block_start:block_end]
                if cache_state != 2:
                    x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, target)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
                for j in range(confidence.shape[0]):
                    reveal = torch.topk(confidence[j], k=num_transfer[j, i]).indices
                    target[j, reveal] = x0[j, reveal]

        if oracle_reduce is None:
            cache = CustomCache(
                n_layers=model.config.n_layers, device=model.device,
                keep_ratio=model.config.keep_ratio,
                cache_scorer=cache_scorer, prompt_length=prompt_len,
                generation_length=gen_length,
                eviction_method=eviction_method, current_reduce=current_reduce,
                baseline_order=True,
                accum_state=accum_state, accum_decay=eviction_accum_decay)
            step_block(cache)
        else:
            _oracle_block(model, x, block_start, block_end, prompt_len,
                          gen_length, step_block, oracle_reduce)

    return x


def _oracle_block(model, x, bs, be, prompt_len, gen_length, step_block, reduce):
    row_reduce, group_reduce, per_head = reduce
    n_layers = model.config.n_layers
    masked_block = x[:, bs:be].clone()

    full = CustomCache(n_layers=n_layers, device=model.device, keep_ratio=1.0,
                       prompt_length=prompt_len, generation_length=gen_length,
                       eviction_method="sparse", baseline_order=True)
    full.collect_pool = True
    step_block(full)

    full.capture_rows = True
    full.capture_per_head = per_head
    full.group_reduce = group_reduce
    model(x[:, bs:be], bs, 2, full)
    full.capture_rows = False
    axis = 1 if per_head else 0
    label = {layer: (full.pending_rows[layer].amax(axis) if row_reduce == "max"
                     else full.pending_rows[layer].mean(axis))
             for layer in range(n_layers)}
    full.pending_rows.clear()

    x[:, bs:be] = masked_block
    cache = CustomCache(n_layers=n_layers, device=model.device,
                        keep_ratio=model.config.keep_ratio,
                        prompt_length=prompt_len, generation_length=gen_length,
                        eviction_method="oracle", baseline_order=True)
    cache.oracle_label = label
    step_block(cache)
    return cache
