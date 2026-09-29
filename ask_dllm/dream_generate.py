import torch
import torch.distributions as dists
import torch.nn.functional as F

from .cache import CustomCache
from .dream_decoding import DEFAULT_DREAM_STEPS, DreamDecoding

MASK_ID = 151666


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except ValueError:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def shift_logits(logits):
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate(model, prompt, steps=None, gen_length=128, block_length=32,
             temperature=0.2, cfg_scale=0., remasking=None,
             mask_id=MASK_ID, cache_scorer=None, *, alg="entropy", top_p=0.95,
             top_k=None, alg_temp=None, eps=1e-3, on_block_complete=None):
    if cfg_scale != 0 or remasking is not None:
        raise ValueError("Dream uses alg/temperature/top_p, not LLaDA cfg_scale/remasking")
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Dream cache decoding requires one non-empty, unpadded prompt")
    if block_length < 1 or model.config.block_len != block_length:
        raise ValueError("block_length must be positive and match model.config.block_len")
    if gen_length < 1 or gen_length % block_length:
        raise ValueError("gen_length must be positive and divisible by block_length")
    if model.config.keep_ratio < 1 and cache_scorer is None:
        raise ValueError("keep_ratio < 1 requires a student scorer")

    settings = DreamDecoding(alg=alg, temperature=temperature, top_p=top_p,
                             steps=DEFAULT_DREAM_STEPS if steps is None else steps, eps=eps,
                             top_k=top_k, alg_temp=alg_temp)
    steps = settings.steps_for_length(gen_length)
    num_blocks = gen_length // block_length
    if steps % num_blocks or steps // num_blocks < 2:
        raise ValueError("steps must divide into at least two steps per block")
    steps_per_block = steps // num_blocks
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    prompt_len = prompt.shape[1]
    timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)

    for block_index in range(num_blocks):
        bs = prompt_len + block_index * block_length
        be = bs + block_length
        selection_input = None

        def step_block(cache):
            nonlocal selection_input
            for i in range(steps_per_block):
                cache_state = min(i, 2)
                model_input = x if cache_state != 2 else x[:, bs:be]
                if i == 1 and on_block_complete is not None:
                    selection_input = x.clone()
                logits = model(input_ids=model_input, position_offset=bs,
                               cache_state=cache_state, customcache=cache,
                               attention_mask="full", position_ids=None).logits
                logits = shift_logits(logits)

                if cache_state == 0:
                    _, x0 = sample_tokens(logits[:, bs:be], temperature=temperature,
                                          top_p=top_p, top_k=top_k)
                    x[:, bs] = x0[:, 0]
                    continue

                if cache_state == 1:
                    model_input = model_input[:, bs:be]
                    logits = logits[:, bs:be]
                mask_index = model_input == mask_id
                mask_logits = logits[mask_index]
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
                    margin_confidence=alg == "topk_margin", neg_entropy=alg == "entropy")
                t, s = timesteps[i], timesteps[i + 1]
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1
                    else int(num_mask_token))
                block_confidence = torch.full_like(model_input, -torch.inf,
                                                   device=model.device, dtype=logits.dtype)
                block_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(block_confidence, number_transfer_tokens)
                    else:
                        block_confidence = F.softmax(block_confidence / alg_temp, dim=-1)
                        transfer_index = torch.multinomial(
                            block_confidence, num_samples=number_transfer_tokens)
                    x_block = torch.zeros_like(model_input, device=model.device,
                                               dtype=torch.long) + mask_id
                    x_block[mask_index] = x0.clone()
                    row_indices = torch.arange(model_input.size(0), device=model.device)
                    row_indices = row_indices.unsqueeze(1).expand_as(transfer_index)
                    x[:, bs:be][row_indices, transfer_index] = x_block[row_indices, transfer_index]

        cache = CustomCache(
            n_layers=model.config.num_hidden_layers, device=model.device,
            keep_ratio=model.config.keep_ratio, cache_scorer=cache_scorer,
            order_full_cache=True)
        cache.collect_pool = on_block_complete is not None
        if cache.collect_pool and model.config.keep_ratio != 1.0:
            raise ValueError("teacher collection requires keep_ratio=1.0")
        step_block(cache)

        if on_block_complete is not None:
            on_block_complete(x, cache, block_index, bs, selection_input)
    return x

