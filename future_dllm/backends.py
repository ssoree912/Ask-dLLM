from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class Backend:

    name: str
    mask_id: int
    n_layers: int
    hidden_dim: int
    kv_heads: int
    native_max_seq_len: int
    generate: Callable

    logit_shift: bool = False
    seed_block_start: bool = False


def detect_family(model_path: str | Path) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"no config.json under {model_path}")
    model_type = str(json.loads(config_path.read_text()).get("model_type", "")).lower()
    if "dream" in model_type:
        return "dream"
    if "llada" in model_type:
        return "llada"
    raise SystemExit(
        f"unsupported model_type {model_type!r} in {config_path}; "
        "future_dllm supports LLaDA and Dream checkpoints"
    )



def _llada_kv_heads(cfg) -> int:
    effective = getattr(cfg, "effective_n_kv_heads", None)
    if effective is not None:
        return int(effective)
    n_kv_heads = getattr(cfg, "n_kv_heads", None)
    multi_query = getattr(cfg, "multi_query_attention", None)
    if n_kv_heads is None:
        return 1 if multi_query is True else int(cfg.n_heads)
    if multi_query is None:
        return int(n_kv_heads)
    expected = 1 if multi_query else int(cfg.n_heads)
    if int(n_kv_heads) != expected:
        raise ValueError(
            "LLaDA config sets both multi_query_attention and a conflicting "
            f"n_kv_heads ({n_kv_heads} against {expected})")
    return expected

def load_model(model_path: str | Path, *, max_seq_len: int, block_length: int,
               keep_ratio: float = 1.0, device_map: str = "cuda:0") -> tuple[torch.nn.Module, Backend]:
    family = detect_family(model_path)

    if family == "dream":
        from .configuration_dream import DreamConfig
        from .modeling_dream import DreamModel
        from .dream_generate import generate as dream_generate

        cfg = DreamConfig.from_pretrained(model_path)
        native = int(getattr(cfg, "max_position_embeddings", max_seq_len))
        cfg.block_len, cfg.keep_ratio = block_length, keep_ratio
        cfg.use_cache = False
        model = DreamModel.from_pretrained(
            model_path, config=cfg, device_map=device_map,
            torch_dtype=torch.bfloat16).eval()
        backend = Backend(
            name="dream",
            mask_id=int(cfg.mask_token_id),
            n_layers=int(cfg.num_hidden_layers),
            hidden_dim=int(cfg.hidden_size),
            kv_heads=int(cfg.num_key_value_heads),
            native_max_seq_len=native,
            generate=dream_generate,
            logit_shift=True,
            seed_block_start=True,
        )
        return model, backend

    from transformers import AutoConfig

    from .llada_generate import generate as llada_generate
    from .modeling_llada import LLaDAModelLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    native = int(getattr(cfg, "max_sequence_length", max_seq_len))
    cfg.max_sequence_length = max_seq_len
    cfg.block_len, cfg.keep_ratio = block_length, keep_ratio
    model = LLaDAModelLM.from_pretrained(
        model_path, config=cfg, device_map=device_map,
        torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    backend = Backend(
        name="llada",
        mask_id=126336,
        n_layers=int(cfg.n_layers),
        hidden_dim=int(cfg.d_model),
        kv_heads=_llada_kv_heads(cfg),
        native_max_seq_len=native,
        generate=llada_generate,
        logit_shift=False,
        seed_block_start=False,
    )
    return model, backend
