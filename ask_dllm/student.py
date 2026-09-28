from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StudentConfig:
    attn_heads: int
    layer_count: int = 32
    hidden_dim: int = 4096
    proj_dim: int = 256
    mlp_dim: int = 512

    def to_json(self) -> dict:
        return {**asdict(self), "heads": ["score"]}


class StudentLayer(nn.Module):
    """Scores every cached candidate against the current block, one score per KV head."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        if config.attn_heads < 1:
            raise ValueError(f"invalid attn_heads: {config.attn_heads}")
        self.attn_heads = int(config.attn_heads)
        self.token_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        self.block_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        self.score_head = nn.Sequential(
            nn.Linear(config.proj_dim * 3, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.attn_heads),
        )

    def forward(self, hidden_states: torch.Tensor, candidate_indices: torch.Tensor,
                block_indices: torch.Tensor) -> torch.Tensor:
        token_proj = self.token_proj(hidden_states.index_select(1, candidate_indices))
        block_state = hidden_states.index_select(1, block_indices).mean(dim=1)
        block_proj = self.block_proj(block_state).unsqueeze(1).expand(
            -1, token_proj.shape[1], -1)
        fused = torch.cat([token_proj, block_proj, token_proj * block_proj], dim=-1)
        scores = self.score_head(fused)
        if self.attn_heads == 1:
            return scores.squeeze(-1)
        return scores.transpose(-2, -1)


class PromptUtilityStudent(nn.Module):

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict(
            {str(i): StudentLayer(config) for i in range(config.layer_count)})

    def forward_layer(self, layer_id: int, hidden_states: torch.Tensor,
                      candidate_indices: torch.Tensor,
                      block_indices: torch.Tensor) -> torch.Tensor:
        return self.layers[str(layer_id)](hidden_states, candidate_indices, block_indices)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.state_dict().items()},
                   directory / "pytorch_model.bin")
        (directory / "config.json").write_text(json.dumps(self.config.to_json()))


def load_student(checkpoint_dir: str | Path, device: torch.device) -> PromptUtilityStudent:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"not a student checkpoint: {checkpoint_dir}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(raw.pop("heads", ("score",))) != ("score",):
        raise ValueError(f"{config_path}: unsupported student heads")
    student = PromptUtilityStudent(StudentConfig(**raw))
    student.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True),
                            strict=True)
    return student.to(device).eval()
