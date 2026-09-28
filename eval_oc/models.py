from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ask_dllm import load_model, load_student  # noqa: E402
from ask_dllm.dream_decoding import (  # noqa: E402
    DreamDecoding, require_matching_decoding, sample_seed)


def _flatten(inputs):
    return [text if isinstance(text, str) else "".join(item["prompt"] for item in text)
            for text in inputs]


class _AskDLLMBase(BaseModel):
    family = ""

    def __init__(self, path: str, max_seq_len: int, block_length: int = 32,
                 keep_ratio: float = 1.0, student_path: str = "", seed: int = 0,
                 meta_template: Optional[dict] = None):
        if not path:
            raise ValueError("path= must point at the model checkpoint")
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template)
        from transformers import AutoTokenizer

        self._block_length = int(block_length)
        self._keep_ratio = float(keep_ratio)
        self._seed = int(seed)
        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.model, self._backend = load_model(
            path, max_seq_len=max_seq_len, block_length=self._block_length,
            keep_ratio=self._keep_ratio)
        if self._backend.name != self.family:
            raise ValueError(f"{type(self).__name__} requires a {self.family} checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"

        self._scorer = None
        if self._keep_ratio < 1.0:
            if not student_path:
                raise ValueError("keep_ratio < 1 requires student_path=<checkpoint>")
            self._check_student(Path(student_path))
            self._scorer = load_student(student_path, next(self.model.parameters()).device)
        print(f"[{type(self).__name__}] keep_ratio={self._keep_ratio} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"student={student_path or '-'} seed={self._seed}", flush=True)

    def _check_student(self, student_path: Path) -> None:
        pass

    def _seed_item(self, text: str) -> None:
        item_seed = sample_seed(self._seed, text)
        random.seed(item_seed)
        np.random.seed(item_seed % (2**32))
        torch.manual_seed(item_seed)
        torch.cuda.manual_seed_all(item_seed)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _flatten([prompt])[0]
        return len(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


class LLaDAOC(_AskDLLMBase):
    family = "llada"

    def __init__(self, steps: int = 256, **kwargs):
        self._steps = int(steps)
        super().__init__(**kwargs)

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 stopping_criteria: List[str] = []) -> List[str]:
        from ask_dllm.llada_generate import generate

        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            gen_length = (gen_length // self._block_length + 1) * self._block_length
        steps = min(self._steps, gen_length)
        prompt_limit = self.max_seq_len - gen_length
        if prompt_limit < 1:
            raise ValueError("generation budget leaves no room for a prompt")
        outputs = []
        for text in _flatten(inputs):
            self._seed_item(text)
            ids = self.tokenizer.batch_encode_plus(
                [text], return_tensors="pt", padding=True, truncation=True,
                add_special_tokens=True, max_length=prompt_limit)["input_ids"]
            ids = ids.to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length, steps=steps,
                           temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                           mask_id=self._backend.mask_id, cache_scorer=self._scorer)
            answer = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(),
                                           skip_special_tokens=True)
            for stop in stopping_criteria:
                answer = answer.split(stop)[0]
            outputs.append(answer)
        return outputs


class DreamOC(_AskDLLMBase):
    family = "dream"

    def __init__(self, alg: str = "entropy", temperature: float = 0.2,
                 top_p: float = 0.95, steps: int = 512, **kwargs):
        self._decoding = DreamDecoding(alg=alg, temperature=float(temperature),
                                       top_p=float(top_p), steps=int(steps))
        super().__init__(**kwargs)

    def _check_student(self, student_path: Path) -> None:
        saved_path = student_path / "decoding.json"
        saved = json.loads(saved_path.read_text()) if saved_path.is_file() else None
        require_matching_decoding(saved, self._decoding.metadata(), saved_path)

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        from ask_dllm.dream_generate import generate

        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            raise ValueError(f"max_out_len ({gen_length}) must be divisible by "
                             f"block_length ({self._block_length})")
        prompt_limit = self.max_seq_len - gen_length
        if prompt_limit < 1:
            raise ValueError("generation budget leaves no room for a prompt")
        eos = self.tokenizer.eos_token
        outputs = []
        for text in _flatten(inputs):
            self._seed_item(text)
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
            ids = self.tokenizer(chat, return_tensors="pt", truncation=True,
                                 add_special_tokens=True,
                                 max_length=prompt_limit)["input_ids"]
            ids = ids.to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length, cache_scorer=self._scorer,
                           **self._decoding.generation_kwargs(gen_length))
            answer = self.tokenizer.decode(out[0, ids.shape[1]:].tolist())
            outputs.append(answer.split(eos)[0] if eos else answer)
        return outputs
