from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel

from .model import _convert_base_messages


class LLaDAFutureOC(BaseModel):

    def __init__(
        self,
        path: str = "",
        max_seq_len: int = 4096,
        block_length: int = 32,
        keep_ratio: Optional[float] = None,
        eviction_method: str = "student",
        student_path: str = "",
        llada_steps: int = 256,
        llada_temperature: float = 0.0,
        llada_cfg_scale: float = 0.0,
        llada_remasking: str = "low_confidence",
        llada_seed: int = 0,
        meta_template: Optional[dict] = None,
    ):
        path = path or os.environ.get("FUTURE_DLLM_LLADA_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_LLADA_MODEL or pass path= in the config")
        if eviction_method == "student":
            student_path = student_path or os.environ.get("FUTURE_DLLM_LLADA_STUDENT", "")

        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from transformers import AutoTokenizer

        from future_dllm import load_model, load_prompt_utility_student
        from future_dllm.dream_decoding import sample_seed

        self._block_length = int(block_length)
        self._keep_ratio = float(
            os.environ.get("FUTURE_DLLM_KEEP_RATIO", "1.0")
            if keep_ratio is None else keep_ratio)
        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self._eviction_method = str(eviction_method)
        self._seed = int(llada_seed)
        self._sample_seed = sample_seed
        self._decoding = dict(steps=int(llada_steps),
                              temperature=float(llada_temperature),
                              cfg_scale=float(llada_cfg_scale),
                              remasking=str(llada_remasking))

        self.model, self._backend = load_model(
            path, max_seq_len=max_seq_len, block_length=self._block_length,
            keep_ratio=self._keep_ratio)
        if self._backend.name != "llada":
            raise ValueError("this OpenCompass wrapper requires a LLaDA checkpoint")
        self.model.eval()
        if (self.model.config.keep_ratio != self._keep_ratio
                or self.model.config.block_len != self._block_length):
            raise RuntimeError(
                f"eviction settings did not reach the model: config says "
                f"keep_ratio={self.model.config.keep_ratio} "
                f"block_len={self.model.config.block_len}, expected "
                f"{self._keep_ratio} / {self._block_length}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"

        self._scorer = None
        if self._eviction_method == "student" and self._keep_ratio < 1.0:
            if not student_path:
                raise ValueError("eviction_method='student' requires a scorer: "
                                 "set FUTURE_DLLM_LLADA_STUDENT or pass student_path=")
            self._scorer = load_prompt_utility_student(
                student_path, next(self.model.parameters()).device)
        elif self._eviction_method not in ("student", "sparse"):
            raise ValueError(f"unknown eviction_method {self._eviction_method!r}; "
                             "expected 'sparse' or 'student'")

        if self._keep_ratio >= 1.0:
            eviction = "none (keep_ratio=1.0)"
        elif self._eviction_method == "sparse":
            eviction = "sparse (baseline attention score, no checkpoint)"
        else:
            eviction = f"student ({student_path})"
        print(f"[LLaDAFutureOC] keep_ratio={self._keep_ratio} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"eviction={eviction} seed={self._seed} "
              f"decoding={self._decoding}", flush=True)

    def _encode(self, text: str, max_length: int):
        return self.tokenizer.batch_encode_plus(
            [text], return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 stopping_criteria: List[str] = []) -> List[str]:
        from future_dllm.llada_generate import generate

        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            gen_length = (gen_length // self._block_length + 1) * self._block_length
        steps = min(self._decoding["steps"], gen_length)
        prompt_limit = self.max_seq_len - gen_length
        if prompt_limit < 1:
            raise ValueError("generation budget leaves no room for a prompt")

        outputs = []
        for text in _convert_base_messages(inputs):
            item_seed = self._sample_seed(self._seed, text)
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
            torch.cuda.manual_seed_all(item_seed)

            ids = self._encode(text, prompt_limit)["input_ids"].to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length, steps=steps,
                           temperature=self._decoding["temperature"],
                           cfg_scale=self._decoding["cfg_scale"],
                           remasking=self._decoding["remasking"],
                           mask_id=self._backend.mask_id,
                           eviction_method=self._eviction_method,
                           cache_scorer=self._scorer)
            text_out = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(),
                                             skip_special_tokens=True)
            for stop in stopping_criteria:
                text_out = text_out.split(stop)[0]
            outputs.append(text_out)
        return outputs
