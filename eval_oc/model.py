"""OpenCompass generation for Dream with a trained cache scorer.

Prompts are flattened into one user turn and formatted with Dream's chat
template. The total sequence budget includes both the prompt and generation.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel


def _convert_base_messages(inputs):
    """Flatten OpenCompass prompts to plain strings.

    Verbatim from Sparse-dLLM's ``dream_wrapper_instruct.py``: a PromptList is
    joined on ``prompt`` with the roles dropped, and the whole thing is handed
    to the chat template as a single user turn.
    """
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            outputs.append(''.join(item['prompt'] for item in _input))
    return outputs


class DreamFutureOC(BaseModel):
    """Dream with future-attention cache eviction, driven by OpenCompass."""

    def __init__(
        self,
        path: str = "",
        max_seq_len: int = 2048,
        block_length: int = 32,
        keep_ratio: Optional[float] = None,
        eviction_method: str = "student",
        student_path: str = "",
        dream_alg: str = "entropy",
        dream_temperature: float = 0.2,
        dream_top_p: float = 0.95,
        dream_steps: int = 256,
        dream_seed: int = 0,
        meta_template: Optional[dict] = None,
    ):
        # The config that builds this is parsed by mmengine in lazy-import
        # mode, which cannot call os.environ.get, so the paths arrive empty and
        # are resolved here instead -- same variables scripts/run_eval.sh uses.
        path = path or os.environ.get("FUTURE_DLLM_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_MODEL or pass path= in the config")
        if eviction_method == "student":
            student_path = student_path or os.environ.get("FUTURE_DLLM_STUDENT", "")

        # No **kwargs: OpenCompass strips only run_cfg / max_out_len /
        # batch_size / abbr / summarizer_abbr / pred_postprocessor /
        # min_out_len before building, so anything else arriving here is a real
        # config key and a typo should raise instead of being swallowed.
        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from transformers import AutoTokenizer

        from future_dllm import load_model, load_prompt_utility_student
        from future_dllm.dream_decoding import DreamDecoding, sample_seed

        self._block_length = int(block_length)
        self._keep_ratio = float(
            os.environ.get("FUTURE_DLLM_KEEP_RATIO", "1.0")
            if keep_ratio is None else keep_ratio)
        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self._eviction_method = str(eviction_method)
        self._seed = int(dream_seed)
        self._sample_seed = sample_seed
        self._decoding = DreamDecoding(alg=dream_alg, temperature=dream_temperature,
                                       top_p=dream_top_p, steps=dream_steps)

        self.model, self._backend = load_model(
            path, max_seq_len=max_seq_len, block_length=self._block_length,
            keep_ratio=self._keep_ratio)
        if self._backend.name != "dream":
            raise ValueError("this OpenCompass wrapper requires a Dream checkpoint")
        self.model.eval()
        # load_model injects these onto the config before from_pretrained, and
        # dream_generate reads them back off model.config rather than from its
        # own arguments. Reading them back here turns "the config said 0.5"
        # into "the model is running at 0.5", which is the claim the whole
        # comparison rests on and is otherwise invisible in the logs.
        if (self.model.config.keep_ratio != self._keep_ratio
                or self.model.config.block_len != self._block_length):
            raise RuntimeError(
                f"eviction settings did not reach the model: config says "
                f"keep_ratio={self.model.config.keep_ratio} "
                f"block_len={self.model.config.block_len}, expected "
                f"{self._keep_ratio} / {self._block_length}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default

        self._scorer = None
        if self._eviction_method == "student" and self._keep_ratio < 1.0:
            if not student_path:
                raise ValueError("eviction_method='student' requires a scorer: "
                                 "set FUTURE_DLLM_STUDENT or pass student_path=")
            if self._keep_ratio < 1:
                # The lm-eval path already refuses a scorer built under a
                # different decoding; this one did not, so the same checkpoint
                # could be run here at a temperature its labels never saw and
                # the row would look like every other row. Same check, same
                # FUTURE_DLLM_ALLOW_SAMPLING_DRIFT escape hatch.
                from future_dllm.dream_decoding import require_matching_decoding
                saved_path = Path(student_path) / "decoding.json"
                saved = (json.loads(saved_path.read_text())
                         if saved_path.is_file() else None)
                require_matching_decoding(saved, self._decoding.metadata(),
                                          saved_path)
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
        print(f"[DreamFutureOC] keep_ratio={self._keep_ratio} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"eviction={eviction} seed={self._seed} "
              f"decoding={self._decoding.metadata()}", flush=True)

    def _encode(self, text: str, max_length: int):
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        return self.tokenizer(chat, return_tensors="pt", truncation=True,
                              add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        # Measured on the un-templated string, as theirs is: OpenCompass uses
        # this only to sort and shard prompts, so it has to mean the same thing
        # on both rows, not be maximally accurate.
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        # GenInferencer inspects this signature and only passes
        # stopping_criteria if it is declared. Dream's block schedule cannot
        # stop early, and their wrapper drops stop words for Dream too, so
        # leaving it undeclared matches the baseline instead of silently
        # accepting a knob that does nothing.
        from future_dllm.dream_generate import generate

        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            # generation_utils.py:389 asserts rather than rounding. Rounding
            # here would quietly give one row a longer budget than the other.
            raise ValueError(
                f"max_out_len ({gen_length}) must be divisible by block_length "
                f"({self._block_length})")

        prompt_limit = self.max_seq_len - gen_length
        if prompt_limit < 1:
            raise ValueError("generation budget leaves no room for a prompt")
        eos = self.tokenizer.eos_token
        outputs = []
        for text in _convert_base_messages(inputs):
            item_seed = self._sample_seed(self._seed, text)
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
            torch.cuda.manual_seed_all(item_seed)

            tokens = self._encode(text, prompt_limit)
            ids = tokens["input_ids"].to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length,
                           eviction_method=self._eviction_method,
                           cache_scorer=self._scorer,
                           **self._decoding.generation_kwargs(gen_length))
            # Their decode keeps special tokens and cuts at the first EOS, so
            # anything the block schedule emits after EOS is dropped.
            text_out = self.tokenizer.decode(out[0, ids.shape[1]:].tolist())
            outputs.append(text_out.split(eos)[0] if eos else text_out)
        return outputs
