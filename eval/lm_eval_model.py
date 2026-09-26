from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "model" / "LLaDA-8B-Instruct"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _generation_kwargs(raw: dict, default_max_gen_toks: int, dream_decoding=None) -> dict:
    out = dict(raw)
    gen_length = int(out.get("gen_length", out.get("max_gen_toks", default_max_gen_toks)))
    out["gen_length"] = gen_length
    if dream_decoding is not None:
        out.update(dream_decoding.generation_kwargs(gen_length))
        return out
    out.setdefault("steps", gen_length)
    if not out.get("do_sample", False):
        out["temperature"] = 0.0
    return out


@register_model("LLaDA_future", "Dream_future")
class FutureDLLM(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        keep_ratio: float = 1.0,
        block_len: int = 32,
        max_seq_len: int = None,
        student_path: str = "",
        dtype: str = "bfloat16",
        dream_alg: str = "entropy",
        dream_temperature: float = 0.2,
        dream_top_p: float = 0.95,
        dream_steps: int = 512,
        dream_seed: int = 0,
        show_speed: bool = True,
        **kwargs,
    ):
        from future_dllm import detect_family, load_model, load_prompt_utility_student
        family = detect_family(pretrained)
        self._block_len = int(block_len)
        self._max_seq_len = int(max_seq_len) if max_seq_len is not None else (
            2048 if family == "dream" else 4096)
        self._max_prompt_len = self._max_seq_len
        self._keep_ratio = float(keep_ratio)
        self._show_speed = bool(show_speed)
        self._dream_seed = int(dream_seed)
        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self._block_len < 1 or self._max_seq_len < 1:
            raise ValueError("block_len and max_seq_len must be positive")
        if self._keep_ratio < 1.0 and not student_path:
            raise ValueError("cache eviction requires student_path=<checkpoint>")
        if str(dtype) not in ("bfloat16", ""):
            raise ValueError("the model backends load in bfloat16")

        model, self._backend = load_model(
            pretrained, max_seq_len=self._max_seq_len,
            block_length=self._block_len, keep_ratio=self._keep_ratio)
        self._generate = self._backend.generate
        self._fallback_mask_id = self._backend.mask_id
        self._dream_decoding = None
        if family == "dream":
            from future_dllm.dream_decoding import DreamDecoding
            self._dream_decoding = DreamDecoding(
                alg=dream_alg, temperature=float(dream_temperature),
                top_p=float(dream_top_p), steps=int(dream_steps))

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("device", "cuda:0")
        super().__init__(pretrained=model, **kwargs)
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"expected a CUDA model, got {device}")
        self._scorer = None
        if student_path and self._keep_ratio < 1.0:
            if self._dream_decoding is not None:
                from future_dllm.dream_decoding import require_matching_decoding
                path = Path(student_path) / "decoding.json"
                saved = json.loads(path.read_text()) if path.is_file() else None
                require_matching_decoding(saved, self._dream_decoding.metadata(), path)
            self._scorer = load_prompt_utility_student(student_path, device)
        tokenizer_mask = getattr(self.tokenizer, "mask_token_id", None)
        if tokenizer_mask is not None and int(tokenizer_mask) != self._backend.mask_id:
            raise RuntimeError("tokenizer and model mask token IDs disagree")
        self._resume_identity = json.dumps({
            "model": str(pretrained), "student": student_path,
            "keep_ratio": self._keep_ratio, "block_len": self._block_len,
            "max_seq_len": self._max_seq_len,
            "decoding": self._dream_decoding.metadata() if self._dream_decoding else None,
            "dream_seed": self._dream_seed,
        }, sort_keys=True)
        print(f"[{family}] total={self._max_seq_len} block={self._block_len} "
              f"keep_ratio={self._keep_ratio}", flush=True)

    @property
    def _mask_id(self) -> int:
        token_id = getattr(self.tokenizer, "mask_token_id", None)
        return self._fallback_mask_id if token_id is None else int(token_id)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Use OpenCompass for ARC-Challenge, PIQA, and GPQA: scripts/run_oc_mc.sh")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("This evaluator supports generation only")

    def _call_generate(self, context_enc, gen_kwargs, gen_length):
        if self._dream_decoding is not None:
            from future_dllm.dream_decoding import sample_seed
            seed = sample_seed(self._dream_seed, repr(context_enc.tolist()))
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return self._generate(
                    self.model, context_enc.to(self.device), gen_length=gen_length,
                    block_length=self._block_len, mask_id=self._mask_id,
                    cache_scorer=self._scorer, eviction_method="student",
                    **self._dream_decoding.generation_kwargs(gen_length))
        return self._generate(
            self.model, context_enc.to(self.device),
            steps=int(gen_kwargs["steps"]), gen_length=gen_length,
            block_length=self._block_len,
            temperature=float(gen_kwargs.get("temperature", 0.0)),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
            remasking=gen_kwargs.get("remasking") or "low_confidence",
            cache_scorer=self._scorer, eviction_method="student")

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        store_path = os.environ.get("FUTURE_DLLM_RESUME", "")
        done, store = {}, None
        if store_path:
            if os.path.exists(store_path):
                with open(store_path) as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        done[rec["key"]] = rec["text"]
            os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
            store = open(store_path, "a")
            print(f"[{self._backend.name}_future] resume store: "
                  f"{len(done)} answers on disk", flush=True)

        results = []
        measured_seconds = 0.0
        measured_tokens = 0
        measured_answers = 0
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="future_dllm generate_until")
        for request in requests:
            context, raw_kwargs = request.args
            key = hashlib.md5(
                (context + repr(sorted(raw_kwargs.items())) + self._resume_identity).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                bar.update(1)
                continue
            gen_kwargs = _generation_kwargs(raw_kwargs, self.max_gen_toks, self._dream_decoding)
            gen_length = int(gen_kwargs["gen_length"])
            if gen_length % self._block_len:
                gen_length += self._block_len - gen_length % self._block_len

            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
            prompt_limit = min(self._max_prompt_len, self._max_seq_len - gen_length)
            if prompt_limit < 1:
                raise ValueError(
                    f"generation length {gen_length} leaves no prompt space "
                    f"within max_seq_len {self._max_seq_len}"
                )
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=prompt_limit)

            started = time.perf_counter()
            out = self._call_generate(context_enc, gen_kwargs, gen_length)
            elapsed = time.perf_counter() - started
            text = self.tokenizer.decode(out[0, context_enc.shape[1]:],
                                         skip_special_tokens=True)
            for term in gen_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            results.append(text)
            measured_seconds += elapsed
            measured_tokens += len(self.tokenizer.encode(text, add_special_tokens=False))
            measured_answers += 1
            if store is not None:
                store.write(json.dumps({"key": key, "text": text}) + "\n")
                store.flush()
                os.fsync(store.fileno())
            bar.update(1)
        bar.close()
        if store is not None:
            store.close()
        if self._show_speed and measured_answers:
            print(
                f"[{self._backend.name}_future] generated {measured_answers} answers, "
                f"{measured_tokens} decoded tokens in {measured_seconds:.1f}s "
                f"({measured_tokens / measured_seconds:.2f} tok/s)",
                flush=True,
            )
        return results
