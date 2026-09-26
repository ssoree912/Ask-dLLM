"""LLaDA: OpenCompass generation at 4096 total tokens and 32-token blocks."""
from mmengine.config import read_base

from eval_oc.llada_model import LLaDAFutureOC

with read_base():
    from .eval_dream_mc import datasets, eval, infer

models = [dict(
    type=LLaDAFutureOC, abbr="llada-ask", path="", student_path="",
    keep_ratio=None, eviction_method="student", block_length=32,
    llada_steps=256, llada_temperature=0.0, llada_cfg_scale=0.0,
    llada_remasking="low_confidence", llada_seed=0,
    max_seq_len=4096, max_out_len=256,
    batch_size=1, run_cfg=dict(num_gpus=1, num_procs=1),
)]
work_dir = "results/oc/llada"
