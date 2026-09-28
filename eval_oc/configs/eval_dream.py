from mmengine.config import read_base

from eval_oc.models import DreamOC

with read_base():
    from .datasets import datasets, summarizer
    from .runtime import eval, infer

models = [dict(
    type=DreamOC, abbr="dream", path="", student_path="", keep_ratio=1.0,
    block_length=32, steps=512, alg="entropy", temperature=0.2, top_p=0.95,
    seed=0, max_seq_len=2048, max_out_len=256,
    batch_size=1, run_cfg=dict(num_gpus=1, num_procs=1),
)]
