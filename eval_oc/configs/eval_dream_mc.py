from mmengine.config import read_base

from eval_oc.model import DreamFutureOC
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]
models = [dict(
    type=DreamFutureOC, abbr="dream-ask", path="", student_path="",
    keep_ratio=None, eviction_method="student", block_length=32,
    dream_steps=512, dream_alg="entropy", dream_temperature=0.2,
    dream_top_p=0.95, dream_seed=0, max_seq_len=2048, max_out_len=256,
    batch_size=1, run_cfg=dict(num_gpus=1, num_procs=1),
)]
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=OpenICLEvalTask)),
)
work_dir = "results/oc/dream"
