from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets
    from opencompass.configs.datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from .arc_c_gen_test import ARC_c_datasets
    from .gpqa_gen_5shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets, *mmlu_datasets]
summarizer = dict(summary_groups=mmlu_summary_groups)
