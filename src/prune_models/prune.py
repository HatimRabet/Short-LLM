OUTPUT_PATH = "./pruned_models"  
LORA_MERGE_CACHE = "/tmp"  
CONFIG_YML = "./prune.yaml"  
COPY_TOKENIZER = True  
LAZY_UNPICKLE = False  
LOW_CPU_MEMORY = False  

# actually do merge
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)
