OUTPUT_PATH = "src/prune_models/pruned_models"  
LORA_MERGE_CACHE = "/tmp"  
CONFIG_YML = "src/prune_models/prune.yaml"  
COPY_TOKENIZER = True  
LAZY_UNPICKLE = False  
LOW_CPU_MEMORY = False  

# actually do merge
import torch
import yaml
import os
import stat
import shutil

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


def prune_from_yaml():
    with open(CONFIG_YML, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Le fichier {OUTPUT_PATH} existe déjà. Suppression en cours...")
        # Vérifie si c'est un fichier
        if os.path.isfile(OUTPUT_PATH):
            os.chmod(OUTPUT_PATH, stat.S_IWUSR)  # Donner permission d'écriture
            os.remove(OUTPUT_PATH)
            print("Fichier supprimé avec succès.")
        
        # Vérifie si c'est un dossier
        elif os.path.isdir(OUTPUT_PATH):
            shutil.rmtree(OUTPUT_PATH)  # Supprime récursivement le dossier et son contenu
            print("Dossier supprimé avec succès.")


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
    


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default=CONFIG_YML, help="Please enter the path to the yaml config file")
    args = argparser.parse_args()
    CONFIG_YML = args.config 

    with open(CONFIG_YML, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Le fichier {OUTPUT_PATH} existe déjà. Suppression en cours...")
        # Vérifie si c'est un fichier
        if os.path.isfile(OUTPUT_PATH):
            os.chmod(OUTPUT_PATH, stat.S_IWUSR)  # Donner permission d'écriture
            os.remove(OUTPUT_PATH)
            print("Fichier supprimé avec succès.")
        
        # Vérifie si c'est un dossier
        elif os.path.isdir(OUTPUT_PATH):
            shutil.rmtree(OUTPUT_PATH)  # Supprime récursivement le dossier et son contenu
            print("Dossier supprimé avec succès.")


    run_merge(
        merge_config,
        out_path=OUTPUT_PATH,
        options=MergeOptions(
            lora_merge_cache=LORA_MERGE_CACHE,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=COPY_TOKENIZER,
            lazy_unpickle=LAZY_UNPICKLE,
            low_cpu_memory=True,
        ),
    )
    
    torch.cuda.empty_cache()        # 🧹 Clear unreferenced GPU memory
    torch.cuda.ipc_collect()        # ♻️ Release CUDA inter-process memory (optional but good in notebooks)