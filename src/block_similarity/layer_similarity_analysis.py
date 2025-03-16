from src.block_similarity.utils import angular_distance, compute_block_distances, get_last_non_padded_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from torch.utils.data import DataLoader
import datasets
import numpy as np 
import csv
from tqdm import tqdm
import os


torch.manual_seed(0)
np.random.seed(0)

def run_layer_similairities(model_path, model_name, dataset_name, batch_size, max_length, n_layers_to_skip, dataset_size=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Since I don't have enough ressources
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16) 

    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                device_map="auto",
                                                quantization_config=quantization_config,
                                                output_hidden_states=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    dataset = datasets.load_dataset(dataset_name, split="train")

    if dataset_size:
        dataset = dataset.select(range(dataset_size))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    all_distances = [[] for _ in range(model.config.num_hidden_layers - n_layers_to_skip)]

    for batch in tqdm(dataloader, desc="Processing Batches"):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states

        last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention_mask)

        # We remove input
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]

        block_distances = compute_block_distances(last_non_padded_hidden_states, n_layers_to_skip)

        for i, block_distance in enumerate(block_distances):
            all_distances[i].append(block_distance)

    average_block_distances = [np.mean(distances) for distances in all_distances]

    min_distance = float('inf')  
    min_distance_layer = 0 
    
    os.makedirs("results/", exist_ok=True)

    with open(f'results/layer_distances_{model_name}.csv', 'w', newline='') as csvfile:
        csvfile.write(f"Analysis of Layer Similarities for {model_name} \n")
        csvfile.write(f"Number of Layers to Skip : {n_layers_to_skip} \n")
        fieldnames = ['block_start', 'block_end', 'average_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, average_distance in enumerate(average_block_distances):
            writer.writerow({
                'block_start': i + 1, 
                'block_end': i + 1 + n_layers_to_skip,
                'average_distance': average_distance
            })
            
            if average_distance < min_distance:
                min_distance = average_distance
                min_distance_layer = i + 1

        
        csvfile.write(f"Layer {min_distance_layer} to {min_distance_layer + n_layers_to_skip} is the best block to prune.\n")







if __name__ == "__main__":
    # model_path = "facebook/opt-125m"
    model_path = "google/gemma-3-1b-it"
    dataset_name = "arcee-ai/sec-data-mini"
    model_name = "gemma_1b"

    batch_size = 64
    max_length = 128

    n_layers_to_skip = 5

    run_layer_similairities(model_path, model_name, dataset_name, batch_size, max_length, n_layers_to_skip)