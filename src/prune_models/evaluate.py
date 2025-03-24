import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import re
import yaml
from src.prune_models.test_model import evaluate_dataset
import torch
import argparse
from tqdm import tqdm

CONFIG_YML = "src/prune_models/prune.yaml"
MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
PRUNED_MODEL = "src/prune_models/pruned_models"

DATASETS = {
    "mmlu": {"name": "cais/mmlu", "subset": "abstract_algebra"},
    "hellaswag": {"name": "Rowan/hellaswag", "subset": None},
    "truthfulqa": {"name": "truthful_qa", "subset": "multiple_choice"},
}


# Function to generate slices based on start and end layers
def generate_yaml(layer_blocks, model_name=MODEL_NAME):
    slices = []
    for start, end in layer_blocks:
        slice_entry = {
            'sources': [
                {
                    'model': model_name,
                    'layer_range': [start, end]
                }
            ]
        }
        slices.append(slice_entry)

    config = {
        'slices': slices,
        'merge_method': 'passthrough',
        'dtype': 'bfloat16'
    }

    # Convert dictionary to YAML
    return yaml.dump(config, default_flow_style=False)


def evaluate_pruned_models(directory, dataset_name="mmlu", model_name=MODEL_NAME):
    # Collect all CSV files, including the main file
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)  # Sort numerically

    # Initialize List to Store results
    optimal_end_start = []

    for i, file in enumerate(files):
        filepath = os.path.join(directory, file)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Extract average_distance values
        for line in lines[-1:]:  # Skip header lines
            if line.startswith("Layer"):
                numbers = re.findall(r'\d+', line)
                numbers = [int(num) for num in numbers]
                optimal_end_start.append({"start layer" : min(numbers[0], numbers[1]), "end layer" : max(numbers[0], numbers[1])})

    print("\n\n Starting evaluation ...")
    performance_res = []
    for block in tqdm(optimal_end_start):
        start_layer = block["start layer"]
        end_layer = block["end layer"]

        # Define the start and end layers for each block 
        layer_blocks = [
            (0, start_layer),
            (end_layer+1, 36)
        ]

        # Generate the YAML file content
        yaml_content = generate_yaml(layer_blocks)

        # Print or save the content to a file
        print(yaml_content)

        # Optional: Save to a file
        with open(CONFIG_YML, 'w') as file:
            file.write(yaml_content)

        with open("src/prune_models/prune.py") as script:
            exec(script.read())

        # Load Model and Tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        info = DATASETS[dataset_name]

        # Run evaluation on all datasets
        results = evaluate_dataset(dataset_name, PRUNED_MODEL, info, device)
        print("\nFinal Results:", results)

        performance_res.append(results)
    return performance_res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Evaluation On a Dataset")
    parser.add_argument("--model_name", type=str, help="Model Name", default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, help="Dataset Name", default="mmlu", choices=["mmlu", "hellaswag", "truthfulqa"])
    parser.add_argument("--directory", type=str, help="Directory containing the CSV files", default="results/mmlu_ministral/")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    DIRECTORY = args.directory

    performance = evaluate_pruned_models(DIRECTORY, DATASET_NAME, MODEL_NAME)
    layers_to_skip = [n+1 for n in range(len(performance))]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plotting the graph
    ax.plot(layers_to_skip, performance, label=f'Accuracy on {DATASET_NAME} Dataset', color='green', marker='o')

    # Adding title and labels
    ax.set_title("Model Accuracy as a function of the numbers of Skipped Layers")  # Title
    ax.set_xlabel("N Layers Skipped")  # X-axis label
    ax.set_ylabel("Accuracy")  # Y-axis label

    # Adding a grid
    ax.grid(True)

    # Adding a legend
    ax.legend()
    plt.savefig(DIRECTORY + "performance")

    # Show the plot
    plt.show()




