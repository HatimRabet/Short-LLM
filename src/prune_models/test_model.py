import torch
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm


# MODEL_NAME = "src/prune_models/pruned_models"
# MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
MODEL_NAME = "google/gemma-3-1b-it"
MAX_LENGTH = 128  # Limit for response length


quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16) 


# Function to format dataset prompts
def format_prompt_mmlu(example):
    question = example["question"]
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example["choices"])])
    return f"Choose the correct answer from the options below.\nQuestion: {question}\n{choices}\nAnswer:"

def format_prompt_hellaswag(example):
    return f"Sentence: {example['ctx']}\nOptions: {example['endings']}\nAnswer:"

def format_prompt_truthfulqa(example):
    question = example["question"]
    choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(example["mc1_targets"]["choices"])])
    return f"Question: {question}\n{choices}\nAnswer:"

FORMATTERS = {
    "mmlu": format_prompt_mmlu,
    "hellaswag": format_prompt_hellaswag,
    "truthfulqa": format_prompt_truthfulqa,
}

# Function to generate model responses
def generate_answer(prompt, model, tokenizer, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Evaluation function
def evaluate_dataset(dataset_name, model_name, dataset_info, device="cuda", quantization_config = None):
    print("Loading model...")
    if not quantization_config:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else :
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            quantization_config = quantization_config
            )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token


    
    # Load dataset
    if dataset_name == "truthfulqa":
        dataset = load_dataset(dataset_info["name"], dataset_info["subset"])["validation"]

    else:
        dataset = load_dataset(dataset_info["name"], dataset_info["subset"])["test"]
    
    # Preprocess dataset
    dataset = dataset.map(lambda x: {"prompt": FORMATTERS[dataset_name](x)})

    # Generate model responses
    predictions = []
    ground_truths = []
    
    print(f"\nEvaluating {dataset_name.upper()}...")
    for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
        # print(example)
        prompt = example["prompt"]
        response = generate_answer(prompt, model, tokenizer, device)
        # print("respone :\n", response)
        try :
            predicted_answer = response.strip().split("\n")[-1]  # Extract last line as answer

            # print("\npredicted ",predicted_answer)
            predicted_answer = predicted_answer.split(":")[1].strip()[0] 
            # print(predicted_answer)
            answer = ord(predicted_answer) - ord('A') + 1
        except :
            answer = -1
        predictions.append(answer)
        # print("\norder ",answer)

        # Ground truth extraction
        if dataset_name == "mmlu":
            ground_truths.append(example["answer"])
    
            #print(example["answer"])
        elif dataset_name == "hellaswag":
            ground_truths.append(example["label"])  # The correct answer index
        elif dataset_name == "truthfulqa":
            ground_truths.append(example["mc1_targets"]["labels"].index(1))  # Correct index

    # Compute accuracy
    correct_count = sum([pred == gt for pred, gt in zip(predictions, ground_truths)])
    accuracy = correct_count / len(ground_truths)
    print(f"{dataset_name.upper()} Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    # Load Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATASETS = {
        "mmlu": {"name": "cais/mmlu", "subset": "abstract_algebra"},
        "hellaswag": {"name": "Rowan/hellaswag", "subset": None},
        "truthfulqa": {"name": "truthful_qa", "subset": "multiple_choice"},
    }

    import argparse

    parser = argparse.ArgumentParser(description="Model Evaluation On a Dataset")
    parser.add_argument("--model_name", type=str, help="Model Name", default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, help="Dataset Name", default="mmlu", choices=["mmlu", "hellaswag", "truthfulqa"])

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name

    info = DATASETS[DATASET_NAME]

    # Run evaluation on all datasets
    results = evaluate_dataset(DATASET_NAME, MODEL_NAME, info, device, quantization_config=quantization_config)
    # print("\nFinal Results:", results)
    results_file = "tmp_results.json"

# Load existing results
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # Append new result
    all_results.append(results)

    # Save updated list
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    
    torch.cuda.empty_cache()        # 🧹 Clear unreferenced GPU memory
    torch.cuda.ipc_collect()        # ♻️ Release CUDA inter-process memory (optional but good in notebooks)