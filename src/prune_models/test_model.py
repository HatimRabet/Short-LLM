import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "your-aitocausallm-model"  
MAX_LENGTH = 128  # Limit for response length

# Function to format dataset prompts
def format_prompt_mmlu(example):
    question = example["question"]
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example["choices"])])
    return f"Question: {question}\n{choices}\nAnswer:"

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
        output = model.generate(**inputs, max_length=MAX_LENGTH)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Evaluation function
def evaluate_dataset(dataset_name, model_name, dataset_info, device="cuda"):
    print(f"\nEvaluating {dataset_name.upper()}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset(dataset_info["name"], dataset_info["subset"])["test"]
    
    # Preprocess dataset
    dataset = dataset.map(lambda x: {"prompt": FORMATTERS[dataset_name](x)})

    # Generate model responses
    predictions = []
    ground_truths = []
    for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
        prompt = example["prompt"]
        response = generate_answer(prompt, model, tokenizer, device)
        predicted_answer = response.strip().split("\n")[-1]  # Extract last line as answer
        predictions.append(predicted_answer)

        # Ground truth extraction
        if dataset_name == "mmlu":
            ground_truths.append(example["answer"])
        elif dataset_name == "hellaswag":
            ground_truths.append(example["label"])  # The correct answer index
        elif dataset_name == "truthfulqa":
            ground_truths.append(example["mc1_targets"]["labels"].index(1))  # Correct index

    # Compute accuracy
    correct_count = sum([pred == str(gt) for pred, gt in zip(predictions, ground_truths)])
    accuracy = correct_count / len(ground_truths)
    print(f"{dataset_name.upper()} Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    # Load Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATASETS = {
        "mmlu": {"name": "cais/mmlu", "subset": "mathematics"},
        "hellaswag": {"name": "Rowan/hellaswag", "subset": None},
        "truthfulqa": {"name": "truthful_qa", "subset": "multiple_choice"},
    }

    import argparse

    parser = argparse.ArgumentParser(description="Model Evaluation On a Dataset")
    parser.add_argument("--model_name", type=str, help="Model Name", default="google/gemma-3-1b-it")
    parser.add_argument("--dataset_name", type=str, help="Dataset Name", default="mmlu", choices=["mmlu", "hellaswag", "truthfulqa"])

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name

    info = DATASETS[DATASET_NAME]

    # Run evaluation on all datasets
    results = evaluate_dataset(DATASET_NAME, MODEL_NAME, info, device)
    print("\nFinal Results:", results)

