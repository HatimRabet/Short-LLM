import torch
import datasets

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from src.block_similarity.layer_similarity_analysis import run_layer_similairities_2

if __name__ == "__main__":
    model_path = "google/gemma-3-1b-it"
    model_name = "gemma_1b"
    
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
    
    dataset_size = 10000
    dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True).take(dataset_size)
    
    batch_size = 32
    max_length = 128
    
    # run_layer_similairities_2(model, tokenizer, dataset, batch_size, max_length, 25, model_name, device)
    
    for n_layers_to_skip in range(1, 27):
        print("n = ", n_layers_to_skip, "\n")
        run_layer_similairities_2(model, tokenizer, dataset, batch_size, max_length, n_layers_to_skip, model_name, device)
        
        print("#####################\n\n")