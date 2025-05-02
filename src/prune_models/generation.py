import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# ckpt = "google/gemma-3-1b-pt"
model_path = "src/prune_models/pruned_models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Gemma3ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Eiffel tower is located in"
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=1, do_sample=False)
    generation = generation[0]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
