import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model with tied weights
model_path = "path/to/Qwen-2.5-3B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Untie the weights
if model.config.tie_word_embeddings:
    print("Untying embeddings...")
    model.config.tie_word_embeddings = False
    # Create a clone of the input embeddings for the output head
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())
    
    # Save the new "untied" model
    save_path = "path/to/Qwen-2.5-3B-Untied"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved untied model to {save_path}")
else:
    print("Model embeddings are already untied.")