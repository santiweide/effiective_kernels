from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add k gist tokens
k = 4
gist_tokens = [f"<gist_{i}>" for i in range(k)]
tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})

# Resize embedding matrix
model.resize_token_embeddings(len(tokenizer))

# (Optional) Custom initialization
with torch.no_grad():
    embed = model.get_input_embeddings().weight
    for token in gist_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embed[token_id].normal_(mean=0.0, std=embed.std())

print("Initialized gist tokens:", gist_tokens)

