
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dsaa_algorithm.attention_score import (
    calculate_attention_importance
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Select available device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Loading model on:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Eager attention is important because we need
# the actual attention tensors as model outputs.
dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    attn_implementation="eager"
)

model.to(device)
model.eval()

# Sample input
text = """
Rahul is an AI engineering student.
He is developing a smart waste management system.
He enjoys machine learning and deep learning.
"""

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

# Extract attention weights
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=False
    )

attentions = outputs.attentions
for i, layer in enumerate(attentions):
    if not torch.isfinite(layer).all():
        raise ValueError(
            f"Invalid attention values detected in layer {i}"
        )

print("All attention tensors contain finite values!")
if attentions is None:
    raise RuntimeError("Attention weights were not returned.")

# Calculate DSAA attention importance
scores = calculate_attention_importance(
    attentions,
    query_window=4
)
print("\nATTENTION DIAGNOSTICS")

print("Number of layers:", len(attentions))

for i, layer in enumerate(attentions):
    print(
        f"Layer {i}:",
        "Shape:", layer.shape,
        "Min:", layer.float().min().item(),
        "Max:", layer.float().max().item(),
        "Mean:", layer.float().mean().item()
    )

print("\nFinal score statistics:")
print("Minimum:", scores.min().item())
print("Maximum:", scores.max().item())
print("Mean:", scores.mean().item())
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nREAL TOKEN IMPORTANCE SCORES\n")

for token, score in zip(tokens, scores):
    print(f"{token:25s} {score.item():.8f}")
dtype=dtype
print("\nTotal tokens:", len(tokens))
print("Attention extraction successful!")
