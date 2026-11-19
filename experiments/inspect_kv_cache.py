
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Select available device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Loading model on:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
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

# Generate actual KV cache
with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
        return_dict=True
    )

past_key_values = outputs.past_key_values

if past_key_values is None:
    raise RuntimeError("Model did not return a KV cache.")

# Support modern Transformers DynamicCache
if hasattr(past_key_values, "to_legacy_cache"):
    cache = past_key_values.to_legacy_cache()
else:
    cache = past_key_values


print("\nREAL KV CACHE DIAGNOSTICS")

print("Number of layers:", len(cache.layers))

total_bytes = 0

for layer_index, layer in enumerate(cache.layers):

    key = layer.keys
    value = layer.values

    if not isinstance(key, torch.Tensor):
        raise TypeError(f"Layer {layer_index} key is not a tensor")

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Layer {layer_index} value is not a tensor")

    if not torch.isfinite(key).all():
        raise ValueError(f"Invalid key values in layer {layer_index}")

    if not torch.isfinite(value).all():
        raise ValueError(f"Invalid value values in layer {layer_index}")

    layer_bytes = (
        key.numel() * key.element_size()
        + value.numel() * value.element_size()
    )

    total_bytes += layer_bytes

    print(f"\nLayer {layer_index}")
    print("Key shape:", tuple(key.shape))
    print("Value shape:", tuple(value.shape))
    print("Key dtype:", key.dtype)
    print("Value dtype:", value.dtype)
    print("Layer cache size:", layer_bytes, "bytes")

print("\nTOTAL KV CACHE SIZE")
print("Total bytes:", total_bytes)
print("Total MB:", round(total_bytes / (1024 ** 2), 4))

print("\nInput tokens:", inputs["input_ids"].shape[-1])
print("KV cache extraction successful!")

