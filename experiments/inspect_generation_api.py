
import inspect
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)

print("\n--- DynamicCache ---")
print("Constructor:", inspect.signature(DynamicCache))
print("Methods:", [
    name for name in dir(DynamicCache)
    if not name.startswith("_")
])

print("\n--- DynamicLayer ---")
print("Constructor:", inspect.signature(DynamicLayer))
print("Methods:", [
    name for name in dir(DynamicLayer)
    if not name.startswith("_")
])


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("\n--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n--- Loading model ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
)

model.eval()

print("\n--- Model forward signature ---")
print(inspect.signature(model.forward))

print("\n--- Inner model forward signature ---")
print(inspect.signature(model.model.forward))

print("\n--- Model generation configuration ---")
print("Use cache:", model.generation_config.use_cache)
print("EOS token:", model.generation_config.eos_token_id)

print("\nAPI inspection completed.")

print("\n--- DynamicCache constructor source ---")
print(inspect.getsource(DynamicCache.__init__))

print("\n--- DynamicLayer update source ---")
print(inspect.getsource(DynamicLayer.update))

print("\n--- DynamicCache update source ---")
print(inspect.getsource(DynamicCache.update))

print("\n--- Model cache configuration ---")
print("Number of layers:", model.config.num_hidden_layers)
print("Number of KV heads:", model.config.num_key_value_heads)
print("Hidden size:", model.config.hidden_size)
head_dim = (
    model.config.hidden_size
    // model.config.num_attention_heads
)

print("Head dimension:", head_dim)

print("\n--- Actual cache layer structure ---")

sample_inputs = tokenizer(
    "Testing KV cache construction.",
    return_tensors="pt"
).to(next(model.parameters()).device)

with torch.inference_mode():
    sample_outputs = model(**sample_inputs, use_cache=True)

cache = sample_outputs.past_key_values

print("Cache initialized:", cache.is_initialized)
print("Number of cache layers:", len(cache.layers))

first_layer = cache.layers[0]

print("First layer type:", type(first_layer))
print("Keys shape:", first_layer.keys.shape)
print("Values shape:", first_layer.values.shape)
print("Keys dtype:", first_layer.keys.dtype)
print("Values dtype:", first_layer.values.dtype)
