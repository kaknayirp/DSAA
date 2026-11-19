
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dsaa_algorithm.attention_score import (
    calculate_attention_importance
)
from dsaa_algorithm.semantic_score import (
    calculate_semantic_importance
)
from dsaa_algorithm.recency_score import (
    calculate_recency_importance
)
from dsaa_algorithm.combined_score import (
    calculate_combined_score
)
from dsaa_algorithm.memory_pressure import (
    calculate_memory_pressure
)
from dsaa_algorithm.retention_policy import (
    select_tokens_to_retain
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Controlled experimental capacity.
# This is a simulation parameter, not actual system memory pressure.
MAX_CAPACITY = 20

PROTECTED_RECENT_TOKENS = 4


def calculate_cache_bytes(cache_layers):
    total_bytes = 0

    for layer in cache_layers:
        key, value = layer

        total_bytes += (
            key.numel() * key.element_size()
            + value.numel() * value.element_size()
        )

    return total_bytes


def main():

    # Select device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading model on:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager"
    )

    model.to(device)
    model.eval()

    # Input text
    text = """
    Rahul is an AI engineering student.
    He is developing a smart waste management system.
    He enjoys machine learning and deep learning.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    num_tokens = inputs["input_ids"].shape[-1]

    print("\nInput tokens:", num_tokens)

    # Extract attention weights and actual KV cache
    with torch.inference_mode():

        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
            return_dict=True
        )

    past_key_values = outputs.past_key_values
    attentions = outputs.attentions

    if past_key_values is None:
        raise RuntimeError("KV cache was not returned.")

    if attentions is None:
        raise RuntimeError("Attention weights were not returned.")

    cache_layers = []

    for layer in past_key_values.layers:

        key = layer.keys
        value = layer.values

        if key is None or value is None:
            raise RuntimeError("Uninitialized KV cache layer.")

        if key.shape[-2] != num_tokens:
            raise ValueError("Key sequence length mismatch.")

        if value.shape[-2] != num_tokens:
            raise ValueError("Value sequence length mismatch.")

        cache_layers.append((key, value))

    original_bytes = calculate_cache_bytes(cache_layers)

    print("Transformer layers:", len(cache_layers))
    print(
        "Original cache size:",
        round(original_bytes / (1024 ** 2), 4),
        "MB"
    )

    # STEP 1: Attention importance
    attention_scores = calculate_attention_importance(
        attentions,
        query_window=4
    )

    # STEP 2: Semantic importance
    semantic_scores = calculate_semantic_importance(
        text,
        tokenizer
    )

    # STEP 3: Recency importance
    recency_scores = calculate_recency_importance(
        num_tokens
    )

    # Verify token alignment
    if not (
        len(attention_scores)
        == len(semantic_scores)
        == len(recency_scores)
        == num_tokens
    ):
        raise ValueError("Importance score lengths do not match tokens.")

    print("\nImportance score lengths:")
    print("Attention:", len(attention_scores))
    print("Semantic:", len(semantic_scores))
    print("Recency:", len(recency_scores))

    # STEP 4: Combined DSAA importance
    combined_scores = calculate_combined_score(
        attention_scores,
        semantic_scores,
        recency_scores
    )

    # STEP 5: Controlled memory pressure
    memory_pressure = calculate_memory_pressure(
        current_tokens=num_tokens,
        max_capacity=MAX_CAPACITY
    )

    print("\nMemory pressure:", memory_pressure)

    # STEP 6: DSAA retention decision
    retained_indices = select_tokens_to_retain(
        combined_scores=combined_scores,
        memory_pressure=memory_pressure,
        pressure_threshold=0.8,
        max_eviction_fraction=0.5,
        protected_recent_tokens=PROTECTED_RECENT_TOKENS
    )

    print("\nDSAA RETENTION RESULTS")
    print("Original tokens:", num_tokens)
    print("Retained tokens:", len(retained_indices))
    print("Evicted tokens:", num_tokens - len(retained_indices))
    print("Retained indices:", retained_indices)

    # STEP 7: Compact actual KV tensors.
    # Keep the original cache untouched.
    compacted_layers = []

    for key, value in cache_layers:

        indices = torch.tensor(
            retained_indices,
            dtype=torch.long,
            device=key.device
        )

        compacted_key = key.index_select(-2, indices)

        compacted_value = value.index_select(-2, indices)

        compacted_layers.append(
            (compacted_key, compacted_value)
        )

    optimized_bytes = calculate_cache_bytes(compacted_layers)

    # STEP 8: Compare memory sizes
    saved_bytes = original_bytes - optimized_bytes

    saved_percentage = (
        saved_bytes / original_bytes * 100
        if original_bytes else 0
    )

    print("\nREAL KV CACHE COMPACTION RESULTS")

    print("Original cache bytes:", original_bytes)
    print("Optimized cache bytes:", optimized_bytes)
    print("Memory saved:", saved_bytes, "bytes")
    print("Memory reduction:", round(saved_percentage, 2), "%")

    print("\nFirst layer after compaction:")
    print("Key shape:", tuple(compacted_layers[0][0].shape))
    print("Value shape:", tuple(compacted_layers[0][1].shape))

    print("\nDSAA REAL CACHE EXPERIMENT SUCCESSFUL!")


if __name__ == "__main__":
    main()

