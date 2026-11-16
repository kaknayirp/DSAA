
import torch


def create_synthetic_kv_cache(
    num_layers=4,
    batch_size=1,
    num_heads=4,
    seq_len=100,
    head_dim=64
):
    """
    Create a synthetic KV cache for testing.
    Each layer contains a key tensor and a value tensor.
    """

    dimensions = [
        num_layers, batch_size,
        num_heads, seq_len, head_dim
    ]

    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    cache = []

    for _ in range(num_layers):
        key = torch.randn(
            batch_size, num_heads, seq_len, head_dim
        )

        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim
        )

        cache.append((key, value))

    return tuple(cache)


def get_cache_size_bytes(cache):
    """Calculate total memory occupied by KV tensors."""

    total_bytes = 0

    for layer in cache:
        key, value = layer

        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()

    return total_bytes


def retain_kv_cache_tokens(cache, retained_indices):
    """
    Retain selected sequence positions across all layers.
    Original token order is preserved.
    """

    if not cache:
        if retained_indices:
            raise ValueError("Cannot retain tokens from an empty cache")
        return tuple()

    first_key = cache[0][0]
    seq_len = first_key.shape[-2]

    indices = list(retained_indices)

    if indices != sorted(set(indices)):
        raise ValueError(
            "Indices must be unique and in ascending order"
        )

    if any(i < 0 or i >= seq_len for i in indices):
        raise ValueError("Invalid token index")

    retained_cache = []

    for key, value in cache:
        if key.shape[-2] != seq_len or value.shape[-2] != seq_len:
            raise ValueError("Inconsistent cache sequence lengths")

        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=key.device
        )

        new_key = key.index_select(-2, index_tensor)

        index_tensor = index_tensor.to(value.device)
        new_value = value.index_select(-2, index_tensor)

        retained_cache.append((new_key, new_value))

    return tuple(retained_cache)


def evaluate_cache_reduction(cache, retained_indices):
    """Measure simulated KV-cache memory savings."""

    original_bytes = get_cache_size_bytes(cache)

    optimized_cache = retain_kv_cache_tokens(
        cache, retained_indices
    )

    optimized_bytes = get_cache_size_bytes(optimized_cache)

    original_tokens = (
        cache[0][0].shape[-2] if cache else 0
    )

    retained_tokens = len(retained_indices)

    memory_saved = original_bytes - optimized_bytes

    savings_percentage = (
        (memory_saved / original_bytes) * 100
        if original_bytes > 0 else 0
    )

    return {
        "original_tokens": original_tokens,
        "retained_tokens": retained_tokens,
        "evicted_tokens": original_tokens - retained_tokens,
        "original_bytes": original_bytes,
        "optimized_bytes": optimized_bytes,
        "memory_saved_bytes": memory_saved,
        "memory_saved_percentage": savings_percentage,
        "optimized_cache": optimized_cache
    }
