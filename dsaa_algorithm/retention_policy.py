
import math
import torch


def select_tokens_to_retain(
    combined_scores,
    memory_pressure,
    pressure_threshold=0.8,
    max_eviction_fraction=0.5,
    protected_recent_tokens=4
):
    """
    Select token indices to retain based on importance
    and memory pressure.

    Returns indices in their original sequence order.
    """

    scores = torch.as_tensor(
        combined_scores,
        dtype=torch.float32
    ).detach().cpu()

    if scores.ndim != 1:
        raise ValueError("Scores must be one-dimensional")

    if not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("Scores must be between 0 and 1")

    if not 0 <= memory_pressure <= 1:
        raise ValueError("Memory pressure must be between 0 and 1")

    if not 0 <= pressure_threshold < 1:
        raise ValueError("Invalid pressure threshold")

    if not 0 <= max_eviction_fraction < 1:
        raise ValueError("Invalid eviction fraction")

    if protected_recent_tokens < 0:
        raise ValueError("Protected token count cannot be negative")

    num_tokens = len(scores)

    if num_tokens == 0:
        return []

    # No eviction until pressure exceeds the threshold.
    if memory_pressure <= pressure_threshold:
        return list(range(num_tokens))

    # Calculate how aggressively to evict.
    normalized_pressure = (
        (memory_pressure - pressure_threshold)
        / (1 - pressure_threshold)
    )

    eviction_fraction = (
        normalized_pressure * max_eviction_fraction
    )

    eviction_count = math.floor(
        num_tokens * eviction_fraction
    )

    # Protect the most recent tokens.
    protected_start = max(
        0, num_tokens - protected_recent_tokens
    )

    eligible_indices = list(range(protected_start))

    eviction_count = min(
        eviction_count, len(eligible_indices)
    )

    # Evict the least important eligible tokens.
    eviction_indices = sorted(
        eligible_indices,
        key=lambda i: (scores[i].item(), i)
    )[:eviction_count]

    eviction_set = set(eviction_indices)

    retained_indices = [
        i for i in range(num_tokens)
        if i not in eviction_set
    ]

    return retained_indices
