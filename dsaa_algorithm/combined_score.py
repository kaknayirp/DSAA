
import torch


def calculate_combined_score(
    attention_scores,
    semantic_scores,
    recency_scores,
    attention_weight=0.4,
    semantic_weight=0.4,
    recency_weight=0.2
):
    """
    Combine attention, semantic and recency importance.

    Memory pressure is handled separately by the eviction policy.
    """

    weights = [
        attention_weight,
        semantic_weight,
        recency_weight
    ]

    if any(w < 0 for w in weights):
        raise ValueError("Weights cannot be negative")

    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("At least one weight must be positive")

    # Use CPU tensors for consistent scoring across devices.
    attention = torch.as_tensor(attention_scores).detach().to(
        device="cpu", dtype=torch.float32
    )
    semantic = torch.as_tensor(semantic_scores).detach().to(
        device="cpu", dtype=torch.float32
    )
    recency = torch.as_tensor(recency_scores).detach().to(
        device="cpu", dtype=torch.float32
    )

    if attention.ndim != 1 or semantic.ndim != 1 or recency.ndim != 1:
        raise ValueError("All score inputs must be one-dimensional")

    if not (attention.shape == semantic.shape == recency.shape):
        raise ValueError("All score vectors must have equal lengths")

    for scores in (attention, semantic, recency):
        if not torch.isfinite(scores).all():
            raise ValueError("Scores must contain only finite values")

        if torch.any(scores < 0) or torch.any(scores > 1):
            raise ValueError("Input scores must be between 0 and 1")

    combined = (
        attention_weight * attention +
        semantic_weight * semantic +
        recency_weight * recency
    ) / total_weight

    return combined
