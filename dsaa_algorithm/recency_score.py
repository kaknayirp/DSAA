
import torch


def calculate_recency_importance(num_tokens):
    """
    Assign higher importance to newer tokens.

    Oldest token: 0
    Newest token: 1
    """

    if num_tokens < 0:
        raise ValueError("num_tokens cannot be negative")

    if num_tokens == 0:
        return torch.empty(0, dtype=torch.float32)

    if num_tokens == 1:
        return torch.ones(1, dtype=torch.float32)

    scores = torch.arange(
        num_tokens,
        dtype=torch.float32
    )

    scores = scores / (num_tokens - 1)

    return scores
