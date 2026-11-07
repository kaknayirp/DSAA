
import torch


def calculate_attention_importance(attentions, query_window=4):
    """
    Calculate token importance using attention weights.

    Args:
        attentions: Attention tensors from transformer layers.
        query_window: Number of recent query positions to consider.

    Returns:
        Normalised importance score for each token.
    """

    if isinstance(attentions, (tuple, list)):
        attentions = torch.stack(attentions)

    if attentions.ndim != 5:
        raise ValueError(
            "Expected attention shape: "
            "[layers, batch, heads, query, key]"
        )

    # Average attention across layers
    scores = attentions.float().mean(dim=0)

    # Average across batch and attention heads
    scores = scores.mean(dim=(0, 1))

    # Consider recent query positions
    scores = scores[-query_window:, :].mean(dim=0)

    # Normalise scores between 0 and 1
    min_score = scores.min()
    max_score = scores.max()

    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = torch.zeros_like(scores)

    return scores


if __name__ == "__main__":
    # Dummy attention data for initial testing
    dummy_attention = torch.rand(4, 1, 8, 16, 16)

    importance = calculate_attention_importance(
        dummy_attention
    )

    print("Token importance scores:")
    print(importance)

    print("Number of tokens:", len(importance))
