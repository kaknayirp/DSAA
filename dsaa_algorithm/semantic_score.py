
import re
import torch


def calculate_semantic_importance(text, tokenizer):
    """
    Lightweight heuristic semantic importance scorer.

    Returns one importance score per tokenizer token.
    This is a baseline, not a full semantic understanding model.
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    offsets = encoded["offset_mapping"][0].tolist()
    input_ids = encoded["input_ids"][0]

    scores = torch.zeros(len(offsets), dtype=torch.float32)

    # Character spans that deserve additional importance.
    patterns = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", 0.9),  # Names
        (r"\b\d+(?:\.\d+)?%?\b", 0.9),                # Numbers
        (r"\b(?:AI|ML|NLP|LLM|Python|PyTorch)\b", 0.8),
        (r"\b(?:important|because|therefore|must|should|requires)\b",
         0.7),
    ]

    spans = []

    for pattern, importance in patterns:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), importance))

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue

        token_score = 0.1

        for span_start, span_end, importance in spans:
            if start < span_end and end > span_start:
                token_score = max(token_score, importance)

        scores[i] = token_score

    # Normalize scores to [0, 1].
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (
            scores.max() - scores.min()
        )

    return scores
