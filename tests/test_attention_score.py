
import torch
from dsaa_algorithm.attention_score import (
    calculate_attention_importance
)


def test_attention_score_shape():
    attention = torch.rand(4, 1, 8, 16, 16)

    scores = calculate_attention_importance(attention)

    assert scores.shape == (16,)


def test_attention_score_range():
    attention = torch.rand(4, 1, 8, 16, 16)

    scores = calculate_attention_importance(attention)

    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)


def test_constant_attention():
    attention = torch.ones(4, 1, 8, 16, 16)

    scores = calculate_attention_importance(attention)

    assert torch.all(scores == 0)


if __name__ == "__main__":
    test_attention_score_shape()
    test_attention_score_range()
    test_constant_attention()

    print("All tests passed!")
