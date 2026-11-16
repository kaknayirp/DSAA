
import torch

from dsaa_algorithm.recency_score import (
    calculate_recency_importance
)


def test_recency_score():
    scores = calculate_recency_importance(10)

    assert scores.shape == (10,)
    assert scores[0] == 0
    assert scores[-1] == 1

    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)

    assert torch.all(scores[1:] >= scores[:-1])

    print("Recency scoring test passed!")
    print("Scores:", scores.tolist())


if __name__ == "__main__":
    test_recency_score()
