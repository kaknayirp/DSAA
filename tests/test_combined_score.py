
import torch

from dsaa_algorithm.combined_score import calculate_combined_score


def test_combined_score():
    attention = torch.tensor([1.0, 0.5, 0.0])
    semantic = torch.tensor([0.5, 1.0, 0.0])
    recency = torch.tensor([0.0, 0.5, 1.0])

    scores = calculate_combined_score(
        attention, semantic, recency
    )

    expected = torch.tensor([0.6, 0.7, 0.2])

    assert torch.allclose(scores, expected)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)

    print("Combined scoring test passed!")
    print("Combined scores:", scores.tolist())


def test_invalid_weights():
    try:
        calculate_combined_score(
            [1.0], [1.0], [1.0],
            -0.1, 0.5, 0.6
        )
        assert False, "Expected ValueError"
    except ValueError:
        print("Invalid weights test passed!")


if __name__ == "__main__":
    test_combined_score()
    test_invalid_weights()
