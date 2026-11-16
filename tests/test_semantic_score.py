
from transformers import AutoTokenizer
from dsaa_algorithm.semantic_score import calculate_semantic_importance


def test_semantic_score():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    text = (
        "Rahul is an AI engineering student. "
        "Python is important. The system requires 95% accuracy."
    )

    scores = calculate_semantic_importance(text, tokenizer)

    assert len(scores) == len(
        tokenizer(
            text,
            return_offsets_mapping=True
        )["input_ids"]
    )

    assert scores.min() >= 0
    assert scores.max() <= 1

    print("Semantic scoring test passed!")
    print("Number of tokens:", len(scores))
    print("Scores:", scores.tolist())


if __name__ == "__main__":
    test_semantic_score()
