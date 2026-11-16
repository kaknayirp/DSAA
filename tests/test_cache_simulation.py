
from evaluation.simulate_cache import (
    create_synthetic_kv_cache,
    evaluate_cache_reduction
)


def test_cache_simulation():

    cache = create_synthetic_kv_cache(
        num_layers=2,
        batch_size=1,
        num_heads=2,
        seq_len=10,
        head_dim=4
    )

    retained_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9]

    results = evaluate_cache_reduction(
        cache,
        retained_indices
    )

    assert results["original_tokens"] == 10
    assert results["retained_tokens"] == 9
    assert results["evicted_tokens"] == 1

    assert results["optimized_bytes"] < results["original_bytes"]

    assert abs(
        results["memory_saved_percentage"] - 10.0
    ) < 0.001

    print("KV cache simulation test passed!")
    print("Original memory:", results["original_bytes"], "bytes")
    print("Optimized memory:", results["optimized_bytes"], "bytes")
    print("Memory saved:", results["memory_saved_bytes"], "bytes")
    print("Memory saved percentage:",
          results["memory_saved_percentage"], "%")


if __name__ == "__main__":
    test_cache_simulation()
