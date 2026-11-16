
from evaluation.simulate_cache import create_synthetic_kv_cache
from evaluation.benchmark_eviction import benchmark_eviction_strategies


def test_benchmark_strategies():

    sequence_length = 10
    budget = 5

    cache = create_synthetic_kv_cache(
        num_layers=2,
        batch_size=1,
        num_heads=2,
        seq_len=sequence_length,
        head_dim=4
    )

    attention = [
        0.1, 0.2, 0.9, 0.3, 0.8,
        0.4, 0.7, 0.5, 0.6, 1.0
    ]

    semantic = [
        0.9, 0.8, 0.7, 0.6, 0.5,
        0.4, 0.3, 0.2, 0.9, 1.0
    ]

    recency = [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0
    ]

    # Deliberately different from insertion order.
    last_access = [
        9, 1, 8, 2, 7,
        3, 6, 4, 5, 10
    ]

    important_tokens = [0, 2, 4, 8, 9]

    results = benchmark_eviction_strategies(
        cache=cache,
        attention_scores=attention,
        semantic_scores=semantic,
        recency_scores=recency,
        last_access_times=last_access,
        retention_budget=budget,
        important_token_indices=important_tokens,
        protected_recent_tokens=2
    )

    assert set(results.keys()) == {"FIFO", "LRU", "DSAA"}

    for name, result in results.items():

        assert result["retained_tokens"] == budget
        assert result["evicted_tokens"] == sequence_length - budget

        assert result["optimized_bytes"] < result["original_bytes"]

        assert 0 <= result["memory_saved_percentage"] <= 100

        assert 0 <= result["important_token_retention_percentage"] <= 100

        print(f"\n{name} Results")
        print("Retained indices:", result["retained_indices"])
        print("Important tokens retained:",
              result["important_tokens_retained"])
        print("Memory saved:",
              result["memory_saved_percentage"], "%")
        print("Selection time:",
              result["selection_time_ms"], "ms")

    # DSAA must protect the two most recent sequence positions.
    assert 8 in results["DSAA"]["retained_indices"]
    assert 9 in results["DSAA"]["retained_indices"]

    print("\nAll benchmarking tests passed!")


if __name__ == "__main__":
    test_benchmark_strategies()
