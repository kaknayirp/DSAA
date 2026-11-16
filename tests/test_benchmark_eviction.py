
from evaluation.simulate_cache import create_synthetic_kv_cache
from evaluation.benchmark_eviction import benchmark_eviction_strategies


def test_benchmark_strategies():

    sequence_length = 100

    cache = create_synthetic_kv_cache(
        num_layers=2,
        batch_size=1,
        num_heads=2,
        seq_len=sequence_length,
        head_dim=4
    )

    attention = [
        ((i * 7) % 100) / 100
        for i in range(sequence_length)
    ]

    semantic = [
        ((i * 13) % 100) / 100
        for i in range(sequence_length)
    ]

    recency = [
        i / (sequence_length - 1)
        for i in range(sequence_length)
    ]

    last_access = [
        (i * 17) % 100
        for i in range(sequence_length)
    ]

    important_tokens = [
        2, 8, 15, 24, 37, 45, 56, 68, 79, 91
    ]

    budgets = [80, 60, 50, 30]

    for budget in budgets:

        protected_recent_tokens = min(2, budget)

        results = benchmark_eviction_strategies(
            cache=cache,
            attention_scores=attention,
            semantic_scores=semantic,
            recency_scores=recency,
            last_access_times=last_access,
            retention_budget=budget,
            important_token_indices=important_tokens,
            protected_recent_tokens=protected_recent_tokens,
            repeats=100,
            warmup_runs=5
        )

        assert set(results.keys()) == {"FIFO", "LRU", "DSAA"}

        for name, result in results.items():

            assert result["retained_tokens"] == budget
            assert result["evicted_tokens"] == sequence_length - budget

            assert result["optimized_bytes"] <= result["original_bytes"]

            assert 0 <= result["memory_saved_percentage"] <= 100

            assert result["benchmark_runs"] == 100

            assert result["mean_time_ms"] > 0
            assert result["std_time_ms"] >= 0

            assert len(result["retained_indices"]) == budget

        # DSAA must protect the latest two tokens.
        assert 98 in results["DSAA"]["retained_indices"]
        assert 99 in results["DSAA"]["retained_indices"]

        print(f"\n===== RETENTION BUDGET: {budget}% =====")

        for name, result in results.items():
            print(f"\n{name}")
            print("Important tokens retained:",
                  result["important_tokens_retained"], "/",
                  len(important_tokens))
            print("Memory saved:",
                  round(result["memory_saved_percentage"], 2), "%")
            print("Selection time:",
                  round(result["selection_time_ms"], 6), "ms")

        print(
            "DSAA score computation:",
            round(results["DSAA"]["score_computation_time_ms"], 6),
            "ms"
        )

        print(
            "DSAA total decision time:",
            round(results["DSAA"]["total_decision_time_ms"], 6),
            "ms"
        )

    print("\nAll benchmarking tests passed!")


if __name__ == "__main__":
    test_benchmark_strategies()
