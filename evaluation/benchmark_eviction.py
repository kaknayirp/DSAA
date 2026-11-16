
import time
import statistics

from dsaa_algorithm.combined_score import calculate_combined_score
from evaluation.simulate_cache import (
    get_cache_size_bytes,
    retain_kv_cache_tokens
)


def _measure(function, repeats, warmup_runs):
    """Measure a function after warm-up and return timing statistics."""
    for _ in range(warmup_runs):
        function()

    timings = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        elapsed = time.perf_counter_ns() - start
        timings.append(elapsed / 1_000_000)

    return {
        "result": result,
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "std": statistics.stdev(timings)
    }


def benchmark_eviction_strategies(
    cache,
    attention_scores,
    semantic_scores,
    recency_scores,
    last_access_times,
    retention_budget,
    important_token_indices=None,
    protected_recent_tokens=2,
    repeats=100,
    warmup_runs=5
):
    """Compare FIFO, LRU and DSAA using repeated measurements."""

    if not cache:
        raise ValueError("Cache cannot be empty")

    if repeats < 2 or warmup_runs < 0:
        raise ValueError("Invalid benchmark run counts")

    num_tokens = cache[0][0].shape[-2]

    if not 0 <= retention_budget <= num_tokens:
        raise ValueError("Invalid retention budget")

    if protected_recent_tokens < 0:
        raise ValueError("Protected token count cannot be negative")

    if protected_recent_tokens > retention_budget:
        raise ValueError("Protected tokens cannot exceed retention budget")

    vectors = [
        attention_scores,
        semantic_scores,
        recency_scores,
        last_access_times
    ]

    if any(len(vector) != num_tokens for vector in vectors):
        raise ValueError("All vectors must match cache length")

    important = set(important_token_indices or [])

    if any(i < 0 or i >= num_tokens for i in important):
        raise ValueError("Invalid important token index")

    if not all(isinstance(x, (int, float)) for x in last_access_times):
        raise ValueError("Access timestamps must be numeric")

    def select_fifo():
        return list(range(num_tokens - retention_budget, num_tokens))

    def select_lru():
        indices = sorted(
            range(num_tokens),
            key=lambda i: (float(last_access_times[i]), i),
            reverse=True
        )[:retention_budget]

        return sorted(indices)

    def select_dsaa(combined_scores):
        protected = list(
            range(num_tokens - protected_recent_tokens, num_tokens)
        )

        protected_set = set(protected)

        candidates = [
            i for i in range(num_tokens)
            if i not in protected_set
        ]

        candidates.sort(
            key=lambda i: (combined_scores[i].item(), i),
            reverse=True
        )

        selected = (
            protected +
            candidates[:retention_budget - protected_recent_tokens]
        )

        return sorted(selected)

    def calculate_scores():
        return calculate_combined_score(
            attention_scores,
            semantic_scores,
            recency_scores
        )

    # Measure DSAA score computation independently.
    score_metrics = _measure(
        calculate_scores, repeats, warmup_runs
    )

    combined_scores = score_metrics["result"]

    # Measure token selection independently.
    fifo_metrics = _measure(
        select_fifo, repeats, warmup_runs
    )

    lru_metrics = _measure(
        select_lru, repeats, warmup_runs
    )

    dsaa_metrics = _measure(
        lambda: select_dsaa(combined_scores),
        repeats,
        warmup_runs
    )

    # Measure the complete DSAA decision process.
    dsaa_total_metrics = _measure(
        lambda: select_dsaa(calculate_scores()),
        repeats,
        warmup_runs
    )

    strategies = {
        "FIFO": fifo_metrics,
        "LRU": lru_metrics,
        "DSAA": dsaa_metrics
    }

    original_bytes = get_cache_size_bytes(cache)
    results = {}

    for name, metrics in strategies.items():

        retained_indices = metrics["result"]

        optimized_cache = retain_kv_cache_tokens(
            cache, retained_indices
        )

        optimized_bytes = get_cache_size_bytes(optimized_cache)
        saved_bytes = original_bytes - optimized_bytes

        important_retained = len(
            set(retained_indices) & important
        )

        retention_rate = (
            important_retained / len(important) * 100
            if important else None
        )

        results[name] = {
            "retained_indices": retained_indices,
            "retained_tokens": len(retained_indices),
            "evicted_tokens": num_tokens - len(retained_indices),
            "original_bytes": original_bytes,
            "optimized_bytes": optimized_bytes,
            "memory_saved_bytes": saved_bytes,
            "memory_saved_percentage": (
                saved_bytes / original_bytes * 100
                if original_bytes else 0
            ),
            "important_tokens_retained": important_retained,
            "important_token_retention_percentage": retention_rate,
            "selection_time_ms": metrics["median"],
            "mean_time_ms": metrics["mean"],
            "std_time_ms": metrics["std"],
            "benchmark_runs": repeats
        }

    results["DSAA"]["score_computation_time_ms"] = (
        score_metrics["median"]
    )

    results["DSAA"]["total_decision_time_ms"] = (
        dsaa_total_metrics["median"]
    )

    return results
