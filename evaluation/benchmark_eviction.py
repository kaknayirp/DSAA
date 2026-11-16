
import time

from dsaa_algorithm.combined_score import calculate_combined_score
from evaluation.simulate_cache import get_cache_size_bytes


def benchmark_eviction_strategies(
    cache,
    attention_scores,
    semantic_scores,
    recency_scores,
    last_access_times,
    retention_budget,
    important_token_indices=None,
    protected_recent_tokens=2
):
    """
    Compare FIFO, LRU and DSAA using the same retention budget.
    """

    if not cache:
        raise ValueError("Cache cannot be empty")

    num_tokens = cache[0][0].shape[-2]

    if not 0 <= retention_budget <= num_tokens:
        raise ValueError("Invalid retention budget")

    if not 0 <= protected_recent_tokens:
        raise ValueError("Protected token count cannot be negative")

    score_vectors = [
        attention_scores,
        semantic_scores,
        recency_scores,
        last_access_times
    ]

    if any(len(scores) != num_tokens for scores in score_vectors):
        raise ValueError("All score vectors must match cache length")

    important = set(important_token_indices or [])

    if any(i < 0 or i >= num_tokens for i in important):
        raise ValueError("Invalid important token index")

    # Calculate combined DSAA importance.
    combined_scores = calculate_combined_score(
        attention_scores,
        semantic_scores,
        recency_scores
    )

    # FIFO: retain the newest inserted tokens.
    start = num_tokens - retention_budget

    start_time = time.perf_counter()
    fifo_indices = list(range(start, num_tokens))
    fifo_time = time.perf_counter() - start_time

    # LRU: retain tokens with the most recent access timestamps.
    start_time = time.perf_counter()

    lru_indices = sorted(
        range(num_tokens),
        key=lambda i: (float(last_access_times[i]), i),
        reverse=True
    )[:retention_budget]

    lru_indices.sort()
    lru_time = time.perf_counter() - start_time

    # DSAA: protect recent tokens, then retain the highest
    # combined-importance tokens within the remaining budget.
    start_time = time.perf_counter()

    protected_count = min(
        protected_recent_tokens, retention_budget
    )

    protected = list(
        range(num_tokens - protected_count, num_tokens)
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

    dsaa_indices = sorted(
        protected + candidates[:retention_budget - protected_count]
    )

    dsaa_time = time.perf_counter() - start_time

    original_bytes = get_cache_size_bytes(cache)

    strategies = {
        "FIFO": (fifo_indices, fifo_time),
        "LRU": (lru_indices, lru_time),
        "DSAA": (dsaa_indices, dsaa_time)
    }

    results = {}

    for name, (indices, elapsed) in strategies.items():

        retained_bytes = (
            original_bytes * len(indices) / num_tokens
            if num_tokens else 0
        )

        saved_bytes = original_bytes - retained_bytes

        important_retained = len(set(indices) & important)

        retention_rate = (
            important_retained / len(important) * 100
            if important else None
        )

        results[name] = {
            "retained_indices": indices,
            "retained_tokens": len(indices),
            "evicted_tokens": num_tokens - len(indices),
            "original_bytes": original_bytes,
            "optimized_bytes": int(retained_bytes),
            "memory_saved_bytes": int(saved_bytes),
            "memory_saved_percentage": (
                saved_bytes / original_bytes * 100
                if original_bytes else 0
            ),
            "important_tokens_retained": important_retained,
            "important_token_retention_percentage": retention_rate,
            "selection_time_ms": elapsed * 1000
        }

    return results
