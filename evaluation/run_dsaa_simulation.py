
from dsaa_algorithm.combined_score import calculate_combined_score
from dsaa_algorithm.retention_policy import select_tokens_to_retain
from evaluation.simulate_cache import evaluate_cache_reduction


def run_dsaa_simulation(
    cache,
    attention_scores,
    semantic_scores,
    recency_scores,
    memory_pressure,
    protected_recent_tokens=4
):
    """
    Run the complete DSAA scoring, retention and
    synthetic KV-cache reduction pipeline.
    """

    if not cache:
        raise ValueError("Cache cannot be empty")

    sequence_length = cache[0][0].shape[-2]

    if any(
        len(scores) != sequence_length
        for scores in (
            attention_scores,
            semantic_scores,
            recency_scores
        )
    ):
        raise ValueError(
            "Score vectors must match cache sequence length"
        )

    combined_scores = calculate_combined_score(
        attention_scores,
        semantic_scores,
        recency_scores
    )

    retained_indices = select_tokens_to_retain(
        combined_scores,
        memory_pressure=memory_pressure,
        protected_recent_tokens=protected_recent_tokens
    )

    results = evaluate_cache_reduction(
        cache,
        retained_indices
    )

    results["retained_indices"] = retained_indices
    results["combined_scores"] = combined_scores
    results["memory_pressure"] = memory_pressure

    return results
