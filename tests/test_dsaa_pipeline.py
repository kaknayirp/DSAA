
import torch

from evaluation.simulate_cache import create_synthetic_kv_cache
from evaluation.run_dsaa_simulation import run_dsaa_simulation


def test_complete_dsaa_pipeline():

    sequence_length = 10

    cache = create_synthetic_kv_cache(
        num_layers=2,
        batch_size=1,
        num_heads=2,
        seq_len=sequence_length,
        head_dim=4
    )

    attention = torch.tensor([
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0
    ])

    semantic = torch.tensor([
        0.9, 0.8, 0.7, 0.6, 0.5,
        0.4, 0.3, 0.2, 0.1, 1.0
    ])

    recency = torch.linspace(0, 1, sequence_length)

    results = run_dsaa_simulation(
        cache=cache,
        attention_scores=attention,
        semantic_scores=semantic,
        recency_scores=recency,
        memory_pressure=1.0,
        protected_recent_tokens=2
    )

    assert results["retained_tokens"] < sequence_length
    assert results["optimized_bytes"] < results["original_bytes"]
    assert results["memory_saved_percentage"] > 0

    # The two most recent tokens must remain.
    assert 8 in results["retained_indices"]
    assert 9 in results["retained_indices"]

    print("Complete DSAA pipeline test passed!")
    print("Retained indices:", results["retained_indices"])
    print("Original memory:", results["original_bytes"], "bytes")
    print("Optimized memory:", results["optimized_bytes"], "bytes")
    print(
        "Memory saved:",
        round(results["memory_saved_percentage"], 2),
        "%"
    )


if __name__ == "__main__":
    test_complete_dsaa_pipeline()
