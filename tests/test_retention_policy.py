
from dsaa_algorithm.retention_policy import (
    select_tokens_to_retain
)


def test_no_eviction_under_threshold():
    scores = [0.1, 0.2, 0.8, 0.9]

    retained = select_tokens_to_retain(
        scores,
        memory_pressure=0.5
    )

    assert retained == [0, 1, 2, 3]


def test_low_importance_tokens_evicted():
    scores = [0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4]

    retained = select_tokens_to_retain(
        scores,
        memory_pressure=1.0,
        protected_recent_tokens=2
    )

    assert 0 not in retained
    assert 1 not in retained

    # Recent tokens must be protected.
    assert 6 in retained
    assert 7 in retained

    print("Eviction policy test passed!")
    print("Retained token indices:", retained)


def test_empty_input():
    assert select_tokens_to_retain([], 0.9) == []
    print("Empty input test passed!")


if __name__ == "__main__":
    test_no_eviction_under_threshold()
    test_low_importance_tokens_evicted()
    test_empty_input()
