
from dsaa_algorithm.memory_pressure import (
    calculate_memory_pressure
)


def test_memory_pressure():
    assert calculate_memory_pressure(300, 1000) == 0.3
    assert calculate_memory_pressure(700, 1000) == 0.7
    assert calculate_memory_pressure(950, 1000) == 0.95
    assert calculate_memory_pressure(1000, 1000) == 1.0

    print("Memory pressure test passed!")


def test_capacity_exceeded():
    assert calculate_memory_pressure(1200, 1000) == 1.0
    print("Capacity exceeded test passed!")


def test_invalid_capacity():
    try:
        calculate_memory_pressure(100, 0)
        assert False, "Expected ValueError"
    except ValueError:
        print("Invalid capacity test passed!")


if __name__ == "__main__":
    test_memory_pressure()
    test_capacity_exceeded()
    test_invalid_capacity()
