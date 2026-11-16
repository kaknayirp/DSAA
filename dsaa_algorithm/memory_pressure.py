
def calculate_memory_pressure(current_tokens, max_capacity):
    """
    Calculate KV cache memory pressure.

    Returns a value between 0 and 1.
    0 = no memory used
    1 = capacity reached or exceeded
    """

    if current_tokens < 0:
        raise ValueError("Current tokens cannot be negative")

    if max_capacity <= 0:
        raise ValueError("Maximum capacity must be positive")

    pressure = current_tokens / max_capacity

    return min(pressure, 1.0)
