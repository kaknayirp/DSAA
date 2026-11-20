"""
Configuration parameters for the DSAA Request Scheduler and Continuous Batching.

Module Owner: Person 1 — LLM & KV Cache Engineer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .exceptions import GenerationConfigError


class SchedulingPolicy(str, Enum):
    """Supported request scheduling policies."""
    FCFS = "fcfs"          # First-Come First-Served (FIFO)
    PRIORITY = "priority"  # Higher priority requests admitted first


@dataclass
class SchedulerConfig:
    """
    Configuration parameters controlling batch sizing, request queuing,
    scheduling policies, and timeout limits.
    """
    max_batch_size: int = 4
    max_active_requests: int = 8
    max_queued_requests: int = 64
    scheduling_policy: str = SchedulingPolicy.FCFS.value
    request_timeout_seconds: Optional[float] = None
    max_sequence_length: int = 4096

    def __post_init__(self) -> None:
        """Validate scheduler configuration parameters."""
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise GenerationConfigError(
                f"max_batch_size must be a positive integer, got {self.max_batch_size}"
            )

        if not isinstance(self.max_active_requests, int) or self.max_active_requests <= 0:
            raise GenerationConfigError(
                f"max_active_requests must be a positive integer, got {self.max_active_requests}"
            )

        if self.max_active_requests < self.max_batch_size:
            raise GenerationConfigError(
                f"max_active_requests ({self.max_active_requests}) cannot be smaller than max_batch_size ({self.max_batch_size})"
            )

        if not isinstance(self.max_queued_requests, int) or self.max_queued_requests <= 0:
            raise GenerationConfigError(
                f"max_queued_requests must be a positive integer, got {self.max_queued_requests}"
            )

        policy_lower = str(self.scheduling_policy).lower().strip()
        valid_policies = {p.value for p in SchedulingPolicy}
        if policy_lower not in valid_policies:
            raise GenerationConfigError(
                f"Invalid scheduling_policy '{self.scheduling_policy}'. Supported: {sorted(list(valid_policies))}"
            )
        self.scheduling_policy = policy_lower

        if self.request_timeout_seconds is not None:
            if not isinstance(self.request_timeout_seconds, (int, float)) or self.request_timeout_seconds <= 0:
                raise GenerationConfigError(
                    f"request_timeout_seconds must be a positive float, got {self.request_timeout_seconds}"
                )

        if not isinstance(self.max_sequence_length, int) or self.max_sequence_length <= 0:
            raise GenerationConfigError(
                f"max_sequence_length must be a positive integer, got {self.max_sequence_length}"
            )
