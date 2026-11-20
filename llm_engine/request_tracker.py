"""
Request tracking and lifecycle management for DSAA continuous batching.

Module Owner: Person 1 — LLM & KV Cache Engineer.
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from .exceptions import SchedulerError
from .inference_engine import InferenceRequest, GenerationState, InferenceResult


class RequestStatus(str, Enum):
    """
    Lifecycle states for an inference request in the scheduler.
    """
    PENDING = "PENDING"          # Created but not yet queued
    QUEUED = "QUEUED"            # Enqueued, waiting for available batch capacity
    PREFILLING = "PREFILLING"    # Actively running prompt prefill
    DECODING = "DECODING"        # Actively generating tokens step-by-step
    COMPLETED = "COMPLETED"      # Finished generation successfully
    FAILED = "FAILED"            # Encountered an unrecoverable error during execution
    CANCELLED = "CANCELLED"      # Explicitly cancelled by caller before or during execution


@dataclass
class TrackedRequest:
    """
    Encapsulates an InferenceRequest with execution metadata, state transitions,
    and thread-safe cancellation controls.
    """
    request_id: str
    request: InferenceRequest
    priority: int = 0
    status: RequestStatus = RequestStatus.PENDING
    state: Optional[GenerationState] = None
    result: Optional[InferenceResult] = None
    error_message: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    completion_event: threading.Event = field(default_factory=threading.Event)
    stream_queue: Optional[Any] = None
    streamed_text: str = ""

    @property
    def is_terminal(self) -> bool:
        """Return True if the request is in a final state."""
        return self.status in {
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
        }

    def transition_to(self, new_status: RequestStatus, error_msg: Optional[str] = None) -> None:
        """Safely transition request state and record timestamps."""
        if self.is_terminal and new_status != self.status:
            raise SchedulerError(
                f"Cannot transition request '{self.request_id}' from terminal state "
                f"'{self.status.value}' to '{new_status.value}'."
            )

        self.status = new_status
        if error_msg:
            self.error_message = error_msg

        now = time.time()
        if new_status in {RequestStatus.PREFILLING, RequestStatus.DECODING} and self.started_at is None:
            self.started_at = now

        if self.is_terminal:
            self.completed_at = now
            self.completion_event.set()
            self.cleanup()

    def cancel(self) -> bool:
        """Cancel the request if not already finished."""
        if self.is_terminal:
            return False
        self.transition_to(RequestStatus.CANCELLED, error_msg="Request was cancelled by user.")
        return True

    def cleanup(self) -> None:
        """Release underlying KV Cache tensors and generation state resources."""
        if self.state is not None:
            self.state.cleanup()
            self.state = None
