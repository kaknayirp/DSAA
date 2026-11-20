"""
Continuous Batching Request Scheduler for DSAA.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Implements iteration-level continuous batching:
- Dynamically admits pending requests into active decoding batches as capacity opens.
- Processes prompt prefill for newly admitted requests.
- Executes single-token decode steps for all active requests using independent KV Caches.
- Immediately evicts finished requests without blocking other active requests.
- Ensures strict request isolation, thread safety, and KV Cache cleanup.
"""

import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .exceptions import (
    SchedulerError,
    QueueFullError,
    RequestNotFoundError,
    RequestCancelledError,
    RequestTimeoutError,
    InferenceEngineError,
    InvalidInputError,
)
from .scheduler_config import SchedulerConfig, SchedulingPolicy
from .request_tracker import RequestStatus, TrackedRequest
from .inference_engine import (
    InferenceEngine,
    InferenceRequest,
    GenerationState,
    InferenceResult,
)
from .generation_config import GenerationConfig
from .kv_extractor import extract_kv_cache

logger = logging.getLogger(__name__)


@dataclass
class BatchStepReport:
    """Summary of operations performed in a single scheduler iteration step."""
    step_number: int
    admitted_requests: List[str] = field(default_factory=list)
    completed_requests: List[str] = field(default_factory=list)
    failed_requests: List[str] = field(default_factory=list)
    cancelled_requests: List[str] = field(default_factory=list)
    active_requests: List[str] = field(default_factory=list)
    queued_count: int = 0
    step_latency_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"[Step {self.step_number:4d}] Active: {len(self.active_requests)} | "
            f"Admitted: {len(self.admitted_requests)} | "
            f"Completed: {len(self.completed_requests)} | "
            f"Queued: {self.queued_count} | "
            f"Time: {self.step_latency_seconds * 1000:.2f}ms"
        )


class RequestScheduler:
    """
    Manages request queues, admits requests into active batches, executes
    iteration-level continuous batching, and manages independent KV Cache states.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.engine = inference_engine
        self.config = config or SchedulerConfig()

        self._queue: List[TrackedRequest] = []
        self._active_requests: Dict[str, TrackedRequest] = {}
        self._all_requests: Dict[str, TrackedRequest] = {}

        self._lock = threading.RLock()
        self._step_counter = 0

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active_requests)

    def submit(
        self,
        prompt_or_request: Union[InferenceRequest, str, List[int]],
        generation_config: Optional[GenerationConfig] = None,
        priority: int = 0,
        stream_queue: Optional[Any] = None,
    ) -> str:
        """
        Submit a new inference request to the scheduler queue.

        Args:
            prompt_or_request: Raw prompt string, list of token IDs, or InferenceRequest.
            generation_config: Optional GenerationConfig (used if prompt is string/list).
            priority: Scheduling priority (higher value = earlier admission in Priority policy).

        Returns:
            Unique request_id string.
        """
        with self._lock:
            if len(self._queue) >= self.config.max_queued_requests:
                raise QueueFullError(
                    f"Scheduler queue is full ({len(self._queue)}/{self.config.max_queued_requests} requests). "
                    "Cannot accept new requests."
                )

            # Build standard InferenceRequest
            if isinstance(prompt_or_request, InferenceRequest):
                req = prompt_or_request
                if generation_config is not None:
                    req.generation_config = generation_config
            elif isinstance(prompt_or_request, str):
                if not prompt_or_request.strip():
                    raise InvalidInputError("Prompt string cannot be empty.")
                if self.engine.tokenizer is None:
                    raise InferenceEngineError("Tokenizer required to encode string prompt.")
                encoded = self.engine.tokenizer(prompt_or_request, return_tensors="pt")
                req = InferenceRequest(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    generation_config=generation_config or GenerationConfig(),
                    prompt_text=prompt_or_request,
                )
            elif isinstance(prompt_or_request, list):
                if len(prompt_or_request) == 0:
                    raise InvalidInputError("Token IDs list cannot be empty.")
                req = InferenceRequest(
                    input_ids=prompt_or_request,
                    generation_config=generation_config or GenerationConfig(),
                )
            else:
                raise InvalidInputError(f"Unsupported input type: {type(prompt_or_request)}")

            # Configure default stop tokens from tokenizer if not explicitly configured
            cfg = req.generation_config
            if not cfg.stop_token_set and self.engine.tokenizer is not None:
                eos_id = getattr(self.engine.tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    cfg.stop_token_set.add(int(eos_id))

            tracked = TrackedRequest(
                request_id=req.request_id,
                request=req,
                priority=priority,
                status=RequestStatus.QUEUED,
                submitted_at=time.time(),
                stream_queue=stream_queue,
            )

            # Insert into queue
            if self.config.scheduling_policy == SchedulingPolicy.PRIORITY.value:
                # Higher priority at front
                insert_idx = 0
                while insert_idx < len(self._queue) and self._queue[insert_idx].priority >= priority:
                    insert_idx += 1
                self._queue.insert(insert_idx, tracked)
            else:
                # FIFO / FCFS
                self._queue.append(tracked)

            self._all_requests[tracked.request_id] = tracked
            logger.debug("Enqueued request '%s' with priority %d.", tracked.request_id, priority)
            return tracked.request_id

    def cancel(self, request_id: str) -> bool:
        """
        Cancel a request by ID, whether queued or actively decoding.

        Returns:
            True if cancelled, False if request is not found or already finished.
        """
        with self._lock:
            tracked = self._all_requests.get(request_id)
            if tracked is None:
                return False

            if tracked.is_terminal:
                return False

            # If queued, remove from waiting queue
            if tracked in self._queue:
                self._queue.remove(tracked)

            # Cancel and release KV Cache
            success = tracked.cancel()
            if tracked.stream_queue is not None:
                tracked.stream_queue.put({"type": "cancelled", "cancelled": True, "error": "Request cancelled."})
            logger.info("Cancelled request '%s'.", request_id)
            return success

    def get_status(self, request_id: str) -> Optional[TrackedRequest]:
        """Retrieve tracking metadata for a request."""
        with self._lock:
            return self._all_requests.get(request_id)

    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        """
        Wait for and retrieve the completed InferenceResult for a request.

        Raises:
            RequestNotFoundError: if request_id does not exist.
            RequestCancelledError: if the request was cancelled.
            InferenceEngineError: if the request failed during execution.
            RequestTimeoutError: if timeout elapsed before completion.
        """
        with self._lock:
            tracked = self._all_requests.get(request_id)
            if tracked is None:
                raise RequestNotFoundError(f"No request found with ID '{request_id}'.")

        # Wait outside lock so other threads can schedule
        finished = tracked.completion_event.wait(timeout=timeout)
        if not finished:
            raise RequestTimeoutError(f"Request '{request_id}' timed out after {timeout} seconds.")

        if tracked.status == RequestStatus.CANCELLED:
            raise RequestCancelledError(f"Request '{request_id}' was cancelled: {tracked.error_message}")
        if tracked.status == RequestStatus.FAILED:
            raise InferenceEngineError(f"Request '{request_id}' failed: {tracked.error_message}")

        return tracked.result

    def step(self) -> BatchStepReport:
        """
        Execute a single scheduling iteration:
        1. Check timeouts for queued and active requests.
        2. Fill available batch capacity by admitting requests from the queue and running prefill.
        3. Execute one decode step for all active decoding requests.
        4. Evict completed, failed, or cancelled requests, freeing batch slots immediately.
        """
        with self._lock:
            self._step_counter += 1
            step_idx = self._step_counter
            t0 = time.time()

            report = BatchStepReport(step_number=step_idx)

            # 1. Check timeouts
            self._check_timeouts(report)

            # 2. Identify requests that are ALREADY in DECODING at the start of this step
            already_decoding = [
                tracked for tracked in self._active_requests.values()
                if tracked.status == RequestStatus.DECODING
            ]

            # 3. Admit new requests up to available batch capacity
            available_slots = min(
                self.config.max_batch_size - len(self._active_requests),
                self.config.max_active_requests - len(self._active_requests),
            )

            while available_slots > 0 and len(self._queue) > 0:
                candidate = self._queue.pop(0)

                # Skip if cancelled while waiting in queue
                if candidate.status == RequestStatus.CANCELLED:
                    report.cancelled_requests.append(candidate.request_id)
                    continue

                # Admit and run prefill
                candidate.transition_to(RequestStatus.PREFILLING)
                self._active_requests[candidate.request_id] = candidate
                report.admitted_requests.append(candidate.request_id)
                available_slots -= 1

                try:
                    state = self.engine.prefill(candidate.request)
                    candidate.state = state

                    if candidate.stream_queue is not None and state.generated_token_ids:
                        first_tok = state.generated_token_ids[-1]
                        candidate.stream_queue.put({
                            "type": "token",
                            "token_id": first_tok,
                            "text": self._decode_stream_delta(candidate),
                        })

                    if state.is_finished:
                        self._finalize_request(candidate)
                        report.completed_requests.append(candidate.request_id)
                    else:
                        candidate.transition_to(RequestStatus.DECODING)
                except Exception as exc:
                    logger.error("Prefill failed for request '%s': %s", candidate.request_id, exc)
                    candidate.transition_to(RequestStatus.FAILED, error_msg=str(exc))
                    if candidate.stream_queue is not None:
                        candidate.stream_queue.put({"type": "error", "error": str(exc)})
                    report.failed_requests.append(candidate.request_id)

            # 4. Decode step for requests that were already decoding at step start
            for tracked in already_decoding:
                if tracked.status != RequestStatus.DECODING:
                    continue

                # Check if request was cancelled while in batch
                if tracked.status == RequestStatus.CANCELLED:
                    report.cancelled_requests.append(tracked.request_id)
                    continue

                try:
                    token, is_finished = self.engine.decode_step(tracked.state)

                    if tracked.stream_queue is not None:
                        tracked.stream_queue.put({
                            "type": "token",
                            "token_id": token,
                            "text": self._decode_stream_delta(tracked),
                        })

                    if is_finished:
                        self._finalize_request(tracked)
                        report.completed_requests.append(tracked.request_id)
                except Exception as exc:
                    logger.error("Decode step failed for request '%s': %s", tracked.request_id, exc)
                    tracked.transition_to(RequestStatus.FAILED, error_msg=str(exc))
                    if tracked.stream_queue is not None:
                        tracked.stream_queue.put({"type": "error", "error": str(exc)})
                    report.failed_requests.append(tracked.request_id)

            # 4. Remove all terminal requests from active batch (frees slots for next iteration!)
            terminal_ids = [
                req_id for req_id, tracked in self._active_requests.items()
                if tracked.is_terminal
            ]
            for req_id in terminal_ids:
                del self._active_requests[req_id]

            report.active_requests = list(self._active_requests.keys())
            report.queued_count = len(self._queue)
            report.step_latency_seconds = time.time() - t0
            return report

    def run_until_complete(
        self,
        max_steps: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, InferenceResult]:
        """
        Execute scheduler steps in a loop until all queued and active requests finish.

        Returns:
            Dictionary mapping request_id -> InferenceResult for all completed requests.
        """
        t_start = time.time()
        step_count = 0

        while True:
            with self._lock:
                if len(self._queue) == 0 and len(self._active_requests) == 0:
                    break

            if max_steps is not None and step_count >= max_steps:
                break
            if timeout is not None and (time.time() - t_start) >= timeout:
                break

            self.step()
            step_count += 1

        results = {}
        with self._lock:
            for req_id, tracked in self._all_requests.items():
                if tracked.result is not None:
                    results[req_id] = tracked.result
        return results

    def stats(self) -> Dict[str, Any]:
        """Return operational snapshot of the scheduler."""
        with self._lock:
            status_counts = {status.value: 0 for status in RequestStatus}
            for tracked in self._all_requests.values():
                status_counts[tracked.status.value] += 1

            return {
                "step_count": self._step_counter,
                "queued_count": len(self._queue),
                "active_count": len(self._active_requests),
                "total_submitted": len(self._all_requests),
                "status_breakdown": status_counts,
                "max_batch_size": self.config.max_batch_size,
                "scheduling_policy": self.config.scheduling_policy,
            }

    def clear(self) -> None:
        """Cancel all queued and active requests, freeing all KV Cache memory."""
        with self._lock:
            for tracked in self._queue:
                tracked.cancel()
            self._queue.clear()

            for tracked in list(self._active_requests.values()):
                tracked.cancel()
            self._active_requests.clear()

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _check_timeouts(self, report: BatchStepReport) -> None:
        """Identify and fail any requests exceeding request_timeout_seconds."""
        if self.config.request_timeout_seconds is None:
            return

        now = time.time()
        timeout = self.config.request_timeout_seconds

        # Check queue
        remaining_queue = []
        for tracked in self._queue:
            if now - tracked.submitted_at > timeout:
                tracked.transition_to(RequestStatus.FAILED, error_msg=f"Request timed out in queue after {timeout}s.")
                report.failed_requests.append(tracked.request_id)
            else:
                remaining_queue.append(tracked)
        self._queue = remaining_queue

        # Check active requests
        for tracked in list(self._active_requests.values()):
            if now - tracked.submitted_at > timeout:
                tracked.transition_to(RequestStatus.FAILED, error_msg=f"Request timed out during execution after {timeout}s.")
                report.failed_requests.append(tracked.request_id)

    def _finalize_request(self, tracked: TrackedRequest) -> None:
        """Build final InferenceResult, attach KV Cache report, and mark COMPLETED."""
        state = tracked.state
        req = tracked.request
        total_time = (time.time() - tracked.started_at) if tracked.started_at else 0.0
        decode_time = sum(state.decode_latencies_sec) if state else 0.0
        num_generated = len(state.generated_token_ids) if state else 0
        tps = (num_generated / total_time) if total_time > 0 else 0.0

        # Extract KV Cache report using Prompt 2's extract_kv_cache utility
        cache_report = None
        if state is not None and state.past_key_values is not None and req.generation_config.return_cache_report:
            try:
                cache_report = extract_kv_cache(state.past_key_values)
            except Exception as exc:
                logger.warning("Could not extract KV Cache report for request '%s': %s", tracked.request_id, exc)

        prompt_token_ids = self.engine._to_token_list(req.input_ids)
        gen_token_ids = list(state.generated_token_ids) if state else []
        output_token_ids = prompt_token_ids + gen_token_ids

        decoded_text = None
        if self.engine.tokenizer is not None and hasattr(self.engine.tokenizer, "decode") and gen_token_ids:
            try:
                decoded_text = self.engine.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
            except Exception:
                pass

        result = InferenceResult(
            request_id=tracked.request_id,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=gen_token_ids,
            output_token_ids=output_token_ids,
            decoded_text=decoded_text,
            finish_reason=state.finish_reason if state else "completed",
            num_prompt_tokens=len(prompt_token_ids),
            num_generated_tokens=num_generated,
            total_time_seconds=total_time,
            prefill_time_seconds=state.prefill_latency_sec if state else 0.0,
            decode_time_seconds=decode_time,
            tokens_per_second=tps,
            final_cache_report=cache_report,
        )

        tracked.result = result
        tracked.transition_to(RequestStatus.COMPLETED)
        if tracked.stream_queue is not None:
            tracked.stream_queue.put({"type": "complete", "result": result})

    def _decode_stream_delta(self, tracked: TrackedRequest) -> str:
        """Decode the cumulative generated prefix and return newly visible text."""
        if self.engine.tokenizer is not None and hasattr(self.engine.tokenizer, "decode"):
            try:
                try:
                    decoded = self.engine.tokenizer.decode(
                        tracked.state.generated_token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    decoded = self.engine.tokenizer.decode(
                        tracked.state.generated_token_ids,
                        skip_special_tokens=True,
                    )
                previous = tracked.streamed_text
                tracked.streamed_text = decoded
                return decoded[len(previous):] if decoded.startswith(previous) else decoded
            except Exception:
                pass
        return ""
