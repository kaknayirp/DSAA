"""
Unit and integration tests for DSAA Continuous Batching Request Scheduler.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Tests verify:
- Scheduler configuration and validation
- Request tracking and state transitions (PENDING, QUEUED, PREFILLING, DECODING, COMPLETED, CANCELLED, FAILED)
- Request queuing and capacity overflow handling
- Continuous batching dynamics:
    * Multiple requests in one batch
    * Requests with different prompt lengths
    * Requests with different generation lengths
    * Early exit of shorter requests and immediate admission of waiting requests
- Independent KV Cache isolation across concurrent requests
- Request cancellation (both queued and actively decoding)
- Failure isolation (one request's error does not affect concurrent requests)
- Thread-safe concurrent submissions
- Integration with Prompt 2's KV Cache inspection utilities (CacheReport)
"""

import sys
import time
import threading
import unittest
from unittest.mock import MagicMock

from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.scheduler_config import SchedulerConfig, SchedulingPolicy
from llm_engine.request_tracker import RequestStatus, TrackedRequest
from llm_engine.exceptions import (
    QueueFullError,
    RequestNotFoundError,
    RequestCancelledError,
    GenerationConfigError,
    InvalidInputError,
)
from llm_engine.inference_engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
)
from llm_engine.scheduler import RequestScheduler, BatchStepReport
from llm_engine.model_loader import ModelArtifacts
from llm_engine.test_inference_engine import StatefulCausalLM, MockTokenizer
from llm_engine.kv_extractor import CacheReport


class TestSchedulerConfig(unittest.TestCase):
    """Test validation of SchedulerConfig parameters."""

    def test_default_config(self):
        cfg = SchedulerConfig()
        self.assertEqual(cfg.max_batch_size, 4)
        self.assertEqual(cfg.max_active_requests, 8)
        self.assertEqual(cfg.max_queued_requests, 64)
        self.assertEqual(cfg.scheduling_policy, "fcfs")
        self.assertIsNone(cfg.request_timeout_seconds)

    def test_valid_custom_config(self):
        cfg = SchedulerConfig(
            max_batch_size=2,
            max_active_requests=4,
            max_queued_requests=10,
            scheduling_policy="priority",
            request_timeout_seconds=30.0,
        )
        self.assertEqual(cfg.max_batch_size, 2)
        self.assertEqual(cfg.scheduling_policy, "priority")
        self.assertEqual(cfg.request_timeout_seconds, 30.0)

    def test_invalid_batch_sizes(self):
        with self.assertRaises(GenerationConfigError):
            SchedulerConfig(max_batch_size=0)
        with self.assertRaises(GenerationConfigError):
            SchedulerConfig(max_active_requests=2, max_batch_size=4)

    def test_invalid_policy(self):
        with self.assertRaises(GenerationConfigError):
            SchedulerConfig(scheduling_policy="round_robin")


class TestRequestTracker(unittest.TestCase):
    """Test TrackedRequest state machine and cancellation."""

    def test_lifecycle_transitions(self):
        req = InferenceRequest(input_ids=[1, 2, 3])
        tracked = TrackedRequest(request_id="test-1", request=req)

        self.assertEqual(tracked.status, RequestStatus.PENDING)
        tracked.transition_to(RequestStatus.QUEUED)
        self.assertEqual(tracked.status, RequestStatus.QUEUED)

        tracked.transition_to(RequestStatus.PREFILLING)
        self.assertEqual(tracked.status, RequestStatus.PREFILLING)
        self.assertIsNotNone(tracked.started_at)

        tracked.transition_to(RequestStatus.DECODING)
        self.assertEqual(tracked.status, RequestStatus.DECODING)

        tracked.transition_to(RequestStatus.COMPLETED)
        self.assertEqual(tracked.status, RequestStatus.COMPLETED)
        self.assertTrue(tracked.is_terminal)
        self.assertTrue(tracked.completion_event.is_set())

    def test_cancellation(self):
        req = InferenceRequest(input_ids=[1, 2, 3])
        tracked = TrackedRequest(request_id="test-cancel", request=req)
        tracked.transition_to(RequestStatus.QUEUED)

        success = tracked.cancel()
        self.assertTrue(success)
        self.assertEqual(tracked.status, RequestStatus.CANCELLED)
        self.assertTrue(tracked.is_terminal)

        # Cannot cancel again once in terminal state
        self.assertFalse(tracked.cancel())


class TestContinuousBatching(unittest.TestCase):
    """
    Core continuous batching test suite:
    Verifies that requests dynamically enter, execute, and leave active batches.
    """

    def setUp(self):
        self.model = StatefulCausalLM(vocab_size=200, num_layers=4, num_kv_heads=2, head_dim=128)
        self.tokenizer = MockTokenizer(eos_token_id=999)
        self.artifacts = ModelArtifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ModelConfig(model_id="mock-qwen"),
            device="cpu",
            dtype="float16",
        )
        self.engine = InferenceEngine(artifacts=self.artifacts)

    def test_continuous_batching_dynamic_entry_and_exit(self):
        """
        Verify the core property of continuous batching:
        - Max batch size is 2.
        - Submit 3 requests:
            Req 1: 3 prompt tokens, max_new_tokens=2 (finishes quickly)
            Req 2: 6 prompt tokens, max_new_tokens=5 (longer generation)
            Req 3: 4 prompt tokens, max_new_tokens=3 (waiting in queue)
        - Step 1: Req 1 and Req 2 admitted (batch full: 2 active). Req 3 queued.
        - Step 2: Req 1 completes after 2 tokens, immediately leaves active batch!
        - Step 3: Req 3 is immediately admitted into the freed slot, while Req 2 continues!
        """
        config = SchedulerConfig(max_batch_size=2, max_active_requests=4, max_queued_requests=10)
        scheduler = RequestScheduler(self.engine, config=config)

        id1 = scheduler.submit([1, 2, 3], generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=999))
        id2 = scheduler.submit([10, 20, 30, 40, 50, 60], generation_config=GenerationConfig(max_new_tokens=5, eos_token_id=999))
        id3 = scheduler.submit([100, 200, 300, 400], generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=999))

        self.assertEqual(scheduler.queued_count, 3)
        self.assertEqual(scheduler.active_count, 0)

        # Iteration 1: Admits Req 1 and Req 2 (batch size 2 reached). Req 3 remains queued.
        rep1 = scheduler.step()
        self.assertEqual(len(rep1.admitted_requests), 2)
        self.assertIn(id1, rep1.admitted_requests)
        self.assertIn(id2, rep1.admitted_requests)
        self.assertEqual(scheduler.queued_count, 1)  # Req 3 still queued
        self.assertEqual(scheduler.active_count, 2)  # Req 1 and Req 2 in batch

        # Iteration 2: Req 1 finishes its 2nd token! Req 1 completes and leaves active batch.
        rep2 = scheduler.step()
        self.assertIn(id1, rep2.completed_requests)
        self.assertEqual(scheduler.get_status(id1).status, RequestStatus.COMPLETED)
        # Req 1 slot was freed, so Req 3 can now be admitted!
        # Iteration 3: Req 3 enters the active batch while Req 2 continues decoding!
        rep3 = scheduler.step()
        self.assertIn(id3, rep3.admitted_requests)
        self.assertIn(id2, rep3.active_requests)

        # Run scheduler until all 3 requests complete
        results = scheduler.run_until_complete()
        self.assertEqual(len(results), 3)
        self.assertIn(id1, results)
        self.assertIn(id2, results)
        self.assertIn(id3, results)

        # Verify token counts
        self.assertEqual(results[id1].num_generated_tokens, 2)
        self.assertEqual(results[id2].num_generated_tokens, 5)
        self.assertEqual(results[id3].num_generated_tokens, 3)

        # Verify all KV Caches were cleaned up
        for req_id in [id1, id2, id3]:
            tracked = scheduler.get_status(req_id)
            self.assertEqual(tracked.status, RequestStatus.COMPLETED)
            self.assertIsNone(tracked.state)  # Cleaned up

    def test_independent_kv_cache_isolation(self):
        """Verify that concurrent requests maintain distinct, non-overlapping KV Caches."""
        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=2))

        id1 = scheduler.submit([1, 2, 3], generation_config=GenerationConfig(max_new_tokens=4, eos_token_id=999))
        id2 = scheduler.submit([10, 20, 30, 40, 50], generation_config=GenerationConfig(max_new_tokens=4, eos_token_id=999))

        scheduler.step()  # Admits both and prefills

        tracked1 = scheduler.get_status(id1)
        tracked2 = scheduler.get_status(id2)

        self.assertIsNotNone(tracked1.state)
        self.assertIsNotNone(tracked2.state)
        # Separate cache instances
        self.assertIsNot(tracked1.state.past_key_values, tracked2.state.past_key_values)
        # Independent prompt lengths in cache
        self.assertEqual(tracked1.state.prompt_length, 3)
        self.assertEqual(tracked2.state.prompt_length, 5)

        scheduler.run_until_complete()

    def test_request_cancellation_queued_and_active(self):
        """Verify cancellation works cleanly in queue and during active decoding."""
        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=1))

        # Req 1 will be active, Req 2 will be queued
        id1 = scheduler.submit([1, 2, 3], generation_config=GenerationConfig(max_new_tokens=10, eos_token_id=999))
        id2 = scheduler.submit([4, 5, 6], generation_config=GenerationConfig(max_new_tokens=10, eos_token_id=999))

        scheduler.step()  # Admits Req 1. Req 2 is in queue.
        self.assertEqual(scheduler.get_status(id1).status, RequestStatus.DECODING)
        self.assertEqual(scheduler.get_status(id2).status, RequestStatus.QUEUED)

        # Cancel queued request
        self.assertTrue(scheduler.cancel(id2))
        self.assertEqual(scheduler.get_status(id2).status, RequestStatus.CANCELLED)

        # Cancel active request
        self.assertTrue(scheduler.cancel(id1))
        self.assertEqual(scheduler.get_status(id1).status, RequestStatus.CANCELLED)

        # Next step cleans them from active pool
        rep = scheduler.step()
        self.assertEqual(scheduler.active_count, 0)
        self.assertEqual(scheduler.queued_count, 0)

    def test_failure_isolation(self):
        """Verify that a failure in one request does not crash other requests in the batch."""
        # Create a model that fails only when given prompt token 666
        class FaultyModel(StatefulCausalLM):
            def __call__(self, input_ids=None, **kwargs):
                flat = []
                if hasattr(input_ids, "tolist"):
                    flat = input_ids.tolist()
                    while flat and isinstance(flat[0], list):
                        flat = flat[0]
                elif isinstance(input_ids, list):
                    flat = input_ids
                    while flat and isinstance(flat[0], list):
                        flat = flat[0]
                if 666 in flat:
                    raise RuntimeError("Simulated GPU Fault for Token 666!")
                return super().__call__(input_ids=input_ids, **kwargs)

        faulty_model = FaultyModel()
        engine = InferenceEngine(artifacts=ModelArtifacts(
            model=faulty_model, tokenizer=self.tokenizer, config=ModelConfig(), device="cpu", dtype="float16"
        ))
        scheduler = RequestScheduler(engine, SchedulerConfig(max_batch_size=2))

        good_id = scheduler.submit([1, 2, 3], generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=999))
        bad_id = scheduler.submit([666], generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=999))

        results = scheduler.run_until_complete()

        # Bad request failed
        self.assertEqual(scheduler.get_status(bad_id).status, RequestStatus.FAILED)
        self.assertIn("Simulated GPU Fault", scheduler.get_status(bad_id).error_message)

        # Good request completed successfully!
        self.assertEqual(scheduler.get_status(good_id).status, RequestStatus.COMPLETED)
        self.assertIn(good_id, results)

    def test_queue_overflow(self):
        """Verify QueueFullError when submitting beyond max_queued_requests."""
        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=1, max_queued_requests=2))

        scheduler.submit([1, 2])
        scheduler.submit([3, 4])
        with self.assertRaises(QueueFullError):
            scheduler.submit([5, 6])

    def test_prompt2_cache_report_integration(self):
        """Verify that completed requests contain valid Prompt 2 CacheReport."""
        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=1))
        req_id = scheduler.submit([1, 2, 3, 4, 5], generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=999))

        scheduler.run_until_complete()
        result = scheduler.get_result(req_id)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.final_cache_report)
        self.assertIsInstance(result.final_cache_report, CacheReport)
        self.assertEqual(result.final_cache_report.num_layers, 4)
        self.assertEqual(result.final_cache_report.num_kv_heads, 2)

    def test_concurrent_submissions(self):
        """Verify thread safety under concurrent request submissions."""
        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=4, max_queued_requests=100))
        submitted_ids = []

        def worker(thread_idx):
            for i in range(5):
                req_id = scheduler.submit([thread_idx, i, 42], generation_config=GenerationConfig(max_new_tokens=2))
                submitted_ids.append(req_id)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(submitted_ids), 20)
        self.assertEqual(scheduler.queued_count, 20)

        # Run scheduler to completion
        results = scheduler.run_until_complete()
        self.assertEqual(len(results), 20)


if __name__ == "__main__":
    unittest.main()
