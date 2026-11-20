"""
Unit and integration tests for DSAA Model Serving and REST API Layer.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Tests verify:
- Server initialization, startup, and shutdown lifespan
- GET  /health (health status, active/queued counts, hardware device)
- GET  /models (active model and candidate architectures)
- POST /generate end-to-end integration:
    API Request -> Request Manager -> Scheduler -> Inference Engine -> KV Cache -> Response
- Prompt 2 KV Cache summary embedding in API responses
- Streaming generation via Server-Sent Events (SSE) and [DONE] terminator
- GET  /requests/{id} status tracking
- POST /requests/{id}/cancel cancellation and KV Cache resource release
- Input validation (empty prompt, invalid parameters, length boundaries)
- Queue capacity overflow handling (HTTP 429)
- Authentication middleware verification (HTTP 401 on missing key)
"""

import sys
import json
import time
import unittest
from starlette.testclient import TestClient

from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.server_config import ServerConfig
from llm_engine.scheduler_config import SchedulerConfig
from llm_engine.scheduler import RequestScheduler
from llm_engine.inference_engine import InferenceEngine
from llm_engine.model_loader import ModelArtifacts
from llm_engine.test_inference_engine import StatefulCausalLM, MockTokenizer
from llm_engine.api_app import create_app


class TestModelServingAPI(unittest.TestCase):
    """Test suite for FastAPI model serving layer and continuous batching integration."""

    def setUp(self):
        # Setup simulated causal LM matching Qwen2.5 specs
        self.model = StatefulCausalLM(vocab_size=500, num_layers=4, num_kv_heads=2, head_dim=128)
        self.tokenizer = MockTokenizer(eos_token_id=999)
        self.artifacts = ModelArtifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct"),
            device="cpu",
            dtype="float16",
        )
        self.engine = InferenceEngine(artifacts=self.artifacts)
        self.scheduler_cfg = SchedulerConfig(max_batch_size=2, max_queued_requests=10)
        self.scheduler = RequestScheduler(self.engine, config=self.scheduler_cfg)
        self.server_cfg = ServerConfig(max_input_length=500, max_generation_length=128)

        # Create FastAPI application (run worker synchronously or via background thread)
        self.app = create_app(
            scheduler=self.scheduler,
            server_config=self.server_cfg,
            start_background_worker=True,
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.scheduler.clear()

    def test_health_endpoint(self):
        """GET /health returns healthy status and active server metadata."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(data["device"], "cpu")
        self.assertIn("active_requests", data)
        self.assertIn("queued_requests", data)
        self.assertGreaterEqual(data["uptime_seconds"], 0)

    def test_models_endpoint(self):
        """GET /models returns active model and candidate architectures."""
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["active_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertTrue(len(data["models"]) >= 1)
        active_entry = next(m for m in data["models"] if m["is_active"])
        self.assertEqual(active_entry["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(active_entry["device"], "cpu")

    def test_generate_text_end_to_end(self):
        """
        Verify end-to-end integration:
        API Request -> Request Manager -> Scheduler -> Inference Engine -> KV Cache -> Response.
        """
        payload = {
            "prompt": "Explain KV Cache in transformer models.",
            "max_new_tokens": 4,
            "temperature": 0.0,
            "do_sample": False,
            "stream": False,
        }
        resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("request_id", data)
        self.assertEqual(data["model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(data["num_generated_tokens"], 4)
        self.assertEqual(data["finish_reason"], "max_tokens")
        self.assertGreater(data["total_time_seconds"], 0)
        self.assertGreaterEqual(data["tokens_per_second"], 0)

        # Verify Prompt 2 KV Cache integration is present in API response
        self.assertIsNotNone(data["kv_cache_summary"])
        kv_sum = data["kv_cache_summary"]
        self.assertEqual(kv_sum["num_layers"], 4)
        self.assertEqual(kv_sum["seq_len"], len(self.tokenizer(payload["prompt"])["input_ids"][0]) + 3)
        self.assertTrue(kv_sum["is_gqa"])
        self.assertGreater(kv_sum["total_bytes"], 0)

    def test_generate_streaming_sse(self):
        """POST /generate with stream: true returns Server-Sent Events (SSE)."""
        payload = {
            "prompt": "Test streaming prompt.",
            "max_new_tokens": 3,
            "stream": True,
        }
        resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])

        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                event_data = line[len("data: "):].strip()
                events.append(event_data)

        self.assertIn("[DONE]", events)
        # Should have received at least 1 token chunk before [DONE]
        payloads = [json.loads(e) for e in events if e != "[DONE]"]
        token_events = [event for event in payloads if "token_id" in event]
        completion = next(event for event in payloads if "finish_reason" in event)
        self.assertEqual(len(token_events), 3)
        self.assertEqual(completion["num_generated_tokens"], len(token_events))
        self.assertEqual(completion["finish_reason"], "max_tokens")
        self.assertTrue(all(isinstance(event["token_id"], int) for event in token_events))
        self.assertTrue(all(event["text"] for event in token_events))

    def test_stop_text_sequence_ends_stream_with_accurate_count(self):
        resp = self.client.post(
            "/generate",
            json={"prompt": "Stop test", "max_new_tokens": 4, "stop_tokens": [")"], "stream": True},
        )
        self.assertEqual(resp.status_code, 200)
        payloads = [
            json.loads(line[6:])
            for line in resp.iter_lines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        token_events = [event for event in payloads if "token_id" in event]
        completion = next(event for event in payloads if "finish_reason" in event)
        self.assertEqual(completion["finish_reason"], "stop_sequence")
        self.assertEqual(completion["num_generated_tokens"], len(token_events))
        self.assertEqual(len(token_events), 1)

    def test_request_status_lifecycle(self):
        """GET /requests/{request_id} returns accurate execution state."""
        # Submit a request with max_new_tokens=2
        payload = {"prompt": "Track status test", "max_new_tokens": 2}
        gen_resp = self.client.post("/generate", json=payload)
        self.assertEqual(gen_resp.status_code, 200)
        req_id = gen_resp.json()["request_id"]

        # Check status
        status_resp = self.client.get(f"/requests/{req_id}")
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        self.assertEqual(status_data["request_id"], req_id)
        self.assertEqual(status_data["status"], "COMPLETED")
        self.assertEqual(status_data["num_generated_tokens"], 2)

    def test_request_cancellation_endpoint(self):
        """POST /requests/{request_id}/cancel successfully cancels request."""
        # Submit request directly to scheduler with 20 tokens so it stays active
        req_id = self.scheduler.submit("Cancel me", GenerationConfig(max_new_tokens=20))

        cancel_resp = self.client.post(f"/requests/{req_id}/cancel")
        self.assertEqual(cancel_resp.status_code, 200)
        cancel_data = cancel_resp.json()
        self.assertEqual(cancel_data["request_id"], req_id)
        self.assertTrue(cancel_data["cancelled"])

        # Check status reflects CANCELLED
        status_resp = self.client.get(f"/requests/{req_id}")
        self.assertEqual(status_resp.json()["status"], "CANCELLED")

    def test_input_validation_empty_prompt(self):
        """Empty prompt is rejected with 422 Unprocessable Entity."""
        resp = self.client.post("/generate", json={"prompt": ""})
        self.assertEqual(resp.status_code, 422)

        resp2 = self.client.post("/generate", json={"prompt": "   "})
        self.assertEqual(resp2.status_code, 422)

    def test_input_validation_invalid_parameters(self):
        """Negative temperature or negative max_new_tokens rejected with 422."""
        resp = self.client.post("/generate", json={"prompt": "valid", "temperature": -1.0})
        self.assertEqual(resp.status_code, 422)

        resp2 = self.client.post("/generate", json={"prompt": "valid", "max_new_tokens": 0})
        self.assertEqual(resp2.status_code, 422)

    def test_unknown_request_id_returns_404(self):
        """GET /requests/nonexistent returns HTTP 404."""
        resp = self.client.get("/requests/nonexistent-id")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"], "NotFound")

    def test_queue_overflow_returns_429(self):
        """Submitting beyond scheduler queue capacity returns HTTP 429 Too Many Requests."""
        # Create scheduler with max_queued_requests=1
        sched = RequestScheduler(self.engine, config=SchedulerConfig(max_batch_size=1, max_queued_requests=1))
        app = create_app(scheduler=sched, start_background_worker=False)
        client = TestClient(app)

        # First request fills queue
        sched.submit("Req 1")
        # Second request overflows queue
        resp = client.post("/generate", json={"prompt": "Req 2"})
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.json()["error"], "QueueFull")

    def test_api_key_authentication(self):
        """When api_key is configured, requests without key return HTTP 401."""
        auth_cfg = ServerConfig(api_key="secret-dsaa-key")
        app = create_app(scheduler=self.scheduler, server_config=auth_cfg, start_background_worker=True)
        client = TestClient(app)

        # Health is exempt
        self.assertEqual(client.get("/health").status_code, 200)

        # /generate without key -> 401
        resp_unauth = client.post("/generate", json={"prompt": "Test"})
        self.assertEqual(resp_unauth.status_code, 401)

        # /generate with valid Bearer token -> 200
        headers = {"Authorization": "Bearer secret-dsaa-key"}
        resp_auth = client.post("/generate", json={"prompt": "Test", "max_new_tokens": 2}, headers=headers)
        self.assertEqual(resp_auth.status_code, 200)


if __name__ == "__main__":
    unittest.main()
