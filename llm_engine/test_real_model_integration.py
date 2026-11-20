"""Opt-in integration check against locally cached Qwen model weights.

Set DSAA_RUN_REAL_MODEL_TESTS=1 to run. The test never downloads model files.
"""

import json
import os
import unittest

from llm_engine.api_app import create_app
from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.inference_engine import InferenceEngine
from llm_engine.model_loader import ModelLoader
from llm_engine.scheduler import RequestScheduler
from llm_engine.scheduler_config import SchedulerConfig
from llm_engine.server_config import ServerConfig


@unittest.skipUnless(
    os.environ.get("DSAA_RUN_REAL_MODEL_TESTS") == "1",
    "Set DSAA_RUN_REAL_MODEL_TESTS=1 to run the local-weight integration test.",
)
class TestRealQwenInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = ModelConfig(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            extra_model_kwargs={"local_files_only": True},
            extra_tokenizer_kwargs={"local_files_only": True},
        )
        cls.loader = ModelLoader(config)
        cls.artifacts = cls.loader.load()
        cls.engine = InferenceEngine(model_loader=cls.loader)

    @classmethod
    def tearDownClass(cls):
        cls.loader.unload()

    def test_real_blocking_and_streaming_generation(self):
        direct = self.engine.generate(
            "Reply with one short sentence about cache memory.",
            GenerationConfig(max_new_tokens=4),
        )
        self.assertGreater(len(direct.generated_token_ids), 0)
        self.assertEqual(
            direct.decoded_text,
            self.artifacts.tokenizer.decode(direct.generated_token_ids, skip_special_tokens=True),
        )
        self.assertIsNotNone(direct.final_cache_report)
        self.assertGreater(direct.final_cache_report.total_bytes, 0)
        self.assertGreater(direct.total_time_seconds, 0)

        scheduler = RequestScheduler(self.engine, SchedulerConfig(max_batch_size=1))
        app = create_app(scheduler, ServerConfig(), start_background_worker=False)
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            blocking = client.post(
                "/generate",
                json={"prompt": "Say hello briefly.", "max_new_tokens": 3},
            )
            self.assertEqual(blocking.status_code, 200)
            blocking_result = blocking.json()
            self.assertGreater(blocking_result["num_generated_tokens"], 0)
            self.assertTrue(blocking_result["generated_text"])
            self.assertNotIn("decoded_text_length", blocking_result["generated_text"])
            self.assertIsNotNone(blocking_result["kv_cache_summary"])

            streamed = client.post(
                "/generate",
                json={"prompt": "Say hello briefly.", "max_new_tokens": 3, "stream": True},
            )
            self.assertEqual(streamed.status_code, 200)
            lines = list(streamed.iter_lines())
            events = [
                line[6:]
                for line in lines
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            payloads = [json.loads(event) for event in events]
            tokens = [event for event in payloads if "token_id" in event]
            completion = next(event for event in payloads if "finish_reason" in event)
            self.assertEqual(completion["num_generated_tokens"], len(tokens))
            self.assertGreater(len(tokens), 0)
            self.assertTrue(all(token["text"] for token in tokens))
            self.assertIn("data: [DONE]", lines)


if __name__ == "__main__":
    unittest.main()
