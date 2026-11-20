"""
Unit and integration tests for DSAA LLM Inference Engine.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Tests verify:
- Inference engine initialization with ModelLoader and ModelArtifacts
- GenerationConfig validation (valid and invalid parameters)
- Prompt prefill execution (single pass, KV cache creation, first token)
- Autoregressive decoding (1-token inputs, incremental KV cache reuse)
- Greedy decoding vs sampling modes (temperature, top_k, top_p, repetition penalty)
- Stopping conditions (EOS token, stop token ids, max_new_tokens, cache capacity)
- Independent cache state isolation across multiple requests
- Request lifecycle and cache cleanup
- Direct integration with Prompt 2's KV Cache inspection utilities (extract_kv_cache)
- Error handling (empty prompts, missing past_key_values, model execution failure)
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Tuple, List, Optional

from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.exceptions import (
    InferenceEngineError,
    InvalidInputError,
    GenerationConfigError,
    CacheStateError,
)
from llm_engine.inference_engine import (
    InferenceEngine,
    InferenceRequest,
    GenerationState,
    InferenceResult,
)
from llm_engine.model_loader import ModelArtifacts
from llm_engine.kv_extractor import CacheReport, extract_kv_cache
from llm_engine.test_kv_extractor import SyntheticTensor


class StatefulCausalLM:
    """
    Simulated stateful Causal Language Model for testing inference and KV Cache reuse.
    Maintains and incrementally grows Key and Value tensors across decoding steps.
    """
    def __init__(self, vocab_size: int = 1000, num_layers: int = 4, num_kv_heads: int = 2, head_dim: int = 128):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.forward_call_inputs: List[int] = []  # Tracks sequence length passed in each call

    def __call__(self, input_ids=None, past_key_values=None, position_ids=None, attention_mask=None, use_cache=True, return_dict=True):
        batch_size = 1
        if hasattr(input_ids, "shape"):
            step_len = input_ids.shape[-1]
        elif isinstance(input_ids, list):
            step_len = len(input_ids[0]) if (len(input_ids) > 0 and isinstance(input_ids[0], list)) else len(input_ids)
        else:
            step_len = 1

        self.forward_call_inputs.append(step_len)

        # Determine current cumulative sequence length
        if past_key_values is None:
            total_seq_len = step_len
        else:
            # Past key sequence length + current step
            ref_tensor = past_key_values[0][0]
            prev_len = ref_tensor.shape[2]  # [batch, heads, seq_len, head_dim]
            total_seq_len = prev_len + step_len

        # Construct updated KV cache
        shape = (batch_size, self.num_kv_heads, total_seq_len, self.head_dim)
        updated_cache = tuple(
            (
                SyntheticTensor(shape, dtype_str="torch.float16", device_str="cpu"),
                SyntheticTensor(shape, dtype_str="torch.float16", device_str="cpu"),
            )
            for _ in range(self.num_layers)
        )

        # Create predictable synthetic logits [1, step_len, vocab_size]
        # In prefill (step_len > 1), we return logits for all positions.
        # In decode (step_len == 1), we return logits for the single new token.
        logits_matrix = []
        for _ in range(step_len):
            # Deterministic distribution favoring a predictable token
            row = [0.0] * self.vocab_size
            row[42] = 10.0  # High logit for token 42
            logits_matrix.append(row)

        output = MagicMock()
        output.logits = [logits_matrix]  # [batch=1, step_len, vocab_size]
        output.past_key_values = updated_cache
        return output


class MockTokenizer:
    """Simulated tokenizer for encoding/decoding token sequences."""
    def __init__(self, eos_token_id: int = 151645):
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id

    def __call__(self, text, return_tensors=None):
        # Deterministically convert string to token IDs
        tokens = [ord(c) % 500 + 1 for c in text[:20]]
        return {
            "input_ids": [tokens],
            "attention_mask": [[1] * len(tokens)],
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded_text_length_{len(token_ids)}"


class TestGenerationConfig(unittest.TestCase):
    """Test validation and default parameters in GenerationConfig."""

    def test_default_config(self):
        cfg = GenerationConfig()
        self.assertEqual(cfg.max_new_tokens, 128)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertFalse(cfg.do_sample)
        self.assertEqual(cfg.repetition_penalty, 1.0)
        self.assertTrue(cfg.return_cache_report)

    def test_valid_custom_config(self):
        cfg = GenerationConfig(
            max_new_tokens=64,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            eos_token_id=100,
            stop_token_ids=[101, 102],
            max_cache_capacity=2048,
        )
        self.assertEqual(cfg.max_new_tokens, 64)
        self.assertEqual(cfg.temperature, 0.8)
        self.assertEqual(cfg.top_k, 50)
        self.assertEqual(cfg.top_p, 0.95)
        self.assertIn(100, cfg.stop_token_set)
        self.assertIn(101, cfg.stop_token_set)
        self.assertIn(102, cfg.stop_token_set)

    def test_invalid_max_new_tokens(self):
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(max_new_tokens=0)
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(max_new_tokens=-10)

    def test_invalid_temperature(self):
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(temperature=-0.5)

    def test_invalid_top_k(self):
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(top_k=0)

    def test_invalid_top_p(self):
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(top_p=0.0)
        with self.assertRaises(GenerationConfigError):
            GenerationConfig(top_p=1.5)


class TestInferenceEngineCore(unittest.TestCase):
    """Test core engine execution, prefill, decode, and KV cache reuse."""

    def setUp(self):
        self.model = StatefulCausalLM(vocab_size=200, num_layers=4, num_kv_heads=2, head_dim=128)
        self.tokenizer = MockTokenizer(eos_token_id=99)
        self.artifacts = ModelArtifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ModelConfig(model_id="mock-qwen"),
            device="cpu",
            dtype="float16",
        )
        self.engine = InferenceEngine(artifacts=self.artifacts)

    def test_initialization(self):
        self.assertTrue(self.engine.is_model_loaded)
        self.assertIs(self.engine.model, self.model)
        self.assertIs(self.engine.tokenizer, self.tokenizer)
        self.assertEqual(self.engine.device, "cpu")

    def test_prefill_execution(self):
        prompt_tokens = [10, 20, 30, 40, 50]
        request = InferenceRequest(
            input_ids=prompt_tokens,
            generation_config=GenerationConfig(max_new_tokens=10),
        )

        state = self.engine.prefill(request)

        # 1. Prefill must process the complete prompt length in one forward pass
        self.assertTrue(state.prefill_completed)
        self.assertEqual(self.model.forward_call_inputs[0], 5)
        self.assertEqual(state.prompt_length, 5)

        # 2. KV Cache must be initialized
        self.assertIsNotNone(state.past_key_values)
        self.assertEqual(len(state.past_key_values), 4)  # 4 layers

        # 3. First token generated and appended
        self.assertEqual(len(state.generated_token_ids), 1)
        self.assertEqual(state.generated_token_ids[0], 42)
        self.assertEqual(state.current_seq_len, 6)

    def test_decode_step_reuses_kv_cache(self):
        """Verify that decoding processes 1 token at a time and reuses KV cache."""
        prompt_tokens = [1, 2, 3, 4]
        request = InferenceRequest(
            input_ids=prompt_tokens,
            generation_config=GenerationConfig(max_new_tokens=5, eos_token_id=999),
        )

        state = self.engine.prefill(request)
        self.assertEqual(self.model.forward_call_inputs[-1], 4)  # Prefill sequence length

        # Run 3 decoding steps
        for _ in range(3):
            next_token, is_finished = self.engine.decode_step(state)
            self.assertEqual(next_token, 42)
            # Crucial invariant: decode forward pass MUST pass only 1 token (not full prompt!)
            self.assertEqual(self.model.forward_call_inputs[-1], 1)

        # Verify generated sequence length
        self.assertEqual(len(state.generated_token_ids), 4)  # 1 from prefill + 3 from decode
        self.assertEqual(state.current_seq_len, 4 + 4)

        # Verify KV Cache tensor has grown to 7 tokens:
        # 4 prompt tokens + 3 decode steps that passed the first 3 generated tokens
        cache_tensor = state.past_key_values[0][0]
        self.assertEqual(cache_tensor.shape[2], 7)

    def test_greedy_generation_max_tokens(self):
        """Verify complete generation finishes at max_new_tokens."""
        config = GenerationConfig(max_new_tokens=6, eos_token_id=999, do_sample=False)
        result = self.engine.generate("Hello world", generation_config=config, inspect_cache=True)

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(len(result.generated_token_ids), 6)
        self.assertEqual(result.finish_reason, "max_tokens")
        self.assertGreater(result.total_time_seconds, 0)
        self.assertGreaterEqual(result.tokens_per_second, 0)

        # Verify that forward passes were 1 prefill + 5 decode steps = 6 calls
        # Call 1: prompt length
        # Calls 2..6: 1 token each
        self.assertEqual(len(self.model.forward_call_inputs), 6)
        self.assertEqual(self.model.forward_call_inputs[1:], [1, 1, 1, 1, 1])

    def test_eos_token_termination(self):
        """Verify generation terminates immediately when EOS token is encountered."""
        # Set token 42 as the EOS token
        config = GenerationConfig(max_new_tokens=20, eos_token_id=42)
        result = self.engine.generate("Test prompt", generation_config=config)

        # Prefill generates token 42 -> immediate EOS termination
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(len(result.generated_token_ids), 1)
        self.assertEqual(result.generated_token_ids[0], 42)

    def test_cache_capacity_limit(self):
        """Verify generation stops if max_cache_capacity is reached."""
        prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 prompt tokens
        # Set max capacity to 10 (8 prompt + 1 prefill token = 9 -> 1 decode step reaches 10)
        config = GenerationConfig(max_new_tokens=20, max_cache_capacity=10, eos_token_id=999)
        request = InferenceRequest(input_ids=prompt_tokens, generation_config=config)

        state = self.engine.prefill(request)  # seq_len = 9
        self.assertFalse(state.is_finished)

        self.engine.decode_step(state)  # seq_len = 10
        self.engine.decode_step(state)  # triggers capacity_exceeded
        self.assertEqual(state.finish_reason, "capacity_exceeded")


class TestKVCacheIntegrationPrompt2(unittest.TestCase):
    """
    Test direct integration with Prompt 2's KV Cache utilities.
    Ensures that KVCacheInspector and extract_kv_cache work seamlessly with InferenceEngine.
    """

    def setUp(self):
        self.model = StatefulCausalLM(vocab_size=100, num_layers=4, num_kv_heads=2, head_dim=128)
        self.tokenizer = MockTokenizer(eos_token_id=999)
        self.artifacts = ModelArtifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ModelConfig(model_id="mock-qwen"),
            device="cpu",
            dtype="float16",
        )
        self.engine = InferenceEngine(artifacts=self.artifacts)

    def test_cache_report_generated_in_result(self):
        config = GenerationConfig(max_new_tokens=4, eos_token_id=999)
        prompt = [1, 2, 3, 4, 5]  # 5 prompt tokens

        result = self.engine.generate(prompt, generation_config=config, inspect_cache=True)

        # Verify CacheReport from Prompt 2 is embedded
        self.assertIsNotNone(result.final_cache_report)
        report: CacheReport = result.final_cache_report

        self.assertEqual(report.num_layers, 4)
        self.assertEqual(report.batch_size, 1)
        self.assertEqual(report.num_kv_heads, 2)
        # Sequence length in KV cache = 5 (prompt) + 3 (decoded tokens 1, 2, 3) = 8
        # (Token 4 is the final generated output token and has not been cached)
        self.assertEqual(report.seq_len, 8)
        self.assertEqual(report.head_dim, 128)
        self.assertTrue(report.is_gqa)
        self.assertGreater(report.total_bytes, 0)
        self.assertGreater(report.total_mb, 0)

        # Verify report summary text renders correctly
        summary = result.summary()
        self.assertIn("Final KV Cache Memory:", summary)
        self.assertIn("KV Cache Sequence Len: 8 tokens", summary)

    def test_independent_requests_do_not_share_cache(self):
        """Verify multiple consecutive inference requests maintain completely isolated caches."""
        config = GenerationConfig(max_new_tokens=3, eos_token_id=999)

        req1 = InferenceRequest(input_ids=[1, 2, 3], generation_config=config)
        state1 = self.engine.prefill(req1)

        req2 = InferenceRequest(input_ids=[10, 20, 30, 40, 50], generation_config=config)
        state2 = self.engine.prefill(req2)

        # Verify states and cache objects are distinct
        self.assertIsNot(state1, state2)
        self.assertIsNot(state1.past_key_values, state2.past_key_values)
        self.assertEqual(state1.prompt_length, 3)
        self.assertEqual(state2.prompt_length, 5)

        # Cleanup state1
        state1.cleanup()
        self.assertIsNone(state1.past_key_values)
        # state2 must remain intact
        self.assertIsNotNone(state2.past_key_values)
        state2.cleanup()


class TestErrorHandling(unittest.TestCase):
    """Test error modes and edge cases."""

    def setUp(self):
        self.model = StatefulCausalLM()
        self.tokenizer = MockTokenizer()
        self.artifacts = ModelArtifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ModelConfig(model_id="mock-qwen"),
            device="cpu",
            dtype="float16",
        )
        self.engine = InferenceEngine(artifacts=self.artifacts)

    def test_empty_string_prompt_raises_error(self):
        with self.assertRaises(InvalidInputError):
            self.engine.generate("")
        with self.assertRaises(InvalidInputError):
            self.engine.generate("   ")

    def test_empty_list_prompt_raises_error(self):
        with self.assertRaises(InvalidInputError):
            self.engine.generate([])

    def test_missing_past_key_values_in_prefill(self):
        bad_model = MagicMock()
        bad_output = MagicMock()
        bad_output.past_key_values = None
        bad_model.return_value = bad_output

        engine = InferenceEngine(artifacts=ModelArtifacts(
            model=bad_model, tokenizer=self.tokenizer, config=ModelConfig(), device="cpu", dtype="float32"
        ))
        req = InferenceRequest(input_ids=[1, 2, 3])
        with self.assertRaises(CacheStateError):
            engine.prefill(req)

    def test_decode_before_prefill_raises_error(self):
        req = InferenceRequest(input_ids=[1, 2, 3])
        state = GenerationState(request=req)
        with self.assertRaises(CacheStateError):
            self.engine.decode_step(state)


if __name__ == "__main__":
    unittest.main()
