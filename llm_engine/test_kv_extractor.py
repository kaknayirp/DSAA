"""
Unit and integration tests for DSAA KV Cache Extraction and Inspection.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Tests cover:
- Extraction from DynamicCache, tuple of tuples, and list of tuples
- Tensor shape and dimension extraction (including Qwen2.5-1.5B GQA layout)
- Accurate memory calculation for keys, values, layers, and total cache
- Multiple data types (float32, float16, bfloat16, int8, int64)
- Physical vs logical storage and shared storage detection
- Device reporting (CPU and simulated CUDA)
- Error handling for None, empty, or mismatched caches
- Report serialization (to_dict, summary_table)
- End-to-end inference integration pipeline
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Tuple, Optional

from llm_engine.kv_extractor import (
    KVCacheInspector,
    CacheReport,
    LayerCacheStats,
    TensorStats,
    KVCacheInspectionError,
    inspect_tensor,
    extract_kv_cache,
    run_inference_and_inspect_cache,
    get_tensor_element_size,
)


class MockStorage:
    """Simulates PyTorch UntypedStorage for testing physical memory tracking."""
    def __init__(self, data_ptr_val: int, nbytes_val: int):
        self._data_ptr = data_ptr_val
        self._nbytes = nbytes_val

    def data_ptr(self) -> int:
        return self._data_ptr

    def nbytes(self) -> int:
        return self._nbytes


class SyntheticTensor:
    """
    Synthetic tensor object simulating torch.Tensor behavior for unit testing
    without requiring GPU hardware or large model weights in test environments.
    """
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype_str: str = "torch.float16",
        device_str: str = "cpu",
        storage_ptr: Optional[int] = None,
        storage_nbytes: Optional[int] = None,
        is_contiguous_val: bool = True,
    ):
        self.shape = shape
        self.dtype = dtype_str
        self.device = device_str
        self._is_contiguous = is_contiguous_val

        # Calculate elements
        numel = 1
        for d in shape:
            numel *= d
        self._numel = numel

        # Element size based on dtype string
        if "float32" in dtype_str or "int32" in dtype_str:
            self._element_size = 4
        elif "float16" in dtype_str or "bfloat16" in dtype_str or "int16" in dtype_str:
            self._element_size = 2
        elif "int8" in dtype_str or "uint8" in dtype_str:
            self._element_size = 1
        elif "int64" in dtype_str or "float64" in dtype_str:
            self._element_size = 8
        else:
            self._element_size = 4

        # Storage simulation
        if storage_ptr is not None:
            nbytes = storage_nbytes if storage_nbytes is not None else (self._numel * self._element_size)
            self._storage = MockStorage(storage_ptr, nbytes)
        else:
            self._storage = None

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def is_contiguous(self) -> bool:
        return self._is_contiguous


class MockDynamicCache:
    """Simulates transformers.cache_utils.DynamicCache."""
    def __init__(self, key_cache: list, value_cache: list):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def __len__(self):
        return len(self.key_cache)


class TestTensorInspection(unittest.TestCase):
    """Test inspection and memory extraction for individual tensors."""

    def test_float32_tensor(self):
        # Shape: (1, 2, 64, 128) -> numel = 16,384 -> 16,384 * 4 bytes = 65,536 bytes (64 KB)
        t = SyntheticTensor(shape=(1, 2, 64, 128), dtype_str="torch.float32", device_str="cpu")
        stats = inspect_tensor(t)

        self.assertEqual(stats.shape, (1, 2, 64, 128))
        self.assertEqual(stats.dtype, "torch.float32")
        self.assertEqual(stats.device, "cpu")
        self.assertEqual(stats.numel, 16384)
        self.assertEqual(stats.element_size_bytes, 4)
        self.assertEqual(stats.logical_bytes, 65536)
        self.assertEqual(stats.logical_kb, 64.0)
        self.assertEqual(stats.logical_mb, 64.0 / 1024.0)
        self.assertTrue(stats.is_contiguous)

    def test_float16_and_bfloat16_tensor(self):
        # 16-bit dtypes are 2 bytes per element
        t_fp16 = SyntheticTensor(shape=(2, 4, 32, 64), dtype_str="torch.float16", device_str="cuda:0")
        t_bf16 = SyntheticTensor(shape=(2, 4, 32, 64), dtype_str="torch.bfloat16", device_str="cuda:0")

        stats_fp16 = inspect_tensor(t_fp16)
        stats_bf16 = inspect_tensor(t_bf16)

        self.assertEqual(stats_fp16.element_size_bytes, 2)
        self.assertEqual(stats_bf16.element_size_bytes, 2)
        self.assertEqual(stats_fp16.logical_bytes, 2 * 4 * 32 * 64 * 2)
        self.assertEqual(stats_bf16.logical_bytes, stats_fp16.logical_bytes)
        self.assertEqual(stats_fp16.device, "cuda:0")

    def test_int8_tensor(self):
        # 8-bit quantized cache: 1 byte per element
        t_int8 = SyntheticTensor(shape=(1, 2, 100, 128), dtype_str="torch.int8")
        stats = inspect_tensor(t_int8)
        self.assertEqual(stats.element_size_bytes, 1)
        self.assertEqual(stats.logical_bytes, 1 * 2 * 100 * 128 * 1)

    def test_none_tensor_raises(self):
        with self.assertRaises(KVCacheInspectionError):
            inspect_tensor(None)


class TestCacheRepresentationFormats(unittest.TestCase):
    """Test extraction across DynamicCache, tuple of tuples, and list of tuples."""

    def setUp(self):
        # Create 4 layers of synthetic (key, value) tensors
        self.num_layers = 4
        self.shape = (1, 2, 32, 128)  # Qwen-style GQA layout (2 KV heads, head_dim 128)
        self.layers = [
            (
                SyntheticTensor(self.shape, "torch.bfloat16"),
                SyntheticTensor(self.shape, "torch.bfloat16"),
            )
            for _ in range(self.num_layers)
        ]

    def test_dynamic_cache_extraction(self):
        keys = [pair[0] for pair in self.layers]
        values = [pair[1] for pair in self.layers]
        dynamic_cache = MockDynamicCache(keys, values)

        inspector = KVCacheInspector()
        cache_type, extracted = inspector.extract_layers(dynamic_cache)

        self.assertEqual(cache_type, "DynamicCache")
        self.assertEqual(len(extracted), self.num_layers)

    def test_tuple_of_tuples_extraction(self):
        tuple_cache = tuple(self.layers)

        inspector = KVCacheInspector()
        cache_type, extracted = inspector.extract_layers(tuple_cache)

        self.assertEqual(cache_type, "tuple_of_tuples")
        self.assertEqual(len(extracted), self.num_layers)

    def test_list_of_tuples_extraction(self):
        list_cache = list(self.layers)

        inspector = KVCacheInspector()
        cache_type, extracted = inspector.extract_layers(list_cache)

        self.assertEqual(cache_type, "list_of_tuples")
        self.assertEqual(len(extracted), self.num_layers)

    def test_empty_cache_raises_error(self):
        inspector = KVCacheInspector()
        with self.assertRaises(KVCacheInspectionError):
            inspector.extract_layers(())
        with self.assertRaises(KVCacheInspectionError):
            inspector.extract_layers([])

    def test_none_cache_raises_error(self):
        inspector = KVCacheInspector()
        with self.assertRaises(KVCacheInspectionError):
            inspector.extract_layers(None)

    def test_mismatched_dynamic_cache_raises(self):
        bad_cache = MockDynamicCache([SyntheticTensor(self.shape)], [])
        inspector = KVCacheInspector()
        with self.assertRaises(KVCacheInspectionError):
            inspector.extract_layers(bad_cache)


class TestQwenMemoryCalculations(unittest.TestCase):
    """
    Simulate full Qwen2.5-1.5B-Instruct architecture KV Cache:
    - 28 transformer layers
    - GQA with 2 key/value heads
    - Head dimension: 128
    - Sequence length: 128 tokens
    - Batch size: 1
    - Precision: float16 (2 bytes per element)
    """

    def setUp(self):
        self.num_layers = 28
        self.batch_size = 1
        self.num_kv_heads = 2
        self.seq_len = 128
        self.head_dim = 128
        self.shape = (self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim)

        self.cache = tuple(
            (
                SyntheticTensor(self.shape, "torch.float16", "cpu", storage_ptr=1000 + i * 2),
                SyntheticTensor(self.shape, "torch.float16", "cpu", storage_ptr=1000 + i * 2 + 1),
            )
            for i in range(self.num_layers)
        )

    def test_exact_memory_calculation(self):
        report = extract_kv_cache(self.cache)

        # Expected calculations:
        # Elements per tensor = 1 * 2 * 128 * 128 = 32,768 elements
        # Bytes per tensor (float16 = 2 bytes) = 32,768 * 2 = 65,536 bytes (64.0 KB)
        # Per layer (key + value) = 65,536 * 2 = 131,072 bytes (128.0 KB)
        # Total for 28 layers = 28 * 131,072 = 3,670,016 bytes = 3,584.0 KB = 3.5 MB
        expected_tensor_bytes = 1 * 2 * 128 * 128 * 2
        expected_layer_bytes = expected_tensor_bytes * 2
        expected_total_bytes = 28 * expected_layer_bytes

        self.assertEqual(report.num_layers, 28)
        self.assertEqual(report.batch_size, 1)
        self.assertEqual(report.num_kv_heads, 2)
        self.assertEqual(report.seq_len, 128)
        self.assertEqual(report.head_dim, 128)
        self.assertTrue(report.is_gqa)

        # Verify Layer 0
        layer0 = report.layers[0]
        self.assertEqual(layer0.key.logical_bytes, expected_tensor_bytes)
        self.assertEqual(layer0.value.logical_bytes, expected_tensor_bytes)
        self.assertEqual(layer0.total_bytes, expected_layer_bytes)
        self.assertEqual(layer0.total_kb, 128.0)

        # Verify Total Cache
        self.assertEqual(report.total_bytes, expected_total_bytes)
        self.assertEqual(report.total_kb, 3584.0)
        self.assertAlmostEqual(report.total_mb, 3.5, places=4)
        self.assertEqual(report.actual_storage_bytes, expected_total_bytes)

    def test_shared_storage_detection(self):
        # Simulate preallocated contiguous buffer where all layers slice from a shared storage ptr
        shared_ptr = 99999
        shared_storage_bytes = 10 * 1024 * 1024  # 10 MB buffer
        shared_cache = tuple(
            (
                SyntheticTensor(self.shape, "torch.float16", storage_ptr=shared_ptr, storage_nbytes=shared_storage_bytes),
                SyntheticTensor(self.shape, "torch.float16", storage_ptr=shared_ptr, storage_nbytes=shared_storage_bytes),
            )
            for _ in range(4)
        )

        report = extract_kv_cache(shared_cache)
        self.assertTrue(report.is_shared_storage)
        # The physical storage should only be counted once (10 MB), not 8 times
        self.assertEqual(report.actual_storage_bytes, shared_storage_bytes)


    def test_report_serialization_and_summary(self):
        report = extract_kv_cache(self.cache)
        rep_dict = report.to_dict()

        self.assertIsInstance(rep_dict, dict)
        self.assertEqual(rep_dict["num_layers"], 28)
        self.assertEqual(rep_dict["total_bytes"], 3670016)
        self.assertIn("layers", rep_dict)
        self.assertEqual(len(rep_dict["layers"]), 28)

        summary_text = report.summary_table()
        self.assertIn("DSAA KV CACHE INSPECTION REPORT", summary_text)
        self.assertIn("Total Layers:         28", summary_text)
        self.assertIn("KV Heads per Layer:   2 (Grouped Query Attention)", summary_text)
        self.assertIn("3,670,016 bytes", summary_text)


class TestModelInferenceIntegration(unittest.TestCase):
    """Test run_inference_and_inspect_cache with mock model pipeline."""

    def test_inference_and_inspect_pipeline(self):
        mock_model = MagicMock()
        mock_model.device = "cpu"

        # Mock forward pass output
        mock_outputs = MagicMock()
        num_layers = 2
        shape = (1, 2, 16, 64)
        mock_outputs.past_key_values = tuple(
            (SyntheticTensor(shape, "torch.float32"), SyntheticTensor(shape, "torch.float32"))
            for _ in range(num_layers)
        )
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value="input_ids_tensor")),
            "attention_mask": MagicMock(to=MagicMock(return_value="mask_tensor")),
        }

        with patch.dict(sys.modules, {"torch": MagicMock()}):
            outputs, report = run_inference_and_inspect_cache(
                mock_model,
                mock_tokenizer,
                prompt="Test prompt",
                device="cpu"
            )

            self.assertIs(outputs, mock_outputs)
            self.assertEqual(report.num_layers, num_layers)
            self.assertEqual(report.seq_len, 16)
            self.assertEqual(report.head_dim, 64)


if __name__ == "__main__":
    unittest.main()
