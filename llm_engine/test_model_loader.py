"""
Unit tests for DSAA LLM Engine Model Loader.

These tests execute in an isolated environment using mocks, validating:
- Device selection logic (CUDA, CPU, Auto)
- Configuration validation (valid/invalid inputs)
- Precision selection (FP32, FP16, BF16, hardware-dependent auto)
- Model loader initialization, caching, and unloading
- Tokenizer setup (padding side, pad token handling)
- Model setup (eval mode, requires_grad=False, use_cache=True)
- Error handling (missing dependencies, network/download errors, OOM)
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from llm_engine.config import (
    ModelConfig,
    DeviceType,
    PrecisionMode,
    DEFAULT_MODEL_ID,
    resolve_device,
    resolve_precision,
)
from llm_engine.exceptions import (
    DependencyMissingError,
    DeviceConfigurationError,
    PrecisionConfigurationError,
    ModelDownloadError,
    ModelInitializationError,
    TokenizerInitializationError,
)
from llm_engine.model_loader import ModelLoader, ModelArtifacts, load_model_and_tokenizer


class TestModelConfig(unittest.TestCase):
    """Test suite for ModelConfig validation and parameter defaults."""

    def test_default_config(self):
        """Verify default configuration values."""
        cfg = ModelConfig()
        self.assertEqual(cfg.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.precision, "auto")
        self.assertTrue(cfg.use_cache)
        self.assertTrue(cfg.trust_remote_code)
        self.assertEqual(cfg.padding_side, "left")

    def test_custom_valid_config(self):
        """Verify custom valid parameters are retained."""
        cfg = ModelConfig(
            model_id="google/gemma-2-2b-it",
            device="cpu",
            precision="float32",
            padding_side="left",
            max_new_tokens=256,
        )
        self.assertEqual(cfg.model_id, "google/gemma-2-2b-it")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.precision, "float32")
        self.assertEqual(cfg.max_new_tokens, 256)

    def test_invalid_model_id(self):
        """ModelConfig must reject empty or whitespace-only model_id."""
        with self.assertRaises(ValueError):
            ModelConfig(model_id="")
        with self.assertRaises(ValueError):
            ModelConfig(model_id="   ")

    def test_invalid_device(self):
        """ModelConfig must reject unsupported device types."""
        with self.assertRaises(DeviceConfigurationError):
            ModelConfig(device="tpu")
        with self.assertRaises(DeviceConfigurationError):
            ModelConfig(device="vpu")

    def test_invalid_precision(self):
        """ModelConfig must reject unsupported precision formats."""
        with self.assertRaises(PrecisionConfigurationError):
            ModelConfig(precision="int4")
        with self.assertRaises(PrecisionConfigurationError):
            ModelConfig(precision="fp8")

    def test_invalid_padding_side(self):
        """ModelConfig must reject invalid padding sides."""
        with self.assertRaises(ValueError):
            ModelConfig(padding_side="center")


class TestDeviceSelection(unittest.TestCase):
    """Test hardware device resolution logic."""

    def setUp(self):
        self.mock_torch = MagicMock()

    def test_auto_device_cuda_available(self):
        """Auto mode resolves to 'cuda' when CUDA is available."""
        self.mock_torch.cuda.is_available.return_value = True
        device = resolve_device("auto", torch_module=self.mock_torch)
        self.assertEqual(device, "cuda")

    def test_auto_device_cuda_unavailable(self):
        """Auto mode resolves to 'cpu' when CUDA is unavailable."""
        self.mock_torch.cuda.is_available.return_value = False
        device = resolve_device("auto", torch_module=self.mock_torch)
        self.assertEqual(device, "cpu")

    def test_explicit_cpu(self):
        """Explicit 'cpu' returns 'cpu' without querying CUDA."""
        device = resolve_device("cpu", torch_module=self.mock_torch)
        self.assertEqual(device, "cpu")

    def test_explicit_cuda_when_available(self):
        """Explicit 'cuda' succeeds when CUDA is available."""
        self.mock_torch.cuda.is_available.return_value = True
        device = resolve_device("cuda", torch_module=self.mock_torch)
        self.assertEqual(device, "cuda")

    def test_explicit_cuda_when_unavailable_raises_error(self):
        """Explicit 'cuda' raises DeviceConfigurationError if hardware lacks CUDA."""
        self.mock_torch.cuda.is_available.return_value = False
        with self.assertRaises(DeviceConfigurationError):
            resolve_device("cuda", torch_module=self.mock_torch)

    def test_explicit_cuda_indexed_device(self):
        """Explicit 'cuda:0' is preserved when CUDA is available."""
        self.mock_torch.cuda.is_available.return_value = True
        device = resolve_device("cuda:0", torch_module=self.mock_torch)
        self.assertEqual(device, "cuda:0")


class TestPrecisionSelection(unittest.TestCase):
    """Test precision resolution and hardware compatibility checks."""

    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_torch.float32 = "torch.float32"
        self.mock_torch.float16 = "torch.float16"
        self.mock_torch.bfloat16 = "torch.bfloat16"

    def test_auto_precision_cuda_with_bf16(self):
        """On CUDA with BF16 instruction support, auto selects bfloat16."""
        self.mock_torch.cuda.is_bf16_supported.return_value = True
        dtype = resolve_precision("auto", "cuda", torch_module=self.mock_torch)
        self.assertEqual(dtype, "torch.bfloat16")

    def test_auto_precision_cuda_without_bf16(self):
        """On CUDA without BF16 support, auto selects float16."""
        self.mock_torch.cuda.is_bf16_supported.return_value = False
        dtype = resolve_precision("auto", "cuda", torch_module=self.mock_torch)
        self.assertEqual(dtype, "torch.float16")

    def test_auto_precision_cpu(self):
        """On CPU, auto selects safe float32."""
        dtype = resolve_precision("auto", "cpu", torch_module=self.mock_torch)
        self.assertEqual(dtype, "torch.float32")

    def test_explicit_float32(self):
        """Explicit float32 returns float32 on both CPU and CUDA."""
        dtype_cpu = resolve_precision("float32", "cpu", torch_module=self.mock_torch)
        dtype_cuda = resolve_precision("float32", "cuda", torch_module=self.mock_torch)
        self.assertEqual(dtype_cpu, "torch.float32")
        self.assertEqual(dtype_cuda, "torch.float32")

    def test_explicit_float16_on_cpu_raises(self):
        """Explicit float16 on CPU raises error due to lack of standard kernel support."""
        with self.assertRaises(PrecisionConfigurationError):
            resolve_precision("float16", "cpu", torch_module=self.mock_torch)

    def test_explicit_bfloat16_unsupported_gpu_raises(self):
        """Explicit bfloat16 on non-BF16 GPU raises PrecisionConfigurationError."""
        self.mock_torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaises(PrecisionConfigurationError):
            resolve_precision("bfloat16", "cuda", torch_module=self.mock_torch)


class TestDependencyHandling(unittest.TestCase):
    """Test handling of missing libraries (torch, transformers)."""

    def test_missing_torch_raises_dependency_error(self):
        loader = ModelLoader()
        with patch.dict(sys.modules, {"torch": None}):
            # Force ImportError when importing torch
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                with self.assertRaises(DependencyMissingError) as ctx:
                    loader._verify_dependencies()
                self.assertIn("torch", str(ctx.exception))

    def test_missing_transformers_raises_dependency_error(self):
        loader = ModelLoader()
        mock_torch = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            def mock_import(name, *args, **kwargs):
                if name == "transformers":
                    raise ImportError("No module named 'transformers'")
                return MagicMock()
            with patch("builtins.__import__", side_effect=mock_import):
                with self.assertRaises(DependencyMissingError) as ctx:
                    loader._verify_dependencies()
                self.assertIn("transformers", str(ctx.exception))


class TestModelLoaderOperations(unittest.TestCase):
    """Test model loading, configuration, caching, and unloading with mocks."""

    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = False
        self.mock_torch.float32 = "torch.float32"

        self.mock_transformers = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model_param = MagicMock()
        self.mock_model.parameters.return_value = [self.mock_model_param]
        self.mock_model.config.use_cache = False

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "<|im_end|>"
        self.mock_tokenizer.eos_token_id = 151645

        self.mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = self.mock_model
        self.mock_transformers.AutoTokenizer.from_pretrained.return_value = self.mock_tokenizer

    def test_initial_state_not_loaded(self):
        loader = ModelLoader()
        self.assertFalse(loader.is_loaded)

    def test_load_configures_model_and_tokenizer(self):
        """Verify model is in eval mode, gradients disabled, cache enabled, tokenizer has left padding."""
        loader = ModelLoader(ModelConfig(device="cpu", precision="float32"))

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            artifacts = loader.load()

            self.assertTrue(loader.is_loaded)
            self.assertEqual(artifacts.device, "cpu")
            self.assertEqual(artifacts.dtype, "torch.float32")

            # Check model configuration
            self.mock_model.eval.assert_called_once()
            self.assertFalse(self.mock_model_param.requires_grad)
            self.assertTrue(self.mock_model.config.use_cache)

            # Check tokenizer configuration
            self.assertEqual(self.mock_tokenizer.padding_side, "left")
            self.assertEqual(self.mock_tokenizer.pad_token, "<|im_end|>")

    def test_load_caching(self):
        """Calling load() multiple times returns cached artifacts without re-reading weights."""
        loader = ModelLoader(ModelConfig(device="cpu", precision="float32"))

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            art1 = loader.load()
            art2 = loader.load()

            self.assertIs(art1, art2)
            self.assertEqual(self.mock_transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)

    def test_unload(self):
        """unload() frees references and sets is_loaded to False."""
        loader = ModelLoader(ModelConfig(device="cpu", precision="float32"))

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            loader.load()
            self.assertTrue(loader.is_loaded)

            loader.unload()
            self.assertFalse(loader.is_loaded)

    def test_get_model_and_get_tokenizer_auto_load(self):
        """get_model() and get_tokenizer() trigger load() if not yet loaded."""
        loader = ModelLoader(ModelConfig(device="cpu", precision="float32"))

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            m = loader.get_model()
            t = loader.get_tokenizer()

            self.assertIs(m, self.mock_model)
            self.assertIs(t, self.mock_tokenizer)
            self.assertTrue(loader.is_loaded)


class TestErrorHandling(unittest.TestCase):
    """Test failure modes: network errors, corrupt models, and OOM conditions."""

    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = False
        self.mock_transformers = MagicMock()

    def test_tokenizer_download_failure(self):
        """OSError during tokenizer loading raises ModelDownloadError."""
        self.mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("Connection timed out")
        loader = ModelLoader()

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            with self.assertRaises(ModelDownloadError):
                loader.load()

    def test_model_download_failure(self):
        """OSError during model weight loading raises ModelDownloadError."""
        self.mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        self.mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = OSError("Repo not found")
        loader = ModelLoader()

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            with self.assertRaises(ModelDownloadError):
                loader.load()

    def test_out_of_memory_error(self):
        """CUDA out of memory error raises ModelInitializationError with memory advice."""
        self.mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        self.mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )
        loader = ModelLoader()

        with patch.object(loader, "_verify_dependencies", return_value=(self.mock_torch, self.mock_transformers)):
            with self.assertRaises(ModelInitializationError) as ctx:
                loader.load()
            self.assertIn("Insufficient memory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
