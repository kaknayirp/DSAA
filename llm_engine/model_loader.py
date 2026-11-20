"""
Model Loader for the DSAA LLM Engine.

Handles downloading, initializing, configuring, and caching pretrained LLMs
and tokenizers with safe device and precision fallbacks.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional, Any, Tuple

from .config import ModelConfig, resolve_device, resolve_precision
from .exceptions import (
    DependencyMissingError,
    ModelDownloadError,
    ModelInitializationError,
    TokenizerInitializationError,
    ModelLoaderError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container holding the loaded model, tokenizer, and resolved runtime configurations."""
    model: Any
    tokenizer: Any
    config: ModelConfig
    device: str
    dtype: Any


class ModelLoader:
    """
    Manages the lifecycle of the LLM and tokenizer:
    - Verifies dependencies (torch, transformers)
    - Validates hardware & resolves optimal device and precision
    - Configures models for inference (eval mode, no grad, use_cache=True)
    - Sets up tokenizers for autoregressive generation (left padding, pad token)
    - Caches loaded artifacts to prevent duplicate memory allocations
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self._artifacts: Optional[ModelArtifacts] = None

    @property
    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are currently loaded in memory."""
        return self._artifacts is not None

    def _verify_dependencies(self) -> Tuple[Any, Any]:
        """
        Check that PyTorch and Hugging Face Transformers are installed.
        Returns imported (torch, transformers) modules.
        Raises DependencyMissingError if either is missing.
        """
        try:
            import torch
        except ImportError as exc:
            raise DependencyMissingError(
                "torch",
                "pip install torch --index-url https://download.pytorch.org/whl/cpu (or CUDA equivalent)"
            ) from exc

        try:
            import transformers
        except ImportError as exc:
            raise DependencyMissingError(
                "transformers",
                "pip install transformers accelerate"
            ) from exc

        return torch, transformers

    def load(self, force_reload: bool = False) -> ModelArtifacts:
        """
        Load the model and tokenizer according to the active configuration.

        Args:
            force_reload: If True, unloads any previously loaded model and reloads fresh.

        Returns:
            ModelArtifacts containing loaded model, tokenizer, device, and dtype.
        """
        if self._artifacts is not None and not force_reload:
            logger.info("Model '%s' is already loaded. Returning cached artifacts.", self.config.model_id)
            return self._artifacts

        if self._artifacts is not None and force_reload:
            self.unload()

        torch, transformers = self._verify_dependencies()

        # Resolve device and precision
        resolved_device = resolve_device(self.config.device, torch_module=torch)
        resolved_dtype = resolve_precision(self.config.precision, resolved_device, torch_module=torch)

        logger.info(
            "Loading model '%s' on device '%s' with precision '%s'",
            self.config.model_id, resolved_device, resolved_dtype
        )

        # 1. Load Tokenizer
        tokenizer = self._load_tokenizer(transformers)

        # 2. Load Model
        model = self._load_model(torch, transformers, resolved_device, resolved_dtype)

        # 3. Cache artifacts
        self._artifacts = ModelArtifacts(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            device=resolved_device,
            dtype=resolved_dtype,
        )

        return self._artifacts

    def _load_tokenizer(self, transformers: Any) -> Any:
        """Initialize and configure tokenizer for causal autoregressive generation."""
        tokenizer_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            **self.config.extra_tokenizer_kwargs,
        }
        if self.config.cache_dir:
            tokenizer_kwargs["cache_dir"] = self.config.cache_dir

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model_id,
                **tokenizer_kwargs
            )
        except OSError as exc:
            raise ModelDownloadError(
                f"Failed to download or load tokenizer for '{self.config.model_id}'. "
                f"Please verify model ID and network connectivity. Error: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise TokenizerInitializationError(
                f"Unexpected error initializing tokenizer for '{self.config.model_id}': {str(exc)}"
            ) from exc

        # Ensure padding side is 'left' for decoder-only autoregressive batch generation
        tokenizer.padding_side = self.config.padding_side

        # Ensure pad_token is defined so attention masks work correctly during batched inference
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Tokenizer has no pad_token; set pad_token = eos_token ('%s').", tokenizer.eos_token)
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                logger.info("Tokenizer has neither pad_token nor eos_token; added '<|pad|>'.")

        return tokenizer

    def _load_model(
        self,
        torch: Any,
        transformers: Any,
        device: str,
        dtype: Any
    ) -> Any:
        """Initialize and configure the model for evaluation and KV Cache generation."""
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
            **self.config.extra_model_kwargs,
        }
        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir

        # Device mapping: if cuda, use direct placement or device_map
        if device.startswith("cuda"):
            # If accelerate is available, use device_map or direct cuda placement
            model_kwargs["device_map"] = device
        else:
            model_kwargs["device_map"] = "cpu"

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
        except (OSError, ConnectionError) as exc:
            raise ModelDownloadError(
                f"Failed to download or locate model weights for '{self.config.model_id}'. "
                f"Check internet connection, Hugging Face Hub status, or model name. Error: {str(exc)}"
            ) from exc
        except (RuntimeError, Exception) as exc:
            # Check for Out of Memory
            is_oom = False
            oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
            if isinstance(oom_type, type) and isinstance(exc, oom_type):
                is_oom = True
            elif "out of memory" in str(exc).lower():
                is_oom = True

            if is_oom:
                raise ModelInitializationError(
                    f"Insufficient memory to load model '{self.config.model_id}' on device '{device}'. "
                    f"Consider using a smaller model, CPU offloading, or lower precision. Error: {str(exc)}"
                ) from exc

            raise ModelInitializationError(
                f"Failed to initialize model '{self.config.model_id}': {str(exc)}"
            ) from exc

        # Configure model specifically for evaluation / autoregressive inference
        model.eval()

        # Disable gradient calculations for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Ensure KV Cache is explicitly enabled
        if hasattr(model, "config"):
            model.config.use_cache = self.config.use_cache

        return model

    def get_model(self) -> Any:
        """Return the loaded model instance. Calls load() if not yet loaded."""
        if self._artifacts is None:
            self.load()
        return self._artifacts.model

    def get_tokenizer(self) -> Any:
        """Return the loaded tokenizer instance. Calls load() if not yet loaded."""
        if self._artifacts is None:
            self.load()
        return self._artifacts.tokenizer

    def get_artifacts(self) -> ModelArtifacts:
        """Return all loaded artifacts (model, tokenizer, device, dtype, config)."""
        if self._artifacts is None:
            self.load()
        return self._artifacts

    def unload(self) -> None:
        """Release loaded model and tokenizer from memory and clear GPU cache."""
        if self._artifacts is not None:
            logger.info("Unloading model '%s' from memory.", self.config.model_id)
            del self._artifacts.model
            del self._artifacts.tokenizer
            self._artifacts = None

        gc.collect()

        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "auto",
    precision: str = "auto",
    **kwargs: Any
) -> ModelArtifacts:
    """
    Convenience function to load a model and tokenizer with a single function call.

    Args:
        model_id: Hugging Face model identifier or local directory.
        device: 'auto', 'cuda', 'cpu', or 'cuda:<idx>'
        precision: 'auto', 'float32', 'float16', 'bfloat16'
        **kwargs: Additional parameters passed to ModelConfig.

    Returns:
        ModelArtifacts container with loaded model, tokenizer, and metadata.
    """
    config = ModelConfig(
        model_id=model_id,
        device=device,
        precision=precision,
        **kwargs
    )
    loader = ModelLoader(config)
    return loader.load()
