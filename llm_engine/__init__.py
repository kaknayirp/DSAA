"""
DSAA LLM Engine Module.

Owned by Person 1 — LLM & KV Cache Engineer.
Provides model loading, configuration, device/precision management,
KV Cache extraction, inspection, and autoregressive inference execution.
"""

from .config import (
    ModelConfig,
    DeviceType,
    PrecisionMode,
    DEFAULT_MODEL_ID,
    SUPPORTED_CANDIDATE_MODELS,
    resolve_device,
    resolve_precision,
)
from .generation_config import GenerationConfig
from .exceptions import (
    LLMEngineError,
    DependencyMissingError,
    DeviceConfigurationError,
    PrecisionConfigurationError,
    ModelDownloadError,
    ModelInitializationError,
    TokenizerInitializationError,
    ModelLoaderError,
    InferenceEngineError,
    InvalidInputError,
    GenerationConfigError,
    CacheStateError,
)
from .model_loader import (
    ModelLoader,
    ModelArtifacts,
    load_model_and_tokenizer,
)
from .kv_extractor import (
    KVCacheInspector,
    CacheReport,
    LayerCacheStats,
    TensorStats,
    KVCacheInspectionError,
    extract_kv_cache,
    run_inference_and_inspect_cache,
)
from .inference_engine import (
    InferenceEngine,
    InferenceRequest,
    GenerationState,
    InferenceResult,
)

__all__ = [
    # Config
    "ModelConfig",
    "GenerationConfig",
    "DeviceType",
    "PrecisionMode",
    "DEFAULT_MODEL_ID",
    "SUPPORTED_CANDIDATE_MODELS",
    "resolve_device",
    "resolve_precision",
    # Exceptions
    "LLMEngineError",
    "DependencyMissingError",
    "DeviceConfigurationError",
    "PrecisionConfigurationError",
    "ModelDownloadError",
    "ModelInitializationError",
    "TokenizerInitializationError",
    "ModelLoaderError",
    "InferenceEngineError",
    "InvalidInputError",
    "GenerationConfigError",
    "CacheStateError",
    # Model Loading
    "ModelLoader",
    "ModelArtifacts",
    "load_model_and_tokenizer",
    # KV Cache Inspection
    "KVCacheInspector",
    "CacheReport",
    "LayerCacheStats",
    "TensorStats",
    "KVCacheInspectionError",
    "extract_kv_cache",
    "run_inference_and_inspect_cache",
    # Inference Engine
    "InferenceEngine",
    "InferenceRequest",
    "GenerationState",
    "InferenceResult",
]
