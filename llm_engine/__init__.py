"""
DSAA LLM Engine Module.

Owned by Person 1 — LLM & KV Cache Engineer.
Provides model loading, configuration, device/precision management,
KV Cache extraction, and cache management.
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
from .exceptions import (
    LLMEngineError,
    DependencyMissingError,
    DeviceConfigurationError,
    PrecisionConfigurationError,
    ModelDownloadError,
    ModelInitializationError,
    TokenizerInitializationError,
    ModelLoaderError,
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

__all__ = [
    "ModelConfig",
    "DeviceType",
    "PrecisionMode",
    "DEFAULT_MODEL_ID",
    "SUPPORTED_CANDIDATE_MODELS",
    "resolve_device",
    "resolve_precision",
    "LLMEngineError",
    "DependencyMissingError",
    "DeviceConfigurationError",
    "PrecisionConfigurationError",
    "ModelDownloadError",
    "ModelInitializationError",
    "TokenizerInitializationError",
    "ModelLoaderError",
    "ModelLoader",
    "ModelArtifacts",
    "load_model_and_tokenizer",
    "KVCacheInspector",
    "CacheReport",
    "LayerCacheStats",
    "TensorStats",
    "KVCacheInspectionError",
    "extract_kv_cache",
    "run_inference_and_inspect_cache",
]
