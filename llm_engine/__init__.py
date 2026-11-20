"""
DSAA LLM Engine Module.

Owned by Person 1 — LLM & KV Cache Engineer.
Provides model loading, configuration, device/precision management,
KV Cache extraction, inspection, autoregressive inference execution,
continuous batching request scheduling, and FastAPI model serving layer.
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
from .scheduler_config import SchedulerConfig, SchedulingPolicy
from .server_config import ServerConfig
from .request_tracker import RequestStatus, TrackedRequest
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
    SchedulerError,
    QueueFullError,
    RequestNotFoundError,
    RequestCancelledError,
    RequestTimeoutError,
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
from .scheduler import (
    RequestScheduler,
    BatchStepReport,
)
from .api_schemas import (
    GenerateRequest,
    GenerateResponse,
    RequestStatusResponse,
    CancelResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    KVCacheSummary,
    ErrorResponse,
)
from .api_app import create_app

__all__ = [
    # Config
    "ModelConfig",
    "GenerationConfig",
    "SchedulerConfig",
    "ServerConfig",
    "SchedulingPolicy",
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
    "SchedulerError",
    "QueueFullError",
    "RequestNotFoundError",
    "RequestCancelledError",
    "RequestTimeoutError",
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
    # Continuous Batching Scheduler
    "RequestScheduler",
    "BatchStepReport",
    "RequestStatus",
    "TrackedRequest",
    # Serving & API Layer
    "create_app",
    "GenerateRequest",
    "GenerateResponse",
    "RequestStatusResponse",
    "CancelResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelListResponse",
    "KVCacheSummary",
    "ErrorResponse",
]
