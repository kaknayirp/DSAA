"""
Custom exception hierarchy for the DSAA LLM Engine.

These exceptions provide clear, descriptive error messages for dependency,
hardware, network, precision, and model loading failures.
"""


class LLMEngineError(Exception):
    """Base exception for all errors originating within the llm_engine module."""
    pass


class ModelLoaderError(LLMEngineError):
    """Generic error raised when model loader operations fail."""
    pass


class DependencyMissingError(LLMEngineError):
    """Raised when required runtime libraries (e.g., PyTorch, Transformers) are not installed."""
    def __init__(self, library_name: str, install_command: str = ""):
        message = (
            f"Required library '{library_name}' is not installed in the current environment.\n"
            f"To resolve, please run: {install_command or f'pip install {library_name}'}"
        )
        super().__init__(message)
        self.library_name = library_name
        self.install_command = install_command


class DeviceConfigurationError(LLMEngineError):
    """Raised when an invalid device is requested or hardware requirements are not met."""
    pass


class PrecisionConfigurationError(LLMEngineError):
    """Raised when an unsupported precision or data type configuration is specified."""
    pass


class ModelDownloadError(LLMEngineError):
    """Raised when downloading model weights or tokenizer files from Hugging Face Hub fails."""
    pass


class ModelInitializationError(LLMEngineError):
    """Raised when model weights cannot be loaded into memory (e.g. OOM, corrupt weights)."""
    pass


class TokenizerInitializationError(LLMEngineError):
    """Raised when tokenizer initialization fails or special tokens cannot be configured."""
    pass


class InferenceEngineError(LLMEngineError):
    """Base exception for errors during inference execution or autoregressive generation."""
    pass


class InvalidInputError(InferenceEngineError):
    """Raised when an empty, malformed, or out-of-bounds input sequence is provided."""
    pass


class GenerationConfigError(InferenceEngineError):
    """Raised when invalid generation parameters (e.g., negative temperature, top_p > 1) are passed."""
    pass


class CacheStateError(InferenceEngineError):
    """Raised when the KV Cache state is missing, corrupted, or violates model invariants."""
    pass


class SchedulerError(InferenceEngineError):
    """Base exception for request scheduling and continuous batching errors."""
    pass


class QueueFullError(SchedulerError):
    """Raised when the scheduler request queue exceeds its maximum capacity."""
    pass


class RequestNotFoundError(SchedulerError):
    """Raised when an operation references a non-existent request ID."""
    pass


class RequestCancelledError(SchedulerError):
    """Raised when attempting to execute or retrieve results for a cancelled request."""
    pass


class RequestTimeoutError(SchedulerError):
    """Raised when a request exceeds its maximum execution or waiting timeout."""
    pass


