"""
Configuration dataclasses and validation logic for LLM model loading and inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from .exceptions import DeviceConfigurationError, PrecisionConfigurationError


class DeviceType(str, Enum):
    """Supported target execution devices."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class PrecisionMode(str, Enum):
    """Supported floating-point precision modes."""
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


SUPPORTED_CANDIDATE_MODELS: List[str] = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class ModelConfig:
    """
    Configuration parameters for model initialization, precision, device mapping,
    and tokenizer settings.
    """
    model_id: str = DEFAULT_MODEL_ID
    device: str = DeviceType.AUTO.value
    precision: str = PrecisionMode.AUTO.value
    use_cache: bool = True
    trust_remote_code: bool = True
    padding_side: str = "left"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    cache_dir: Optional[str] = None
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters upon initialization."""
        if not self.model_id or not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("ModelConfig 'model_id' must be a non-empty string.")

        self.model_id = self.model_id.strip()

        # Validate device option
        valid_devices = {d.value for d in DeviceType}
        # Also allow cuda:0, cuda:1, etc.
        device_lower = self.device.lower().strip()
        is_cuda_idx = device_lower.startswith("cuda:") and device_lower.split(":")[-1].isdigit()
        if device_lower not in valid_devices and not is_cuda_idx:
            raise DeviceConfigurationError(
                f"Invalid device '{self.device}'. Supported options are: {sorted(list(valid_devices))} "
                f"or 'cuda:<index>'."
            )
        self.device = device_lower

        # Validate precision option
        valid_precisions = {p.value for p in PrecisionMode}
        precision_lower = self.precision.lower().strip()
        if precision_lower not in valid_precisions:
            raise PrecisionConfigurationError(
                f"Invalid precision '{self.precision}'. Supported options are: {sorted(list(valid_precisions))}."
            )
        self.precision = precision_lower

        # Validate padding side (decoder-only autoregressive generation requires left padding)
        padding_side_lower = self.padding_side.lower().strip()
        if padding_side_lower not in {"left", "right"}:
            raise ValueError(f"Invalid padding_side '{self.padding_side}'. Supported options are: 'left', 'right'.")
        self.padding_side = padding_side_lower


def resolve_device(requested_device: str, torch_module: Any = None) -> str:
    """
    Determine the concrete execution device (e.g. 'cuda' or 'cpu') based on request
    and available hardware.

    Args:
        requested_device: 'auto', 'cuda', 'cpu', or 'cuda:<idx>'
        torch_module: Optional imported torch module (or mock).

    Returns:
        Resolved device string ('cuda', 'cuda:<idx>', or 'cpu').
    """
    if torch_module is None:
        try:
            import torch
            torch_module = torch
        except ImportError:
            torch_module = None

    req = requested_device.lower().strip()

    if req == DeviceType.AUTO.value:
        if torch_module is not None and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    if req.startswith("cuda"):
        if torch_module is None:
            raise DeviceConfigurationError("PyTorch is not available to verify CUDA support.")
        if not (hasattr(torch_module, "cuda") and torch_module.cuda.is_available()):
            raise DeviceConfigurationError(
                f"CUDA device '{req}' was explicitly requested, but CUDA is not available on this machine."
            )
        return req

    if req == DeviceType.CPU.value:
        return "cpu"

    return req


def resolve_precision(
    requested_precision: str,
    resolved_device: str,
    torch_module: Any = None
) -> Any:
    """
    Determine the safe and optimal torch.dtype for the resolved device.

    Rules:
    - If precision is 'auto':
      - CUDA with BF16 support -> torch.bfloat16
      - CUDA without BF16 support -> torch.float16
      - CPU -> torch.float32 (safe default for CPU inference)
    - If explicit precision requested:
      - 'bfloat16': validated; on CPU or non-bf16 CUDA, raises or returns torch.bfloat16 if PyTorch supports it.
      - 'float16': on CPU, warns or raises if unsupported.
      - 'float32': universally supported.

    Args:
        requested_precision: 'auto', 'float32', 'float16', 'bfloat16'
        resolved_device: 'cuda', 'cuda:<idx>', or 'cpu'
        torch_module: Optional imported torch module (or mock).

    Returns:
        torch.dtype object corresponding to selected precision.
    """
    if torch_module is None:
        try:
            import torch
            torch_module = torch
        except ImportError:
            torch_module = None

    req = requested_precision.lower().strip()
    is_cuda = resolved_device.startswith("cuda")

    if req == PrecisionMode.AUTO.value:
        if is_cuda and torch_module is not None:
            # Check if BF16 is supported on this GPU
            bf16_supported = False
            if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "is_bf16_supported"):
                try:
                    bf16_supported = torch_module.cuda.is_bf16_supported()
                except Exception:
                    bf16_supported = False
            if bf16_supported:
                return torch_module.bfloat16 if torch_module else "bfloat16"
            return torch_module.float16 if torch_module else "float16"
        else:
            # CPU fallback: default to float32 for numerical stability and broad CPU kernel support
            return torch_module.float32 if torch_module else "float32"

    if req == PrecisionMode.FLOAT32.value:
        return torch_module.float32 if torch_module else "float32"

    if req == PrecisionMode.FLOAT16.value:
        if not is_cuda and torch_module is not None:
            # Most CPU backends do not have optimized float16 kernels and can cause slow execution or errors
            raise PrecisionConfigurationError(
                "FP16 (float16) is not safely supported for standard CPU inference in PyTorch. "
                "Please use 'float32', 'bfloat16' (if CPU supports it), or 'auto'."
            )
        return torch_module.float16 if torch_module else "float16"

    if req == PrecisionMode.BFLOAT16.value:
        if is_cuda and torch_module is not None:
            if hasattr(torch_module.cuda, "is_bf16_supported") and not torch_module.cuda.is_bf16_supported():
                raise PrecisionConfigurationError(
                    "bfloat16 precision requested, but the active CUDA device does not support BF16 hardware instructions."
                )
        return torch_module.bfloat16 if torch_module else "bfloat16"

    raise PrecisionConfigurationError(f"Unsupported precision configuration: '{requested_precision}'.")
