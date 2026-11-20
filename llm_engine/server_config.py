"""
Server configuration for the DSAA LLM API and serving layer.

Module Owner: Person 1 — LLM & KV Cache Engineer.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .exceptions import GenerationConfigError


@dataclass
class ServerConfig:
    """
    Configuration controlling HTTP serving, CORS, request validation limits,
    and optional API key authentication.
    """
    host: str = "0.0.0.0"
    port: int = 8000
    max_input_length: int = 4096
    max_generation_length: int = 2048
    request_timeout_seconds: float = 60.0
    enable_streaming: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate server configuration parameters."""
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise GenerationConfigError(f"port must be an integer between 1 and 65535, got {self.port}")

        if not isinstance(self.max_input_length, int) or self.max_input_length <= 0:
            raise GenerationConfigError(f"max_input_length must be positive integer, got {self.max_input_length}")

        if not isinstance(self.max_generation_length, int) or self.max_generation_length <= 0:
            raise GenerationConfigError(f"max_generation_length must be positive integer, got {self.max_generation_length}")

        if not isinstance(self.request_timeout_seconds, (int, float)) or self.request_timeout_seconds <= 0:
            raise GenerationConfigError(f"request_timeout_seconds must be positive float, got {self.request_timeout_seconds}")
