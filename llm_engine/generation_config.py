"""
Generation configuration for the DSAA LLM Inference Engine.

Encapsulates decoding settings: sampling strategies, stopping conditions,
token limits, and cache capacity constraints.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union

from .exceptions import GenerationConfigError


@dataclass
class GenerationConfig:
    """
    Configuration controlling autoregressive token generation.
    """
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = False
    eos_token_id: Optional[Union[int, List[int]]] = None
    stop_token_ids: List[int] = field(default_factory=list)
    stop_token_sequences: List[List[int]] = field(default_factory=list)
    max_cache_capacity: Optional[int] = None
    repetition_penalty: float = 1.0
    return_cache_report: bool = True

    def __post_init__(self) -> None:
        """Validate generation parameters to ensure numerical stability and correct decoding."""
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise GenerationConfigError(
                f"max_new_tokens must be a positive integer, got {self.max_new_tokens}"
            )

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0.0:
            raise GenerationConfigError(
                f"temperature must be a non-negative float, got {self.temperature}"
            )

        if self.top_k is not None:
            if not isinstance(self.top_k, int) or self.top_k <= 0:
                raise GenerationConfigError(
                    f"top_k must be a positive integer, got {self.top_k}"
                )

        if self.top_p is not None:
            if not isinstance(self.top_p, (int, float)) or not (0.0 < self.top_p <= 1.0):
                raise GenerationConfigError(
                    f"top_p must be a float in the range (0.0, 1.0], got {self.top_p}"
                )

        if self.max_cache_capacity is not None:
            if not isinstance(self.max_cache_capacity, int) or self.max_cache_capacity <= 0:
                raise GenerationConfigError(
                    f"max_cache_capacity must be a positive integer, got {self.max_cache_capacity}"
                )

        if not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty <= 0.0:
            raise GenerationConfigError(
                f"repetition_penalty must be a positive float, got {self.repetition_penalty}"
            )

        # Normalize eos_token_id into a list for fast membership checks
        if self.eos_token_id is not None:
            if isinstance(self.eos_token_id, int):
                self._resolved_eos_ids = [self.eos_token_id]
            elif isinstance(self.eos_token_id, (list, tuple, set)):
                self._resolved_eos_ids = [int(tid) for tid in self.eos_token_id]
            else:
                raise GenerationConfigError(f"Unsupported type for eos_token_id: {type(self.eos_token_id)}")
        else:
            self._resolved_eos_ids = []

        # Merge stop token ids
        all_stops = set(self._resolved_eos_ids)
        if self.stop_token_ids:
            for st in self.stop_token_ids:
                all_stops.add(int(st))
        self.stop_token_set = all_stops
        self.stop_token_sequences = [
            [int(token_id) for token_id in sequence]
            for sequence in self.stop_token_sequences
            if sequence
        ]
