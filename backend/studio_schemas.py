from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Strategy = Literal["original", "h2o", "snapkv", "dsaa"]
STRATEGIES = ("original", "h2o", "snapkv", "dsaa")


class RunSettings(BaseModel):
    attention: float = Field(default=0.5, ge=0, le=1)
    semantic: float = Field(default=0.3, ge=0, le=1)
    recency: float = Field(default=0.2, ge=0, le=1)
    retention_budget: int = Field(default=512, ge=1, le=32768)
    eviction_threshold: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def check_weights(self):
        if not abs(self.attention + self.semantic + self.recency - 1) < 1e-6:
            raise ValueError("Attention, semantic and recency weights must sum to 1.")
        return self


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Enter a prompt.")
        return prompt


class InferenceRequest(CompareRequest):
    strategy: Strategy = "original"


class CacheMetrics(BaseModel):
    kv_cache_bytes: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    peak_ram_bytes: int | None = Field(default=None, ge=0)
    first_token_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tokens_per_second: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    answer_accuracy: float | None = Field(default=None, ge=0, le=1)
    answer: str | None = None


class CacheEntry(BaseModel):
    position: int = Field(ge=0)
    token: str
    retained: bool
    token_id: int | None = None
    layer: int | None = None
    head: int | None = None
    attention: float | None = None
    semantic: float | None = None
    recency: float | None = None
    score: float | None = None
    reason: str | None = None
    saved_bytes: int | None = Field(default=None, ge=0)


class InferenceResult(BaseModel):
    metrics: CacheMetrics
    entries: list[CacheEntry] = Field(default_factory=list)
    context_tokens: int | None = Field(default=None, ge=1)
    memory_pressure: float | None = Field(default=None, ge=0, le=1)


class BenchmarkRequest(BaseModel):
    dataset: Literal["recall"] = "recall"
    strategies: list[Strategy] = Field(min_length=1, max_length=4)
    case_count: int = Field(default=3, ge=1, le=3)
    context_blocks: int = Field(default=8, ge=1, le=64)
    max_new_tokens: int = Field(default=32, ge=1, le=256)

    @field_validator("strategies")
    @classmethod
    def unique_strategies(cls, strategies):
        if len(set(strategies)) != len(strategies):
            raise ValueError("Choose each strategy at most once.")
        return strategies
