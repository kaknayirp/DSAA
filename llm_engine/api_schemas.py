"""
Pydantic Request and Response Schemas for the DSAA LLM Serving API.

Module Owner: Person 1 — LLM & KV Cache Engineer.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Schema for text generation requests submitted to POST /generate."""
    prompt: str = Field(..., description="Input text prompt to be processed by the model.")
    model: Optional[str] = Field(None, description="Model identifier to verify compatibility.")
    max_new_tokens: int = Field(128, ge=1, le=4096, description="Maximum number of tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0 implies greedy decoding).")
    top_k: Optional[int] = Field(None, ge=1, description="Top-K sampling limit.")
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, description="Top-P nucleus sampling threshold.")
    do_sample: bool = Field(False, description="Whether to use stochastic sampling or greedy decoding.")
    stop_tokens: List[str] = Field(default_factory=list, description="List of string sequences that trigger generation halt.")
    stream: bool = Field(False, description="Whether to stream tokens via Server-Sent Events (SSE).")

    @field_validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or purely whitespace.")
        return v


class KVCacheSummary(BaseModel):
    """Summary of KV Cache memory consumption from Prompt 2's CacheReport."""
    total_bytes: int
    total_kb: float
    total_mb: float
    num_layers: int
    seq_len: int
    is_gqa: bool


class GenerateResponse(BaseModel):
    """Schema returned by POST /generate for non-streaming requests."""
    request_id: str
    model: str
    generated_text: str
    finish_reason: str
    num_prompt_tokens: int
    num_generated_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    kv_cache_summary: Optional[KVCacheSummary] = None


class RequestStatusResponse(BaseModel):
    """Schema returned by GET /requests/{request_id}."""
    request_id: str
    status: str
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    num_generated_tokens: int = 0
    error: Optional[str] = None


class CancelResponse(BaseModel):
    """Schema returned by POST /requests/{request_id}/cancel."""
    request_id: str
    cancelled: bool
    message: str


class HealthResponse(BaseModel):
    """Schema returned by GET /health."""
    status: str
    model_loaded: bool
    model_id: str
    device: str
    active_requests: int
    queued_requests: int
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Details for a candidate or active model."""
    model_id: str
    device: str
    precision: str
    context_length: int
    is_active: bool


class ModelListResponse(BaseModel):
    """Schema returned by GET /models."""
    models: List[ModelInfo]
    active_model: str


class ErrorResponse(BaseModel):
    """Standardized error response envelope."""
    error: str
    detail: Optional[str] = None
    code: int
