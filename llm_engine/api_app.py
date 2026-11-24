"""
FastAPI Application and Model Serving Layer for DSAA.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Exposes REST and SSE streaming endpoints for the DSAA LLM engine:
- GET  /health               Server and infrastructure health status
- GET  /models               Model configuration and capabilities
- POST /generate             Text generation (blocking JSON or streaming SSE)
- GET  /requests/{id}        Request execution status
- POST /requests/{id}/cancel Cancel running or queued request
"""

import time
import json
import queue
import logging
import threading
from typing import Optional, Dict, Any, Generator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import SUPPORTED_CANDIDATE_MODELS, DEFAULT_MODEL_ID
from .server_config import ServerConfig
from .generation_config import GenerationConfig
from .scheduler import RequestScheduler
from .request_tracker import RequestStatus
from .inference_engine import InferenceEngine, InferenceRequest
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
from .exceptions import (
    LLMEngineError,
    QueueFullError,
    RequestNotFoundError,
    RequestCancelledError,
    RequestTimeoutError,
    InvalidInputError,
    GenerationConfigError,
    InferenceEngineError,
    SchedulerError,
)

logger = logging.getLogger(__name__)


class BackgroundSchedulerWorker:
    """
    Background worker thread that drives scheduler.step() continuously
    whenever requests are queued or actively decoding.
    """
    def __init__(self, scheduler: RequestScheduler, poll_interval: float = 0.005):
        self.scheduler = scheduler
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SchedulerWorker")
        self._thread.start()
        logger.info("BackgroundSchedulerWorker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("BackgroundSchedulerWorker stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            has_work = (self.scheduler.queued_count > 0 or self.scheduler.active_count > 0)
            if has_work:
                try:
                    self.scheduler.step()
                except Exception as exc:
                    logger.error("Error in scheduler step: %s", exc)
                time.sleep(self.poll_interval)
            else:
                time.sleep(0.01)


def create_app(
    scheduler: RequestScheduler,
    server_config: Optional[ServerConfig] = None,
    start_background_worker: bool = True,
) -> FastAPI:
    """
    Application factory building the FastAPI model serving app.
    Integrates directly with the existing RequestScheduler and InferenceEngine.
    """
    cfg = server_config or ServerConfig()
    worker = BackgroundSchedulerWorker(scheduler) if start_background_worker else None
    start_time = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if worker:
            worker.start()
        yield
        # Shutdown
        if worker:
            worker.stop()
        scheduler.clear()

    app = FastAPI(
        title="DSAA LLM Serving API",
        description="High-performance model serving with Dynamic Smart Attention-based Allocation & Continuous Batching.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API Key Authentication Middleware
    @app.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next):
        if cfg.api_key is not None:
            # Exempt health and docs
            if request.url.path not in ["/health", "/docs", "/openapi.json", "/redoc"]:
                auth_header = request.headers.get("X-API-Key") or request.headers.get("Authorization")
                expected = f"Bearer {cfg.api_key}" if auth_header and "Bearer" in auth_header else cfg.api_key
                if auth_header != expected:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"error": "Unauthorized", "detail": "Invalid or missing API key.", "code": 401},
                    )
        return await call_next(request)

    # --------------------------------------------------------------------------
    # Exception Handlers
    # --------------------------------------------------------------------------

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "QueueFull", "detail": str(exc), "code": 429},
        )

    @app.exception_handler(RequestNotFoundError)
    async def not_found_handler(request: Request, exc: RequestNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "NotFound", "detail": str(exc), "code": 404},
        )

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "InvalidInput", "detail": str(exc), "code": 400},
        )

    @app.exception_handler(GenerationConfigError)
    async def config_error_handler(request: Request, exc: GenerationConfigError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "InvalidConfiguration", "detail": str(exc), "code": 400},
        )

    @app.exception_handler(RequestCancelledError)
    async def cancelled_handler(request: Request, exc: RequestCancelledError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "RequestCancelled", "detail": str(exc), "code": 400},
        )

    @app.exception_handler(RequestTimeoutError)
    async def timeout_handler(request: Request, exc: RequestTimeoutError):
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "RequestTimeout", "detail": str(exc), "code": 504},
        )

    @app.exception_handler(InferenceEngineError)
    async def inference_error_handler(request: Request, exc: InferenceEngineError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InferenceExecutionFailure", "detail": str(exc), "code": 500},
        )

    @app.exception_handler(LLMEngineError)
    async def llm_engine_error_handler(request: Request, exc: LLMEngineError):
        """Return a structured response for loader, device, cache, and engine errors."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "LLMEngineFailure", "detail": str(exc), "code": 500},
        )

    # --------------------------------------------------------------------------
    # REST Endpoints
    # --------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def get_health() -> HealthResponse:
        """Return server health status, active request counts, and hardware device info."""
        engine = scheduler.engine
        artifacts = getattr(engine, "_artifacts", None)
        model_id = artifacts.config.model_id if artifacts else DEFAULT_MODEL_ID
        device_str = engine.device if artifacts else "unknown"

        return HealthResponse(
            status="healthy",
            model_loaded=engine.is_model_loaded,
            model_id=model_id,
            device=device_str,
            active_requests=scheduler.active_count,
            queued_requests=scheduler.queued_count,
            uptime_seconds=round(time.time() - start_time, 2),
        )

    @app.get("/models", response_model=ModelListResponse, tags=["Models"])
    def list_models() -> ModelListResponse:
        """Return information about the currently active model and supported candidates."""
        engine = scheduler.engine
        artifacts = getattr(engine, "_artifacts", None)
        active_id = artifacts.config.model_id if artifacts else DEFAULT_MODEL_ID
        device = engine.device if artifacts else "cpu"
        precision = artifacts.config.precision if artifacts else "auto"

        models = [
            ModelInfo(
                model_id=active_id,
                device=device,
                precision=precision,
                context_length=scheduler.config.max_sequence_length,
                is_active=True,
            )
        ]

        # Add other supported candidate models
        for cand in SUPPORTED_CANDIDATE_MODELS:
            if cand != active_id:
                models.append(
                    ModelInfo(
                        model_id=cand,
                        device=device,
                        precision="auto",
                        context_length=scheduler.config.max_sequence_length,
                        is_active=False,
                    )
                )

        return ModelListResponse(models=models, active_model=active_id)

    @app.post("/generate", tags=["Inference"])
    def generate_text(req: GenerateRequest):
        """
        Generate text from prompt. Supports standard blocking JSON response
        or streaming Server-Sent Events (SSE) based on `stream: true`.
        """
        # Validate prompt length
        if len(req.prompt) > cfg.max_input_length:
            raise InvalidInputError(
                f"Prompt length ({len(req.prompt)} chars) exceeds maximum allowed ({cfg.max_input_length})."
            )

        gen_config = GenerationConfig(
            max_new_tokens=min(req.max_new_tokens, cfg.max_generation_length),
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            do_sample=req.do_sample,
            return_cache_report=True,
        )
        if req.stop_tokens:
            tokenizer = scheduler.engine.tokenizer
            for stop_text in req.stop_tokens:
                encoded_stop = tokenizer(stop_text, add_special_tokens=False, return_tensors="pt")
                token_sequence = scheduler.engine._to_token_list(encoded_stop["input_ids"])
                if token_sequence:
                    gen_config.stop_token_sequences.append(token_sequence)

        # 1. Streaming response (Server-Sent Events)
        if req.stream:
            if not cfg.enable_streaming:
                raise InvalidInputError("Streaming is disabled by server configuration.")

            stream_q: queue.Queue = queue.Queue()
            request_id = scheduler.submit(
                prompt_or_request=req.prompt,
                generation_config=gen_config,
                stream_queue=stream_q,
            )

            def event_generator() -> Generator[str, None, None]:
                try:
                    while True:
                        # If no background thread is actively stepping, step synchronously
                        if worker is None or not worker._thread or not worker._thread.is_alive():
                            if scheduler.queued_count > 0 or scheduler.active_count > 0:
                                scheduler.step()

                        try:
                            # Wait for next token chunk
                            poll_timeout = 0.05 if (worker is None or not worker._thread or not worker._thread.is_alive()) else cfg.request_timeout_seconds
                            chunk = stream_q.get(timeout=poll_timeout)
                        except queue.Empty:
                            tracked = scheduler.get_status(request_id)
                            if tracked and tracked.is_terminal:
                                break
                            continue

                        event_type = chunk.get("type", "token")
                        if event_type in {"error", "cancelled", "complete"}:
                            if event_type == "error":
                                yield f"data: {json.dumps({'error': chunk['error'], 'code': 500})}\n\n"
                            elif event_type == "cancelled":
                                yield f"data: {json.dumps({'cancelled': True, 'finish_reason': 'cancelled'})}\n\n"
                            else:
                                res = chunk.get("result")
                                final_payload = {
                                    "request_id": request_id,
                                    "finish_reason": res.finish_reason,
                                    "num_prompt_tokens": res.num_prompt_tokens,
                                    "num_generated_tokens": res.num_generated_tokens,
                                    "total_time_seconds": round(res.total_time_seconds, 4),
                                    "tokens_per_second": round(res.tokens_per_second, 2),
                                }
                                yield f"data: {json.dumps(final_payload)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        # Stream individual token chunk
                        payload = {
                            "request_id": request_id,
                            "token_id": chunk.get("token_id"),
                            "text": chunk.get("text", ""),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                except GeneratorExit:
                    # Client disconnected: cancel request in scheduler to free KV Cache!
                    logger.info("Client disconnected during streaming of '%s'. Cancelling.", request_id)
                    scheduler.cancel(request_id)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # 2. Standard blocking JSON response
        request_id = scheduler.submit(
            prompt_or_request=req.prompt,
            generation_config=gen_config,
        )

        # If background worker is not active, drive step synchronously
        if worker is None or not worker._thread or not worker._thread.is_alive():
            scheduler.run_until_complete(timeout=cfg.request_timeout_seconds)

        result = scheduler.get_result(request_id, timeout=cfg.request_timeout_seconds)
        if result is None:
            raise InferenceEngineError("Inference finished without returning an InferenceResult.")

        # Build KV Cache summary if present from Prompt 2's CacheReport
        kv_summary = None
        if result.final_cache_report is not None:
            rep = result.final_cache_report
            kv_summary = KVCacheSummary(
                total_bytes=rep.total_bytes,
                total_kb=round(rep.total_kb, 2),
                total_mb=round(rep.total_mb, 4),
                num_layers=rep.num_layers,
                seq_len=rep.seq_len,
                is_gqa=rep.is_gqa,
            )

        artifacts = getattr(scheduler.engine, "_artifacts", None)
        active_model = artifacts.config.model_id if artifacts else DEFAULT_MODEL_ID

        return GenerateResponse(
            request_id=result.request_id,
            model=active_model,
            generated_text=result.decoded_text or "",
            finish_reason=result.finish_reason,
            num_prompt_tokens=result.num_prompt_tokens,
            num_generated_tokens=result.num_generated_tokens,
            total_time_seconds=round(result.total_time_seconds, 4),
            tokens_per_second=round(result.tokens_per_second, 2),
            kv_cache_summary=kv_summary,
        )

    @app.get("/requests/{request_id}", response_model=RequestStatusResponse, tags=["Inference"])
    def get_request_status(request_id: str) -> RequestStatusResponse:
        """Query current execution state and token progress of an inference request."""
        tracked = scheduler.get_status(request_id)
        if tracked is None:
            raise RequestNotFoundError(f"Request '{request_id}' not found.")

        if tracked.result is not None:
            num_tokens = tracked.result.num_generated_tokens
        elif tracked.state is not None:
            num_tokens = len(tracked.state.generated_token_ids)
        else:
            num_tokens = 0
        return RequestStatusResponse(
            request_id=tracked.request_id,
            status=tracked.status.value,
            submitted_at=tracked.submitted_at,
            started_at=tracked.started_at,
            completed_at=tracked.completed_at,
            num_generated_tokens=num_tokens,
            error=tracked.error_message,
        )

    @app.post("/requests/{request_id}/cancel", response_model=CancelResponse, tags=["Inference"])
    def cancel_request(request_id: str) -> CancelResponse:
        """Cancel a pending, queued, or actively decoding request and release its KV Cache."""
        tracked = scheduler.get_status(request_id)
        if tracked is None:
            raise RequestNotFoundError(f"Request '{request_id}' not found.")

        cancelled = scheduler.cancel(request_id)
        msg = "Request successfully cancelled and cache released." if cancelled else "Request was already completed or cancelled."
        return CancelResponse(
            request_id=request_id,
            cancelled=cancelled,
            message=msg,
        )

    return app
