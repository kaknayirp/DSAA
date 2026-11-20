# DSAA LLM Engine (`llm_engine/`)

**Module Owner:** Person 1 — LLM & KV Cache Engineer  
**Project:** DSAA (Dynamic Smart Attention-based Allocation)

The `llm_engine/` module is the foundation of the DSAA system. It provides the complete LLM infrastructure: loading pretrained language models, managing hardware device and precision mapping, extracting and inspecting the Key-Value (KV) Cache, executing autoregressive inference, managing continuous batching request scheduling, and exposing the entire infrastructure through a robust, production-grade FastAPI serving layer.

---

## 1. Module Structure

```text
llm_engine/
├── __init__.py                     # Package exports for clean public API
├── config.py                       # ModelConfig, enums, device and precision resolution
├── generation_config.py            # GenerationConfig, sampling and stopping parameters
├── scheduler_config.py             # SchedulerConfig, batch sizes, scheduling policies, timeouts
├── server_config.py                # ServerConfig, host, port, CORS, request size/timeout limits
├── request_tracker.py              # TrackedRequest, RequestStatus state machine, cancellation
├── api_schemas.py                  # Pydantic schemas for requests, responses, and errors
├── api_app.py                      # FastAPI application, lifespan, endpoints, streaming SSE
├── exceptions.py                   # Custom error hierarchy (Model, Device, Cache, Inference, Scheduler)
├── model_loader.py                 # ModelLoader class and load_model_and_tokenizer helper
├── kv_extractor.py                 # KVCacheInspector, TensorStats, CacheReport, memory calculations
├── inference_engine.py             # InferenceEngine, InferenceRequest, GenerationState, InferenceResult
├── scheduler.py                    # RequestScheduler, BatchStepReport, iteration-level continuous batching
├── example_usage.py                # Standalone runnable model loader usage example
├── example_kv_inspection.py        # Standalone runnable KV Cache inspection & memory simulation
├── example_inference.py            # Standalone runnable autoregressive generation & cache reuse demo
├── example_continuous_batching.py  # Standalone runnable continuous batching dynamic entry/exit demo
├── example_server.py               # Standalone runnable API server & client demonstration
├── test_model_loader.py            # Unit tests for model loader (28 passing tests)
├── test_kv_extractor.py            # Unit tests for KV cache inspection & memory calculations (14 passing tests)
├── test_inference_engine.py        # Unit tests for inference engine & KV cache integration (18 passing tests)
├── test_scheduler.py               # Unit tests for continuous batching and scheduling (13 passing tests)
├── test_api.py                     # Unit and integration tests for serving API (11 passing tests)
└── README.md                       # Complete technical documentation and usage guide
```

---

## 2. Serving & Inference Architecture

The API layer is directly integrated with the existing continuous batching scheduler and inference engine:

```
  [ HTTP Client / Browser / Dashboard ]
                   │
                   ▼
       ┌───────────────────────┐
       │     FastAPI App       │  (api_app.py)
       │  CORS, Auth, Schemas  │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   RequestScheduler    │  (scheduler.py - Prompt 4)
       │  Continuous Batching  │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   InferenceEngine     │  (inference_engine.py - Prompt 3)
       │   Prefill & Decode    │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   KVCacheInspector    │  (kv_extractor.py - Prompt 2)
       │ Memory Profiling (MB) │
       └───────────┬───────────┘
                   │
                   ▼
       [ Structured HTTP Response / SSE Stream ]
```

---

## 3. REST API Endpoints

### `GET /health`
Returns health status, active and queued requests, uptime, and hardware device.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "cpu",
  "active_requests": 0,
  "queued_requests": 0,
  "uptime_seconds": 12.4
}
```

---

### `GET /models`
Returns list of active and supported candidate models.

**Response:**
```json
{
  "models": [
    {
      "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
      "device": "cpu",
      "precision": "auto",
      "context_length": 4096,
      "is_active": true
    },
    {
      "model_id": "google/gemma-2-2b-it",
      "device": "cpu",
      "precision": "auto",
      "context_length": 4096,
      "is_active": false
    }
  ],
  "active_model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

---

### `POST /generate`
Generates text from prompt. Supports blocking JSON or streaming Server-Sent Events (SSE).

**Request (JSON):**
```json
{
  "prompt": "Explain KV Cache in LLMs.",
  "max_new_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Response (Blocking JSON):**
```json
{
  "request_id": "47457df8",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "generated_text": "The KV Cache stores previous keys and values...",
  "finish_reason": "max_tokens",
  "num_prompt_tokens": 12,
  "num_generated_tokens": 64,
  "total_time_seconds": 0.42,
  "tokens_per_second": 152.38,
  "kv_cache_summary": {
    "total_bytes": 102400,
    "total_kb": 100.0,
    "total_mb": 0.0977,
    "num_layers": 4,
    "seq_len": 75,
    "is_gqa": true
  }
}
```

**Streaming (SSE - `stream: true`):**
```text
data: {"request_id": "e5707ad8", "token_id": 42, "text": "The"}

data: {"request_id": "e5707ad8", "token_id": 105, "text": " KV"}

data: {"request_id": "e5707ad8", "finish_reason": "completed", "num_generated_tokens": 2}

data: [DONE]
```

---

### `GET /requests/{request_id}`
Returns current status of a request: `PENDING`, `QUEUED`, `PREFILLING`, `DECODING`, `COMPLETED`, `FAILED`, or `CANCELLED`.

---

### `POST /requests/{request_id}/cancel`
Cancels a running or queued request, stops decoding, and immediately releases its KV Cache tensors.

---

## 4. Example `curl` Commands

```bash
# Health check
curl -X GET http://localhost:8000/health

# List models
curl -X GET http://localhost:8000/models

# Standard text generation
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is DSAA?", "max_new_tokens": 32, "temperature": 0.7}'

# Streaming text generation (SSE)
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain continuous batching.", "max_new_tokens": 32, "stream": true}'

# Request status
curl -X GET http://localhost:8000/requests/<request_id>

# Cancel request
curl -X POST http://localhost:8000/requests/<request_id>/cancel
```

---

## 5. Starting the Server

To launch the live Uvicorn HTTP server:

```bash
python -m llm_engine.example_server --serve
```

---

## 6. Running the Complete Test Suite

All 84 unit and integration tests across `llm_engine/` are strictly self-contained:

```bash
# Run all 84 tests across all 5 milestones
python -m unittest discover -s llm_engine -p "test_*.py" -v
```

### Breakdown by Component:
1. **Model Loader Tests (28 tests):** `python -m unittest llm_engine.test_model_loader -v`
2. **KV Cache Extractor Tests (14 tests):** `python -m unittest llm_engine.test_kv_extractor -v`
3. **Inference Engine Tests (18 tests):** `python -m unittest llm_engine.test_inference_engine -v`
4. **Scheduler & Continuous Batching Tests (13 tests):** `python -m unittest llm_engine.test_scheduler -v`
5. **Model Serving & API Tests (11 tests):** `python -m unittest llm_engine.test_api -v`

Total: **84 tests, 0 failures, 0 errors**.
