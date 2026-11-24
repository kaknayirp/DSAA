# DSAA LLM Engine (`llm_engine/`)

**Module Owner:** Person 1 — LLM & KV Cache Engineer  
**Project:** DSAA (Dynamic Smart Attention-based Allocation)

The `llm_engine/` module connects model loading, autoregressive inference, per-request KV Cache management, request scheduling, and a FastAPI serving layer. The API and scheduler are tested with deterministic mocks. The optional real-Qwen test requires PyTorch, Transformers, cached model files, and a functioning local event loop. The scheduler interleaves requests but currently executes each request's model forward pass serially; it does not form tensor batches.

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
├── test_model_loader.py            # Model loader tests (mocked dependencies)
├── test_kv_extractor.py             # Cache inspection tests (synthetic tensors)
├── test_inference_engine.py        # Inference tests (mock model and tokenizer)
├── test_scheduler.py               # Scheduler tests (mock model and tokenizer)
├── test_api.py                     # API and SSE tests (mock model and tokenizer)
├── test_real_model_integration.py  # Opt-in local Qwen integration test
└── README.md                       # Usage, architecture, verification, and limitations
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

All response values below are illustrative schema examples, not measured output.

**Response (Blocking JSON):**
```json
{
    "request_id": "<request-id>",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "generated_text": "The KV Cache stores previous keys and values...",
  "finish_reason": "max_tokens",
  "num_prompt_tokens": 12,
  "num_generated_tokens": 64,
  "total_time_seconds": 0.42,
  "tokens_per_second": 152.38,
  "kv_cache_summary": {
    "total_bytes": 4300800,
    "total_kb": 4200.0,
    "total_mb": 4.1016,
    "num_layers": 28,
    "seq_len": 75,
    "is_gqa": true
  }
}
```

**Streaming (SSE - `stream: true`; schematic placeholders):**
```text
data: {"request_id": "<request-id>", "token_id": <model-token-id>, "text": "<decoded text delta>"}

data: {"request_id": "<request-id>", "token_id": <model-token-id>, "text": "<decoded text delta>"}

data: {"request_id": "<request-id>", "finish_reason": "max_tokens", "num_prompt_tokens": 8, "num_generated_tokens": 2, "total_time_seconds": 0.12, "tokens_per_second": 16.67}

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
python -m llm_engine.example_server
```

This command starts the API with the configured pretrained model loader. Model and
tokenizer weights are loaded on the first inference request. It does not use the
test-only `StatefulCausalLM` or `MockTokenizer`. Run `python -m
llm_engine.example_server --demo` to exercise the HTTP flow from the demonstration
client; this also uses the configured pretrained model.

---

## 6. Installation and Model Setup

Use a 64-bit Python environment and install PyTorch for the desired compute platform. For Windows CPU-only setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, first confirm an NVIDIA GPU and driver with `nvidia-smi`, then choose the matching Windows/Pip/CUDA command from the official PyTorch installer selector. The repository requirements are currently unpinned, so installs are not version-locked.

The default server model is `Qwen/Qwen2.5-1.5B-Instruct`. `ModelLoader.load()` downloads the tokenizer and weights on first use and caches them in the Hugging Face cache. This requires network access on first load and enough system memory for the chosen precision. The current Python model loader uses float32 on CPU by default; available RAM was not measured as part of the unit suite.

## 7. Execution Flow and Module Responsibilities

```text
HTTP request -> FastAPI schema/auth -> RequestScheduler.submit()
 -> InferenceEngine tokenizer/prefill -> model forward(use_cache=True)
 -> token selection -> decode_step(token + past_key_values)
 -> scheduler completion -> JSON response or SSE events
```

- `model_loader.py`: loads pretrained weights and tokenizer, sets evaluation mode, disables gradients, and enables model cache use.
- `inference_engine.py`: prefill, sampling, stopping, incremental cache passing/updating, timing, and result construction.
- `kv_extractor.py`: reads cache tensor shape/dtype/storage metadata. Logical cache bytes are computed from tensors; this is not process RSS or total GPU memory.
- `scheduler.py` and `request_tracker.py`: queueing, admission, per-request state, cancellation, and request-level cache cleanup.
- `api_app.py` and `api_schemas.py`: health/model endpoints, blocking generation, SSE, request status/cancel, input validation, and optional API-key middleware.

**Batching limitation:** `max_batch_size` controls how many requests can be active. `RequestScheduler.step()` invokes `engine.prefill()` or `engine.decode_step()` separately for each request. Requests share scheduling iterations, not one batched tensor forward pass. Do not interpret this as GPU tensor batching or throughput batching.

## 8. API Usage

Start the server with `python -m llm_engine.example_server`. It loads the real model on the first generation request. Before that request, `/health` can correctly report `model_loaded: false`.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/models
curl.exe -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"prompt":"Explain KV caching briefly.","max_new_tokens":32}'
curl.exe -N -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"prompt":"Explain KV caching briefly.","max_new_tokens":32,"stream":true}'
```

Streaming emits token events, then a completion event with `finish_reason`, counts, and measured timing fields, followed by `data: [DONE]`. An EOS token can have empty visible text because special tokens are skipped during decoding.

## 9. Test Suite

Run the complete test suite from the repository root:

```bash
# Run unit/API tests plus the opt-in real-model test (skipped by default)
python -m unittest discover -s llm_engine -p "test_*.py" -v
```

The latest full run was **88 run, 0 failures, 1 skipped** (87 passed). The skip is `test_real_model_integration.py`; it requires `DSAA_RUN_REAL_MODEL_TESTS=1` and locally cached model files. The 87 passing tests use mocks, synthetic tensors, or API test doubles; they do not verify actual Qwen inference. In this managed environment, run TestClient tests outside the sandbox because sandboxed local socket-pair creation can block AnyIO's event loop. This restriction does not apply to a normal local terminal.

Run the local-cache-only real-model integration test after downloading model files:

```powershell
$env:DSAA_RUN_REAL_MODEL_TESTS = "1"
python -m unittest llm_engine.test_real_model_integration -v
```

## 10. Measurements and Known Limitations

- Inference result timing includes measured prefill/decode execution; model download/load time is not returned as a metric.
- TTFT is not a dedicated field. The code has prefill latency, but no end-to-end TTFT measurement.
- `tokens_per_second` is generated-token count divided by request inference elapsed time; it is not a benchmark across repeated runs.
- Cache reports estimate cache tensor bytes and optionally storage allocation. They do not measure peak process RAM, allocator-reserved GPU memory, CPU/GPU utilization, or concurrent system-wide footprint.
- `test_real_model_integration.py` tests local weights only and is skipped by default. In the environment used for the latest run, explicitly enabling it produced a setup error because the Qwen files were not cached; the test does not download them. Available RAM was about 1.5 GB, below the roughly 6.2 GB needed just for 1.54B float32 parameters, before runtime overhead. Real Qwen inference and real cache tensors therefore remain unverified here.
- API and scheduler unit tests use mocks. Their passing results establish control-flow behavior, not model quality or real-model performance.
- The scheduler interleaves requests but performs per-request forwards; true tensor batching, DSAA cache eviction, semantic scoring, and full benchmark evaluation are outside this LLM infrastructure module's implemented scope.
