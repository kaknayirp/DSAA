# DSAA Studio API

Start FastAPI from the repository root. This mode exposes real host RAM telemetry and the labeled cache sample without loading a model.

```powershell
python -m pip install "uvicorn[standard]" fastapi psutil
python -m uvicorn backend.main:app
```

## Live Original inference

Install the project dependencies and point the backend at a local Hugging Face Qwen2 or Qwen2.5 instruct checkpoint that uses full attention, such as Qwen2.5-1.5B-Instruct. The adapter loads from local files and uses greedy generation. It limits prompt plus output to 8,192 tokens or the model limit, whichever is lower.

```powershell
python -m pip install -r requirements.txt
$env:DSAA_MODEL = 'C:/path/to/Qwen2.5-1.5B-Instruct'
$env:DSAA_DEVICE = 'cpu' # Set to cuda:0 for a CUDA-enabled PyTorch installation.
python -m uvicorn backend.main:app
```

The adapter measures final KV tensor bytes, sampled peak process RSS during generation, time to the first generated token, and generated tokens per second over the full generation interval. The final cache contains positions processed during generation. The final sampled output token may not have a KV entry yet. Host memory pressure is the fraction of system RAM in use. On a configured CUDA device, system telemetry also reports memory allocated to PyTorch tensors in the API process.

## Team integration

The default server calls [configure_project](integration.py) on startup. Fill in its marked registration points as the team modules land.

| Module | Connect in `configure_project` |
| --- | --- |
| `llm_engine/` | Load the shared model, tokenizer, and KV cache manager. Set the model and device metadata and register the Original runner. |
| `dsaa_algorithm/` and `llm_engine/` | Connect DSAA scoring to KV cache eviction and register the DSAA runner. |
| H2O and SnapKV implementations | Register each baseline once it can run against the same model. |
| `evaluation/` | Assign the team's recall, scoring, and aggregation adapter to `studio.evaluator`. |

Each strategy callable receives an `InferenceRequest` and a snapshot of `RunSettings`, and returns an `InferenceResult` from `backend.studio_schemas`. Use matching model, tokenization, and decoding settings for all strategies. Populate token IDs, positions, attention, semantic and recency contributions, final scores, decisions and reasons where the engine exposes them. The DSAA adapter should use memory pressure to adjust eviction aggressiveness independently of the three scoring weights.

For a custom app or a test, you can also supply adapters directly through `create_app(runners=..., evaluator=..., model=..., device=...)`.

The server serializes model work and queues benchmarks on a separate worker. Use a single API worker. Config, recent runs, and experiment records live in memory and reset on restart. Download reports to keep the results.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Registered strategies and runtime configuration |
| `GET /api/system/metrics`, `WS /ws/metrics` | Host RAM, process RAM, and compute status |
| `GET /api/demo` | Labeled sample for the explorer and comparison view |
| `POST /api/inference`, `POST /api/compare` | Measured single-strategy and matched Original/DSAA runs |
| `GET /api/cache/{run_id}`, `GET /api/cache/{run_id}/tokens` | Run metadata and per-token telemetry |
| `POST /api/benchmarks`, `GET /api/benchmarks` | Queue and list recall experiments |
| `GET /api/benchmarks/{id}`, `GET /api/benchmarks/{id}/report` | Progress and downloadable completed results |
| `GET /api/config`, `PUT /api/config` | Read or save settings for future runs |

The OpenAPI UI is at `http://127.0.0.1:8000/docs`. To enable Benchmark Lab, pass the evaluation team's adapter as `create_app(evaluator=...)`. It implements the `BenchmarkEvaluator` protocol in `backend.studio` with a recall case generator, an exact-match scoring method, and a results summary. Reports include the inputs and settings, per-case responses, and the evaluator's aggregates. Retention ratio compares final cached positions to the matching Original run, when present. Different generated output lengths can affect that ratio.
