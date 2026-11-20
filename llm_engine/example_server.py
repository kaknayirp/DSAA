"""
Standalone Model Serving Demonstration and Server Runner for DSAA.

Module Owner: Person 1 — LLM & KV Cache Engineer.

Demonstrates:
1. Creating the FastAPI application from the existing InferenceEngine and RequestScheduler.
2. Interacting with endpoints (/health, /models, /generate blocking and streaming, /requests/{id}).
3. Provides command-line entry point to run uvicorn server.
"""

import sys
import json
import logging
from starlette.testclient import TestClient

from llm_engine.config import ModelConfig
from llm_engine.server_config import ServerConfig
from llm_engine.scheduler_config import SchedulerConfig
from llm_engine.scheduler import RequestScheduler
from llm_engine.inference_engine import InferenceEngine
from llm_engine.model_loader import ModelArtifacts
from llm_engine.test_inference_engine import StatefulCausalLM, MockTokenizer
from llm_engine.api_app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_demo_app():
    """Build FastAPI serving application with simulated Qwen2.5 model artifacts."""
    model = StatefulCausalLM(vocab_size=1000, num_layers=4, num_kv_heads=2, head_dim=128)
    tokenizer = MockTokenizer(eos_token_id=999)
    artifacts = ModelArtifacts(
        model=model,
        tokenizer=tokenizer,
        config=ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct"),
        device="cpu",
        dtype="float16",
    )
    engine = InferenceEngine(artifacts=artifacts)
    scheduler = RequestScheduler(engine, SchedulerConfig(max_batch_size=4))
    server_cfg = ServerConfig(host="0.0.0.0", port=8000)
    return create_app(scheduler=scheduler, server_config=server_cfg, start_background_worker=True)


def run_api_demonstration():
    print("=" * 80)
    print("  DSAA MODEL SERVING API LAYER DEMONSTRATION")
    print("=" * 80)

    app = build_demo_app()
    client = TestClient(app)

    # 1. Health Check
    print("\n[1] Calling GET /health ...")
    health_resp = client.get("/health")
    print(f"Status Code: {health_resp.status_code}")
    print(json.dumps(health_resp.json(), indent=2))

    # 2. Model Information
    print("\n[2] Calling GET /models ...")
    models_resp = client.get("/models")
    print(f"Status Code: {models_resp.status_code}")
    print(json.dumps(models_resp.json(), indent=2))

    # 3. Text Generation (Blocking JSON)
    prompt = "Explain how DSAA dynamically manages the KV Cache."
    print(f"\n[3] Calling POST /generate (Blocking JSON) for: '{prompt}' ...")
    payload = {
        "prompt": prompt,
        "max_new_tokens": 6,
        "temperature": 0.0,
        "do_sample": False,
        "stream": False,
    }
    gen_resp = client.post("/generate", json=payload)
    print(f"Status Code: {gen_resp.status_code}")
    gen_data = gen_resp.json()
    print(json.dumps(gen_data, indent=2))
    req_id = gen_data["request_id"]

    # 4. Request Status Query
    print(f"\n[4] Calling GET /requests/{req_id} ...")
    status_resp = client.get(f"/requests/{req_id}")
    print(f"Status Code: {status_resp.status_code}")
    print(json.dumps(status_resp.json(), indent=2))

    # 5. Streaming Text Generation (Server-Sent Events)
    print("\n[5] Calling POST /generate (Streaming SSE) ...")
    stream_payload = {
        "prompt": "Streaming generation test prompt.",
        "max_new_tokens": 4,
        "stream": True,
    }
    stream_resp = client.post("/generate", json=stream_payload)
    print(f"Status Code: {stream_resp.status_code}")
    print("Streamed SSE Events:")
    for line in stream_resp.iter_lines():
        if line.strip():
            print(f"  {line}")

    print("\n" + "=" * 80)
    print("  EXAMPLE CURL INVOCATIONS:")
    print("=" * 80)
    print("""
# 1. Health Check:
curl -X GET http://localhost:8000/health

# 2. List Models:
curl -X GET http://localhost:8000/models

# 3. Blocking Text Generation:
curl -X POST http://localhost:8000/generate \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "What is the KV cache?", "max_new_tokens": 64, "temperature": 0.7}'

# 4. Streaming Text Generation (SSE):
curl -X POST http://localhost:8000/generate \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "Explain continuous batching.", "max_new_tokens": 64, "stream": true}'

# 5. Check Request Status:
curl -X GET http://localhost:8000/requests/<request_id>

# 6. Cancel Request:
curl -X POST http://localhost:8000/requests/<request_id>/cancel
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        import uvicorn
        app = build_demo_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_api_demonstration()
