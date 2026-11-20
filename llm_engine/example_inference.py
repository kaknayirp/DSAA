"""
Standalone demonstration for the DSAA LLM Inference Engine with KV Cache Integration.

Demonstrates:
1. Initializing GenerationConfig (max_new_tokens, temperature, top_k, top_p, stop tokens).
2. Prompt prefill execution (processes prompt sequence once and creates KV Cache).
3. Incremental autoregressive decode steps (processes 1 token at a time, reusing the KV Cache).
4. Extracting and printing the final KV Cache memory and layer report (Prompt 2 integration).
5. Attempting live inference if dependencies and weights are present.
"""

import sys
import logging
from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.inference_engine import InferenceEngine, InferenceRequest
from llm_engine.model_loader import ModelArtifacts, ModelLoader
from llm_engine.test_inference_engine import StatefulCausalLM, MockTokenizer
from llm_engine.exceptions import DependencyMissingError, ModelDownloadError, LLMEngineError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_simulated_inference_demo():
    print("=" * 80)
    print("  DSAA INFERENCE ENGINE: AUTOREGRESSIVE GENERATION & KV CACHE REUSE DEMO")
    print("=" * 80)

    # 1. Setup simulated model matching Qwen2.5-1.5B specs
    model = StatefulCausalLM(vocab_size=1000, num_layers=28, num_kv_heads=2, head_dim=128)
    tokenizer = MockTokenizer(eos_token_id=999)
    artifacts = ModelArtifacts(
        model=model,
        tokenizer=tokenizer,
        config=ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct"),
        device="cpu",
        dtype="float16",
    )
    engine = InferenceEngine(artifacts=artifacts)

    # 2. Configure generation parameters
    gen_config = GenerationConfig(
        max_new_tokens=8,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        do_sample=False,  # Greedy for deterministic demo
        eos_token_id=999,
        return_cache_report=True,
    )
    prompt = "Explain how Dynamic Smart Attention Allocation saves memory during inference."
    print(f"\n[1] Generation Request:")
    print(f"    - Prompt:           '{prompt}'")
    print(f"    - Max New Tokens:   {gen_config.max_new_tokens}")
    print(f"    - Sampling Mode:    {'Sampling' if gen_config.do_sample else 'Greedy'}")

    # 3. Execute generation
    print("\n[2] Executing Autoregressive Generation...")
    result = engine.generate(prompt, generation_config=gen_config, inspect_cache=True)

    print("\n[3] Inference Result Summary:")
    print(result.summary())

    print("[4] Verification of KV Cache Recomputation Invariant:")
    print(f"    - Total forward passes executed: {len(model.forward_call_inputs)}")
    print(f"    - Call 1 (Prefill): {model.forward_call_inputs[0]} tokens (Prompt sequence)")
    for step_idx, tokens_passed in enumerate(model.forward_call_inputs[1:], start=1):
        print(f"    - Call {step_idx + 1} (Decode Step {step_idx}): {tokens_passed} token passed (Previous tokens retrieved from KV Cache)")

    # 5. Show Prompt 2 KV Cache report
    if result.final_cache_report:
        print("\n[5] Embedded Prompt 2 KV Cache Inspection Report:")
        print(result.final_cache_report.summary_table())


def run_live_inference_demo():
    print("\n" + "=" * 80)
    print("  DSAA LIVE INFERENCE DEMO (Active model & hardware check)")
    print("=" * 80)

    try:
        config = ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="auto")
        loader = ModelLoader(config)
        engine = InferenceEngine(model_loader=loader)

        print("Attempting live generation with Qwen2.5-1.5B-Instruct...")
        result = engine.generate("What is the KV Cache in transformer models?", GenerationConfig(max_new_tokens=16))
        print(result.summary())
    except DependencyMissingError as exc:
        print(f"[ENVIRONMENT NOTICE] Cannot run live model inference: {exc}")
    except ModelDownloadError as exc:
        print(f"[NETWORK NOTICE] Cannot download live model weights: {exc}")
    except LLMEngineError as exc:
        print(f"[LLM ENGINE NOTICE] {exc}")


if __name__ == "__main__":
    run_simulated_inference_demo()
    run_live_inference_demo()
