"""
Usage demonstration for KV Cache extraction and inspection utilities.

Demonstrates:
1. Extracting and analyzing KV Cache structures (DynamicCache and tuples).
2. Simulating the exact Qwen2.5-1.5B-Instruct architecture (28 layers, GQA with 2 KV heads).
3. Memory scaling analysis across sequence lengths (512, 2048, and 32768 tokens).
4. Generating human-readable summary reports and JSON-compatible statistics.
5. Attempting live model inference inspection when weights/dependencies are present.
"""

import sys
import logging
from llm_engine.kv_extractor import (
    KVCacheInspector,
    extract_kv_cache,
    run_inference_and_inspect_cache,
)
from llm_engine.test_kv_extractor import SyntheticTensor
from llm_engine.model_loader import ModelLoader
from llm_engine.config import ModelConfig
from llm_engine.exceptions import DependencyMissingError, ModelDownloadError, LLMEngineError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_qwen_simulation():
    print("=" * 80)
    print("  DSAA KV CACHE INSPECTION: Qwen2.5-1.5B-Instruct SIMULATION")
    print("=" * 80)

    # Architectural specs for Qwen/Qwen2.5-1.5B-Instruct:
    num_layers = 28
    batch_size = 1
    num_kv_heads = 2      # Grouped Query Attention (12 Q heads, 2 KV heads)
    head_dim = 128
    dtype_str = "torch.bfloat16"  # 2 bytes per element

    test_lengths = [128, 512, 2048, 8192, 32768]
    print(f"\nModel Configuration:")
    print(f"  - Layers:        {num_layers}")
    print(f"  - Attention:     Grouped Query Attention (GQA)")
    print(f"  - KV Heads:      {num_kv_heads} (vs 12 Query heads)")
    print(f"  - Head Dim:      {head_dim}")
    print(f"  - Precision:     {dtype_str} (2 bytes/elem)")
    print(f"  - Tensor Layout: [batch_size, num_kv_heads, seq_len, head_dim]")

    print("\nMemory Footprint Scaling Across Context Lengths:")
    print("-" * 80)
    print(f"{'Context (Tokens)':<18} | {'Tensor Shape':<24} | {'Total Memory (MB)':<18} | {'Per Token (KB)':<14}")
    print("-" * 80)

    for seq_len in test_lengths:
        shape = (batch_size, num_kv_heads, seq_len, head_dim)
        cache = tuple(
            (
                SyntheticTensor(shape, dtype_str),
                SyntheticTensor(shape, dtype_str),
            )
            for _ in range(num_layers)
        )
        report = extract_kv_cache(cache)
        per_token_kb = (report.total_bytes / seq_len) / 1024.0
        print(f"{seq_len:<18} | {str(shape):<24} | {report.total_mb:<18.4f} | {per_token_kb:<14.2f}")

    print("-" * 80)

    # Detailed report for context = 512
    shape_512 = (batch_size, num_kv_heads, 512, head_dim)
    cache_512 = tuple(
        (SyntheticTensor(shape_512, dtype_str), SyntheticTensor(shape_512, dtype_str))
        for _ in range(num_layers)
    )
    report_512 = extract_kv_cache(cache_512)
    print("\nDetailed Report Sample (Context = 512 tokens):")
    print(report_512.summary_table())


def run_live_inference_check():
    print("\n" + "=" * 80)
    print("  DSAA LIVE INFERENCE CACHE INSPECTION (If dependencies & weights available)")
    print("=" * 80)

    config = ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="auto", precision="auto")
    loader = ModelLoader(config)

    try:
        print("Attempting to load model via ModelLoader...")
        artifacts = loader.load()
        print("Model loaded successfully. Running single forward pass with use_cache=True...")
        outputs, report = run_inference_and_inspect_cache(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            prompt="Explain the difference between Key-Value cache and standard activations in LLMs."
        )
        print("\nLive Inference KV Cache Report:")
        print(report.summary_table())
        loader.unload()
    except DependencyMissingError as exc:
        print(f"[ENVIRONMENT NOTICE] Cannot run live model inference: {exc}")
    except ModelDownloadError as exc:
        print(f"[NETWORK NOTICE] Cannot download live model weights: {exc}")
    except LLMEngineError as exc:
        print(f"[LLM ENGINE NOTICE] {exc}")


if __name__ == "__main__":
    run_qwen_simulation()
    run_live_inference_check()
