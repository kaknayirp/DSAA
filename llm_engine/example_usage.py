"""
Example usage script for DSAA LLM Engine Model Loader.

Demonstrates:
1. Configuring the model loader for Qwen2.5-1.5B-Instruct.
2. Device and precision resolution.
3. Loading the model and tokenizer.
4. Handling missing dependencies or network limitations cleanly.
5. Resource cleanup (unloading).
"""

import sys
import logging
from llm_engine.config import ModelConfig, DEFAULT_MODEL_ID, resolve_device, resolve_precision
from llm_engine.model_loader import ModelLoader
from llm_engine.exceptions import (
    DependencyMissingError,
    ModelDownloadError,
    ModelInitializationError,
    LLMEngineError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_engine.example")


def run_example():
    print("=" * 70)
    print("DSAA LLM Engine: Model Loader Demonstration")
    print(f"Primary Target Model: {DEFAULT_MODEL_ID}")
    print("=" * 70)

    # 1. Initialize configuration
    config = ModelConfig(
        model_id=DEFAULT_MODEL_ID,
        device="auto",        # Automatically selects CUDA if available, CPU otherwise
        precision="auto",     # Selects BF16/FP16 on GPU, FP32 on CPU
        use_cache=True,       # KV Cache must be enabled for DSAA
        padding_side="left",  # Decoder-only autoregressive generation standard
    )
    print(f"\n[1] Initialized Configuration:")
    print(f"    - Model ID:     {config.model_id}")
    print(f"    - Device Mode:  {config.device}")
    print(f"    - Precision:    {config.precision}")
    print(f"    - Use KV Cache: {config.use_cache}")
    print(f"    - Padding Side: {config.padding_side}")

    # 2. Inspect resolved device and precision
    resolved_dev = resolve_device(config.device)
    resolved_prec = resolve_precision(config.precision, resolved_dev)
    print(f"\n[2] Hardware Resolution:")
    print(f"    - Resolved Device:    {resolved_dev}")
    print(f"    - Resolved Precision: {resolved_prec}")

    # 3. Attempt model loading
    print(f"\n[3] Attempting Model Loading...")
    loader = ModelLoader(config)

    try:
        artifacts = loader.load()
        print("\n--> SUCCESS: Model and Tokenizer loaded successfully!")
        print(f"    Model Type: {type(artifacts.model).__name__}")
        print(f"    Tokenizer Type: {type(artifacts.tokenizer).__name__}")
        print(f"    Evaluation Mode: {not artifacts.model.training}")

        # Example prompt formatting
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain KV Cache in LLMs in one sentence."}
        ]
        if hasattr(artifacts.tokenizer, "apply_chat_template"):
            formatted_prompt = artifacts.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"\n[4] Formatted Chat Prompt Sample:\n{formatted_prompt}")

        # 5. Clean up resources
        print(f"\n[5] Releasing Model Resources...")
        loader.unload()
        print(f"    Is Loaded: {loader.is_loaded}")

    except DependencyMissingError as exc:
        print("\n--> [ENVIRONMENT NOTICE] Missing Dependency:")
        print(f"    {str(exc)}")
        print("\n    Note: This is expected if 'torch' or 'transformers' has not yet")
        print("    been installed in the current environment.")
    except ModelDownloadError as exc:
        print("\n--> [NETWORK / HUB NOTICE] Download Failed:")
        print(f"    {str(exc)}")
    except ModelInitializationError as exc:
        print("\n--> [HARDWARE NOTICE] Initialization / Memory Limit Reached:")
        print(f"    {str(exc)}")
    except LLMEngineError as exc:
        print(f"\n--> [ERROR] LLM Engine Failure: {str(exc)}")

    print("\n" + "=" * 70)
    print("Demonstration Complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
