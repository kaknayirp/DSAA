# DSAA LLM Engine (`llm_engine/`)

**Module Owner:** Person 1 — LLM & KV Cache Engineer  
**Project:** DSAA (Dynamic Smart Attention-based Allocation)

The `llm_engine/` module is the foundation of the DSAA system. It provides the core LLM infrastructure: loading pretrained language models, managing hardware device and precision mapping, extracting and inspecting the Key-Value (KV) Cache, and executing autoregressive inference with incremental KV Cache updates.

---

## 1. Module Structure

```text
llm_engine/
├── __init__.py                # Package exports for clean public API
├── config.py                  # ModelConfig, enums, device and precision resolution
├── generation_config.py       # GenerationConfig, sampling and stopping parameters
├── exceptions.py              # Custom error hierarchy (Dependency, Device, Precision, OOM, Inference, Cache)
├── model_loader.py            # ModelLoader class and load_model_and_tokenizer helper
├── kv_extractor.py            # KVCacheInspector, TensorStats, CacheReport, memory calculations
├── inference_engine.py        # InferenceEngine, InferenceRequest, GenerationState, InferenceResult
├── example_usage.py           # Standalone runnable model loader usage example
├── example_kv_inspection.py   # Standalone runnable KV Cache inspection & memory simulation
├── example_inference.py       # Standalone runnable autoregressive generation & cache reuse demo
├── test_model_loader.py       # Unit tests for model loader (28 passing tests)
├── test_kv_extractor.py       # Unit tests for KV cache inspection & memory calculations (14 passing tests)
├── test_inference_engine.py   # Unit tests for inference engine & KV cache integration (18 passing tests)
└── README.md                  # Complete technical documentation and usage guide
```

---

## 2. Architecture & Pipeline Overview

```
                                  [ User Input / Prompt ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │       Tokenizer (Left-Pad)     │
                             └───────────────┬───────────────┘
                                             │
                                     [ Input Token IDs ]
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │  InferenceEngine   │
                                  └──────────┬─────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ 1. Prefill Stage ]                        [ 2. Decode Stage ]
      Processes prompt (N tokens) at once        Processes 1 token per iteration
        Generates past_key_values (N)             Feeds token + past_key_values
      Returns first generated token (t_0)          Incremental cache update (N+1, N+2...)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    KVCacheInspector (Prompt 2) │
                             │  - Non-destructive inspection │
                             │  - Memory breakdown (B, KB, MB)│
                             │  - CacheReport generation     │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                                    [ InferenceResult ]
                              - generated_token_ids
                              - decoded_text
                              - latency & throughput
                              - final_cache_report
```

---

## 3. Core Components

### A. Configuration (`config.py` & `generation_config.py`)
* **`ModelConfig`:** Hardware resolution (`auto`, `cuda`, `cpu`), precision modes (`float32`, `float16`, `bfloat16`), left-padding, and model candidate identifiers (`Qwen/Qwen2.5-1.5B-Instruct` default).
* **`GenerationConfig`:** Decoding settings:
  * `max_new_tokens`: token budget.
  * `temperature`, `top_k`, `top_p`: sampling distribution controls.
  * `do_sample`: toggle between greedy (`argmax`) and stochastic sampling.
  * `repetition_penalty`: discounts already-generated tokens.
  * `stop_token_ids` & `eos_token_id`: early exit triggers.
  * `max_cache_capacity`: enforces hardware memory bounds on the KV Cache.

### B. Model Loader (`model_loader.py`)
* Pretrained causal language model initialization.
* Parameter freezing (`requires_grad = False`) and inference mode (`model.eval()`).
* Explicit cache enabling (`model.config.use_cache = True`).
* Idempotent caching and memory cleanup (`unload()`).

### C. KV Cache Extraction & Inspection (`kv_extractor.py`)
* Normalizes `DynamicCache` and tuple-based caches into uniform layer tensors.
* Inspects Key and Value shapes: `[batch_size, num_kv_heads, seq_len, head_dim]`.
* Detects Grouped Query Attention (GQA) (e.g. Qwen2.5 uses 2 KV heads vs 12 Q heads).
* Calculates exact logical memory and tracks physical buffer pointers to avoid double-counting.

### D. Inference Engine (`inference_engine.py`)
* **Prefill Stage:**
  1. Validates prompt token sequence.
  2. Executes single model forward pass for the complete prompt.
  3. Initializes `past_key_values` in an isolated `GenerationState`.
  4. Selects first token from prompt's final logits.
* **Decode Stage:**
  1. Supplies only the single latest token `[1, 1]` to `model()`.
  2. Supplies existing `past_key_values` — **strictly avoiding recomputation of prompt tokens**.
  3. Model returns new logits and appends new Key/Value representations to the cache.
  4. Checks stopping conditions (EOS, stop token, max tokens, capacity).
* **Request & Cache Lifecycle:**
  * Every request receives an independent `GenerationState`.
  * Guarantees zero cross-request contamination.
  * Safe cleanup: `state.cleanup()` drops tensor references upon generation exit.
* **Prompt 2 Integration:**
  * Embeds a full `CacheReport` directly in `InferenceResult.final_cache_report`.

---

## 4. Usage Examples

### Full Autoregressive Generation with KV Cache

```python
from llm_engine import ModelLoader, ModelConfig, InferenceEngine, GenerationConfig

# 1. Load Model and Initialize Engine
config = ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="auto")
loader = ModelLoader(config)
engine = InferenceEngine(model_loader=loader)

# 2. Configure Generation
gen_config = GenerationConfig(
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 3. Generate Tokens
result = engine.generate(
    prompt="Explain the difference between prefill and decode stages in LLMs.",
    generation_config=gen_config,
    inspect_cache=True,
)

# 4. View Results and KV Cache Statistics
print(result.summary())
print(f"Generated text: {result.decoded_text}")
print(f"Final KV Cache Memory: {result.final_cache_report.total_mb:.2f} MB")
```

---

## 5. Running the Test Suite

All unit and integration tests are strictly self-contained within `llm_engine/`:

```bash
# Run all 60 tests across the entire module
python -m unittest discover -s llm_engine -p "test_*.py" -v
```

### Individual Test Suites:
1. **Model Loader Tests (28 tests):** `python -m unittest llm_engine.test_model_loader -v`
2. **KV Cache Extractor Tests (14 tests):** `python -m unittest llm_engine.test_kv_extractor -v`
3. **Inference Engine Tests (18 tests):** `python -m unittest llm_engine.test_inference_engine -v`

Total: **60 tests, 0 failures, 0 errors**.

---

## 6. Standalone Demonstration Scripts

```bash
# Model Loader demo
python -m llm_engine.example_usage

# KV Cache inspection & Qwen2.5 memory scaling simulation
python -m llm_engine.example_kv_inspection

# Complete autoregressive inference engine demo
python -m llm_engine.example_inference
```
