# DSAA LLM Engine (`llm_engine/`)

**Module Owner:** Person 1 — LLM & KV Cache Engineer  
**Project:** DSAA (Dynamic Smart Attention-based Allocation)

The `llm_engine/` module is the foundation of the DSAA system. It provides the core LLM infrastructure: loading pretrained language models, managing hardware device and precision mapping, configuring tokenizers for causal autoregressive generation, and managing the Key-Value (KV) Cache.

---

## 1. Module Structure

```text
llm_engine/
├── __init__.py           # Package exports for clean public API
├── config.py             # ModelConfig, enums, device and precision resolution
├── exceptions.py         # Custom error hierarchy (Dependency, Device, Precision, Download, OOM)
├── model_loader.py       # ModelLoader class and load_model_and_tokenizer helper
├── example_usage.py      # Standalone runnable usage example
├── test_model_loader.py  # Unit tests with isolated mocks (28 passing tests)
└── README.md             # Architecture documentation and usage guide
```

---

## 2. Core Components

### A. Configuration (`config.py`)
Encapsulates all runtime parameters for inference:
* **Primary Candidate:** `Qwen/Qwen2.5-1.5B-Instruct` (default)
* **Alternative Candidates:** `google/gemma-2-2b-it`, `meta-llama/Llama-3.1-8B-Instruct`
* **Device Resolution (`resolve_device`):**
  * `"auto"`: Auto-detects CUDA; falls back to CPU if unavailable.
  * `"cuda"` / `"cuda:<idx>"`: Validates CUDA availability on hardware; raises `DeviceConfigurationError` if missing.
  * `"cpu"`: Forces CPU execution.
* **Precision Resolution (`resolve_precision`):**
  * `"auto"`:
    * CUDA with BF16 instruction support $\rightarrow$ `torch.bfloat16`
    * CUDA without BF16 support $\rightarrow$ `torch.float16`
    * CPU fallback $\rightarrow$ `torch.float32` (ensuring broad kernel support and numerical stability).
  * Explicit modes (`float32`, `float16`, `bfloat16`) are validated against hardware capabilities. Attempting `float16` on CPU or `bfloat16` on unsupported GPUs raises `PrecisionConfigurationError`.
* **Causal Autoregressive Defaults:**
  * `padding_side = "left"`: Mandatory for decoder-only autoregressive generation so that padding does not alter sequence position embeddings.
  * `use_cache = True`: Ensures the transformer produces and maintains the KV Cache during forward passes.

### B. Model Loader (`model_loader.py`)
Responsible for loading, preparing, and managing the model lifecycle:
* **Dependency Verification:** Validates that `torch` and `transformers` are present; raises `DependencyMissingError` with specific installation commands if missing.
* **Model Configuration for Inference:**
  * `model.eval()`: Puts layers (Dropout, BatchNorm) into inference mode.
  * `param.requires_grad = False`: Freezes all parameters to eliminate autograd overhead and save memory.
  * `model.config.use_cache = True`: Explicitly enables KV Cache generation.
* **Tokenizer Setup:**
  * Configures left padding.
  * Ensures `pad_token` is mapped (falls back to `eos_token` if `pad_token` is undefined, preventing padding-token crashes).
* **Resource Caching & Cleanup:**
  * `load()` caches `ModelArtifacts` to prevent duplicate memory allocations when accessed repeatedly.
  * `unload()` explicitly deletes references, triggers garbage collection (`gc.collect()`), and clears CUDA cache (`torch.cuda.empty_cache()`).

### C. Exception Hierarchy (`exceptions.py`)
All module errors inherit from `LLMEngineError`:
* `DependencyMissingError`: Raised when runtime libraries (`torch`, `transformers`) are not installed.
* `DeviceConfigurationError`: Raised for invalid device names or missing CUDA hardware.
* `PrecisionConfigurationError`: Raised for unsupported floating point formats.
* `ModelDownloadError`: Raised on Hugging Face Hub network errors or repository lookup failures.
* `ModelInitializationError`: Raised on Out-of-Memory (OOM) or weight initialization failures.
* `TokenizerInitializationError`: Raised if tokenizer fails to initialize.

---

## 3. Usage Examples

### Standard Usage with `ModelLoader`

```python
from llm_engine import ModelLoader, ModelConfig

# 1. Configure the model
config = ModelConfig(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",        # Automatically selects CUDA if available, CPU otherwise
    precision="auto",     # Selects BF16/FP16 on GPU, FP32 on CPU
    use_cache=True,       # Required for DSAA KV cache management
    padding_side="left",  # Standard for autoregressive decoders
)

# 2. Instantiate loader
loader = ModelLoader(config)

# 3. Load artifacts
try:
    artifacts = loader.load()
    model = artifacts.model
    tokenizer = artifacts.tokenizer
    print(f"Loaded on {artifacts.device} with precision {artifacts.dtype}")
except Exception as e:
    print(f"Failed to load: {e}")

# 4. Access cached model & tokenizer later
model = loader.get_model()
tokenizer = loader.get_tokenizer()

# 5. Clean up memory when finished
loader.unload()
```

### Quick One-Liner

```python
from llm_engine import load_model_and_tokenizer

artifacts = load_model_and_tokenizer(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",
    precision="auto"
)
model = artifacts.model
tokenizer = artifacts.tokenizer
```

---

## 4. Running Tests

Unit tests are located directly in `llm_engine/test_model_loader.py` and run without requiring external weights or active GPUs by utilizing mocks.

To run the full test suite:

```bash
python -m unittest llm_engine.test_model_loader -v
```

### Verified Test Categories (28 Tests):
1. **ModelConfig Validation:** Default parameters, custom valid parameters, rejection of empty model IDs, invalid devices, invalid precisions, and invalid padding sides.
2. **Device Selection:** Auto-detection with CUDA available/unavailable, explicit CPU, explicit CUDA, and explicit CUDA indexing (`cuda:0`).
3. **Precision Selection:** Auto precision on CUDA (BF16 vs FP16), auto on CPU (FP32), explicit FP32, explicit FP16 CPU rejection, explicit BF16 unsupported GPU rejection.
4. **Dependency Handling:** Graceful detection and specific exception raising when `torch` or `transformers` is missing.
5. **Model Loader Lifecycle:** Initial state, eval mode configuration, requires_grad freezing, use_cache setting, left-padding assignment, pad-token fallback, idempotent caching, and clean resource unloading.
6. **Error Handling:** Tokenizer download failures, model weight download failures, and CUDA Out of Memory (OOM) scenarios.

---

## 5. Standalone Example Script

Run the standalone example script inside `llm_engine`:

```bash
python -m llm_engine.example_usage
```

---

## 6. Current Limitations & Next Step

* **Local Environment Dependencies:** In environments where `torch` or `transformers` have not yet been installed, `ModelLoader` safely catches the missing dependency and outputs explicit installation instructions.
* **Next Task (Task 2):** KV Cache inspection utilities (`kv_extractor.py`) to inspect tensor shapes, layers, attention heads, and exact memory consumption of the cached Key/Value states during generation.
