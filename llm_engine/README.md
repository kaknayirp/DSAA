# DSAA LLM Engine (`llm_engine/`)

**Module Owner:** Person 1 — LLM & KV Cache Engineer  
**Project:** DSAA (Dynamic Smart Attention-based Allocation)

The `llm_engine/` module is the foundation of the DSAA system. It provides the core LLM infrastructure: loading pretrained language models, managing hardware device and precision mapping, configuring tokenizers for causal autoregressive generation, extracting and inspecting the Key-Value (KV) Cache, and managing cache state.

---

## 1. Module Structure

```text
llm_engine/
├── __init__.py                # Package exports for clean public API
├── config.py                  # ModelConfig, enums, device and precision resolution
├── exceptions.py              # Custom error hierarchy (Dependency, Device, Precision, Download, OOM)
├── model_loader.py            # ModelLoader class and load_model_and_tokenizer helper
├── kv_extractor.py            # KVCacheInspector, TensorStats, CacheReport, memory calculations
├── example_usage.py           # Standalone runnable model loader usage example
├── example_kv_inspection.py   # Standalone runnable KV Cache inspection & memory simulation
├── test_model_loader.py       # Unit tests for model loader (28 passing tests)
├── test_kv_extractor.py       # Unit tests for KV cache inspection & memory calculations (14 passing tests)
└── README.md                  # Architecture documentation and usage guide
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

### C. KV Cache Extraction & Inspection (`kv_extractor.py`)
Provides non-destructive extraction, inspection, and memory profiling of the model's Key-Value Cache:
* **Supported Cache Representations:**
  * **Hugging Face `DynamicCache`** (`transformers.cache_utils.DynamicCache`): standard across modern Transformers ($\ge 4.36$).
  * **Legacy Tuple of Tuples:** `((key_0, value_0), (key_1, value_1), ..., (key_{L-1}, value_{L-1}))`.
  * **List of Tuples / Custom Sequence Containers.**
* **Tensor Layout Inspection:**
  * Identifies standard 4D layout: `[batch_size, num_kv_heads, seq_len, head_dim]`.
  * Automatically detects **Grouped Query Attention (GQA)** when `num_kv_heads < 8`.
* **Memory Calculation Methodology:**
  * **Logical Tensor Memory:** Calculated as $\text{numel} \times \text{element\_size\_bytes}$.
  * **Physical Storage Tracking:** Queries underlying storage pointer (`untyped_storage().data_ptr()`) to avoid double-counting shared or preallocated tensor memory buffers.
  * Reports memory precisely in bytes, kilobytes (KB), and megabytes (MB) at tensor, layer, and global cache levels.
* **Structured Output (`CacheReport`):**
  * Provides serialized dictionary (`to_dict()`) for integration with Person 2 (algorithm) and Person 3 (evaluation/dashboard).
  * Provides formatted ASCII summary table (`summary_table()`).

---

## 3. Qwen2.5-1.5B-Instruct Cache Architecture & Dimensions

For our primary candidate model, `Qwen/Qwen2.5-1.5B-Instruct`, the KV Cache exhibits the following observed specifications:

* **Transformer Layers ($L$):** 28
* **Attention Heads:** 12 query heads, **2 key-value heads** (Grouped Query Attention - GQA)
* **Head Dimension ($D$):** 128 ($1536 / 12$)
* **Default Precision:** `bfloat16` / `float16` (2 bytes per element)
* **Key/Value Tensor Shape per Layer:** `(batch_size, 2, seq_len, 128)`
* **Memory Scaling per Token:**
  $$\text{Elements per Token per Layer} = 2 \text{ (K + V)} \times 1 \text{ (batch)} \times 2 \text{ (KV heads)} \times 128 \text{ (head dim)} = 512 \text{ elements}$$
  $$\text{Memory per Token per Layer} = 512 \times 2 \text{ bytes} = 1,024 \text{ bytes} = 1.00 \text{ KB}$$
  $$\text{Total Memory per Token across 28 Layers} = 28 \times 1,024 \text{ bytes} = 28,672 \text{ bytes} = \mathbf{28.00 \text{ KB/token}}$$

### Context Length Footprint Table
| Sequence Length | Key/Value Shape | Total KV Cache Memory | Memory per Token |
|:---|:---|:---|:---|
| 128 tokens | `(1, 2, 128, 128)` | **3.50 MB** | 28.00 KB |
| 512 tokens | `(1, 2, 512, 128)` | **14.00 MB** | 28.00 KB |
| 2,048 tokens | `(1, 2, 2048, 128)` | **56.00 MB** | 28.00 KB |
| 8,192 tokens | `(1, 2, 8192, 128)` | **224.00 MB** | 28.00 KB |
| 32,768 tokens | `(1, 2, 32768, 128)` | **896.00 MB** | 28.00 KB |

---

## 4. Usage Examples

### Inspecting KV Cache from Model Inference

```python
from llm_engine import ModelLoader, ModelConfig, run_inference_and_inspect_cache

# 1. Load model and tokenizer
config = ModelConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="auto")
loader = ModelLoader(config)
artifacts = loader.load()

# 2. Run inference and extract cache report
outputs, report = run_inference_and_inspect_cache(
    model=artifacts.model,
    tokenizer=artifacts.tokenizer,
    prompt="Explain KV Cache in Large Language Models."
)

# 3. Print human-readable summary
print(report.summary_table())

# 4. Programmatic access for algorithm/eval modules
print(f"Total KV Memory: {report.total_mb:.2f} MB")
print(f"Layer 0 Key shape: {report.layers[0].key.shape}")
print(f"Layer 0 Total bytes: {report.layers[0].total_bytes} bytes")
```

### Inspecting an Existing `past_key_values` Object

```python
from llm_engine import extract_kv_cache

# past_key_values can be DynamicCache or tuple of tuples
report = extract_kv_cache(past_key_values)
print(f"Detected cache type: {report.cache_type}")
print(f"Total KV cache size: {report.total_kb:.2f} KB")
```

---

## 5. Running Tests

Unit tests are strictly self-contained within `llm_engine/` and run without external downloads or active GPUs.

Run all tests:
```bash
python -m unittest discover -s llm_engine -p "test_*.py" -v
```

Or run individual suites:
```bash
# Model Loader tests (28 tests)
python -m unittest llm_engine.test_model_loader -v

# KV Cache Extractor tests (14 tests)
python -m unittest llm_engine.test_kv_extractor -v
```

Total: **42 tests, 0 failures, 0 errors**.

---

## 6. Standalone Demonstration Scripts

```bash
# Model loader demonstration
python -m llm_engine.example_usage

# KV Cache inspection & memory simulation
python -m llm_engine.example_kv_inspection
```
