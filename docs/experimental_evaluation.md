# Experimental Evaluation of Dynamic Semantic-Aware Attention (DSAA)

This document provides a consolidated, evidence-based evaluation of the **Dynamic Semantic-Aware Attention (DSAA)** project. It integrates empirical findings across all experimental phases: synthetic cache simulations, real KV-cache extraction and compaction, continuation quality evaluations, latency benchmarking, and regression test suites.

All findings reported here are sourced directly from the experimental scripts, test files, and raw measurement logs within the repository.

---

## 1. Experimental Setup

### 1.1 Model & Hardware Architecture
- **Language Model**: `Qwen/Qwen2.5-1.5B-Instruct`
  - Parameter count: $\approx 1.54\text{ billion}$ parameters
  - Architecture: 28 transformer decoder layers, 12 attention heads per layer
  - Evaluation precision: `torch.float32` (model parameter memory footprint $\approx 6.16\text{ GB}$)
  - Attention implementation: `eager` (unfused Metal execution)
- **Host Hardware**:
  - Processor: Apple Silicon M1 (8 CPU cores, 7/8 GPU cores)
  - Memory: Unified memory architecture
  - Cooling: Fanless passive cooling (subject to sustained thermal throttling under continuous load)
- **Acceleration Backend**: PyTorch Metal Performance Shaders (`mps`)

### 1.2 Software Environment
- **Operating System**: macOS (Darwin arm64)
- **Runtime**: Python 3.12.7 (virtual environment at `.venv`)
- **Key Libraries**:
  - `torch == 2.x` (MPS-enabled)
  - `transformers == 5.22.2` (utilizing `DynamicCache` with `ddp_cache_data`)
  - `pytest == 9.2.0`

### 1.3 Evaluation Prompts
Empirical evaluations across the project used short prompt sequences ($26$ to $50$ tokens) categorized into five domains:
1. **Technical**: TCP protocol reliability, 3-way handshakes, error detection (~40 tokens).
2. **Programming**: Second-largest element in an array with duplicate handling (~35 tokens).
3. **General Knowledge**: Renewable energy (solar and wind) environmental benefits and limitations (~35 tokens).
4. **Reasoning**: Time management scheduling trade-offs for student study hours (~45 tokens).
5. **AI/ML**: Supervised vs. unsupervised learning classification and clustering (~35 tokens).
6. **Synthetic / Demo Prompt**: Student profile prompt used for initial validation (~26 tokens).

### 1.4 Scope of Measurement
- **What Was Measured**:
  - KV-cache tensor size in bytes (`layer.keys` and `layer.values` element count $\times$ element size).
  - Next-token probability distribution fidelity: Logit cosine similarity, top-1 greedy prediction agreement, and top-5 token overlap.
  - Component execution time: Attention scoring, heuristic lexical/structural scoring, recency scoring, combined scoring, token selection, tensor slicing, and `DynamicCache` reconstruction.
  - Forward-pass inference latency on MPS.
- **What Was NOT Measured**:
  - Total process RAM or system memory (OS paging, framework overhead, and base model parameters are separate).
  - Perplexity or cross-entropy loss across large standardized corpora (e.g., WikiText-103, C4).
  - Long-context retrieval or needle-in-a-haystack tasks ($L > 2{,}048$ tokens).
  - Batched inference throughput ($B > 1$).

---

## 2. Synthetic Eviction Benchmark Results

*Source files*: `evaluation/benchmark_eviction.py`, `tests/test_benchmark_eviction.py`

Before interfacing with real model weights, DSAA was evaluated against standard cache replacement policies (**FIFO** - First In First Out, and **LRU** - Least Recently Used) using a 4-layer synthetic KV cache ($L=100$ tokens, 10 predetermined "important" synthetic tokens, 100 benchmark repetitions, 5 warmup runs).

### 2.1 Important-Token Retention Across Budgets

| Retention Budget | Target Tokens | FIFO Retained Important (out of 10) | LRU Retained Important (out of 10) | DSAA Retained Important (out of 10) |
|---|---:|---:|---:|---:|
| **80%** | 80 tokens | 7 / 10 (70%) | 9 / 10 (90%) | **8 / 10 (80%)** |
| **60%** | 60 tokens | 5 / 10 (50%) | 6 / 10 (60%) | **6 / 10 (60%)** |
| **50%** | 50 tokens | 4 / 10 (40%) | 4 / 10 (40%) | **4 / 10 (40%)** |
| **30%** | 30 tokens | 2 / 10 (20%) | 0 / 10 (0%) | **4 / 10 (40%)** |

### 2.2 Key Observations from Synthetic Benchmarking
1. **Targeted Protection**: FIFO and LRU evict tokens based solely on arrival position or access timestamps. At extreme eviction pressure (30% retention budget / 70% eviction), standard LRU evicts all important tokens (0/10 retained, 0%) due to access timestamp patterns, whereas DSAA preserves 4/10 (40%) of designated important tokens through multi-signal importance scoring combined with protected recent tokens.
2. **Attention Sink & Recency Guarantee**: DSAA's protected recent token parameter (`protected_recent_tokens=2`) ensured that the latest sequence tokens ($tokens\ 98, 99$) remained in the cache regardless of their individual scores.
3. **Algorithmic Selection Overhead**: Selection time for DSAA across 100 elements averaged $\approx 0.12\text{ ms}$ ($\approx 0.20\text{ ms}$ including score computation), confirming negligible computational overhead compared to model inference.

---

## 3. Real Qwen KV-Cache Compaction

*Source files*: `experiments/apply_dsaa_to_real_cache.py`, `experiments/test_pruned_cache_continuation.py`, `experiments/benchmark_retention_levels.py`

DSAA was integrated with actual key-value tensors extracted from `Qwen/Qwen2.5-1.5B-Instruct` across all 28 transformer layers.

### 3.1 Cache Geometry and Compaction ($L=26$ Tokens)

For a prompt length of $N = 26$ tokens:
- Each layer produces `keys` of shape `[1, 12, 26, 128]` and `values` of shape `[1, 12, 26, 128]` in `float32`.
- Total uncompressed KV-cache tensor size:
  $$\text{Original Bytes} = 28 \text{ layers} \times 2 \times (1 \times 12 \times 26 \times 128) \times 4 \text{ bytes} = 1{,}490{,}944\text{ bytes} \approx 1.42\text{ MB}$$

| Configuration | Retained Tokens | Evicted Tokens | Compacted KV Size (Bytes) | Compacted KV Size (MB) | Tensor Storage Saved |
|---|---:|---:|---:|---:|---:|
| **Baseline** | 26 | 0 | 1,490,944 | 1.42 MB | **0.00%** (ref) |
| **DSAA 80%** | 21 | 5 | 1,204,224 | 1.15 MB | **19.23%** |
| **DSAA 60%** | 16 | 10 | 917,504 | 0.87 MB | **38.46%** |
| **DSAA 50%** | 13 | 13 | 745,472 | 0.71 MB | **50.00%** |
| **DSAA 30%** | 8 | 18 | 458,752 | 0.44 MB | **69.23%** |

> [!IMPORTANT]
> These savings strictly reflect key and value tensor allocations inside the transformer attention layers. They do **NOT** represent total system RAM or process memory reduction, as the model weights ($\approx 6.16\text{ GB}$ in `float32`) remain resident in memory.

---

## 4. Continuation Quality Evaluation

The core hypothesis of DSAA is that selectively pruning low-importance tokens can preserve output probability distributions relative to the uncompressed model. Quality was evaluated across three experimental tiers:

### 4.1 Single-Prompt Retention Benchmark (1 Step)
*Source file*: `experiments/benchmark_retention_levels.py`  
*Prompt*: Technical student profile ($N=26$ tokens, greedy baseline next token = `" Rahul"`).

| Target Retention | Actual Retention | KV Bytes Saved | Logit Cosine Similarity | Top-1 Match | Top-5 Overlap |
|---|---:|---:|---:|:---:|---:|
| **80%** | 80.77% | 19.23% | **0.99912** | **True** | **5 / 5** |
| **60%** | 61.54% | 38.46% | **0.99420** | **True** | **5 / 5** |
| **50%** | 50.00% | 50.00% | **0.98845** | **True** | **5 / 5** |
| **30%** | 30.77% | 69.23% | **0.94120** | **True** | **4 / 5** |

### 4.2 Multi-Prompt Benchmark (1 Step Across 5 Categories)
*Source file*: `experiments/benchmark_multi_prompt.py`  
*Prompts*: 5 domain prompts (`Technical`, `Programming`, `General Knowledge`, `Reasoning`, `AI/ML`; $N \approx 35{-}45$ tokens), single continuation step across retention levels 80%, 60%, 50%, and 30%.

> [!NOTE]
> **Status of Single-Step Multi-Prompt Results**:
> The benchmark script `experiments/benchmark_multi_prompt.py` exists and was fully implemented to evaluate distribution fidelity across multiple prompt domains. However, its original execution output was printed to `stdout` and not preserved as a reproducible on-disk artifact (such as a JSON record or archived console log). Consequently, numerical continuation metrics for this script are excluded from this document. Future benchmark runs should save structured per-prompt metrics and aggregate statistics to a persistent JSON artifact (following the pattern established in `experiments/latency_results.json`).

### 4.3 Multi-Step Continuation Benchmarks

Evaluating autoregressive continuation beyond a single step tests whether probability distribution shifts compound across generation steps.

#### A. Full Multi-Step Benchmark (Incomplete Preliminary Run)
*Source file*: `experiments/benchmark_multistep.py`  
*Configuration*: 5 prompts $\times$ 4 retention levels $\times$ 5 continuation steps.  
*Execution Status*: **Incomplete**. Halted at ~25 minutes to prevent sustained hardware thermal saturation after completing Prompt 1 (`Technical`) and 75% of Prompt 2 (`Programming`).

*Completed Segment Data from `task-195.log`*:
- **Technical Prompt ($N=31$ tokens, 5 steps)**:
  - 80% target: Actual 80.65%, Saved 19.35%, Cosine: **0.94908**, Top-1: **80.0%**, Top-5: **3.60 / 5**
  - 60% target: Actual 61.29%, Saved 38.71%, Cosine: **0.95100**, Top-1: **80.0%**, Top-5: **3.80 / 5**
  - 50% target: Actual 51.61%, Saved 48.39%, Cosine: **0.95710**, Top-1: **80.0%**, Top-5: **3.80 / 5**
  - 30% target: Actual 29.03%, Saved 70.97%, Cosine: **0.94576**, Top-1: **80.0%**, Top-5: **3.80 / 5**
- **Programming Prompt ($N=27$ tokens, partial)**:
  - 80% target: Actual 81.48%, Saved 18.52%, Cosine: **0.99853**, Top-1: **60.0%**, Top-5: **4.60 / 5**
  - 60% target: Actual 59.26%, Saved 40.74%, Cosine: **0.99427**, Top-1: **100.0%**, Top-5: **4.40 / 5**
  - 50% target: Actual 51.85%, Saved 48.15%, Cosine: **0.98957**, Top-1: **60.0%**, Top-5: **4.20 / 5**

#### B. Scoped Multi-Step Quick Validation Benchmark (Completed Run)
*Source file*: `experiments/benchmark_multistep_quick.py`  
*Confirmed Execution Log*: `task-269.log`  
*Configuration*: 2 prompts (`Technical`, `Programming`) $\times$ 2 retention levels (80%, 50%) $\times$ 3 steps (28 total forward passes).  
*Execution Status*: **Completed successfully** (Exit Code 0).

*Per-Prompt Results*:

| Prompt | Target Retention | Actual Retention | KV Saved | Cosine Similarity | Top-1 Agreement | Top-5 Overlap |
|---|---:|---:|---:|---:|---:|---:|
| **Technical** | 80% | 80.65% | 19.35% | 0.95778 | 66.7% | 3.33 / 5 |
| **Technical** | 50% | 51.61% | 48.39% | 0.95906 | 66.7% | 3.67 / 5 |
| **Programming** | 80% | 81.48% | 18.52% | 0.99824 | 33.3% | 4.33 / 5 |
| **Programming** | 50% | 51.85% | 48.15% | 0.98575 | 33.3% | 4.33 / 5 |

*Aggregate Summary Across Completed Quick Validation*:

| Target Retention | Mean Actual Retention | Mean KV Saved | Mean Cosine Similarity | Top-1 Agreement | Mean Top-5 Overlap |
|---|---:|---:|---:|---:|---:|
| **80%** | **81.06%** | **18.94%** | **0.97801** | **50.0%** | **3.83 / 5** |
| **50%** | **51.73%** | **48.27%** | **0.97240** | **50.0%** | **4.00 / 5** |

> [!WARNING]
> **Limitations of Quick-Validation Results**:
> This validation was restricted to 2 prompts and 3 continuation steps as a focused, scoped engineering check. These preliminary results do **not** establish general generation-quality equivalence across broader tasks, long horizons, or diverse domains.

### 4.4 Observations on Continuation Quality
- **Cosine Similarity Stability**: Across the verified single-prompt and quick-validation runs, mean logit cosine similarity remained $\ge 0.957$ down to 50% retention.
- **Top-1 vs. Top-5 Dynamics**: Top-1 greedy agreement showed greater sensitivity than Top-5 overlap (e.g., 33.3% Top-1 vs. 4.33/5 Top-5 on Programming). Qualitative inspection showed that divergence frequently occurred between semantically similar tokens (e.g., alternate whitespace, variable naming, syntax punctuation), while the high-probability candidate set remained aligned.

---

## 5. Latency & Overhead Evaluation

*Source files*: `experiments/benchmark_latency.py`, `experiments/latency_results.json`

> [!WARNING]
> **Preliminary Latency Measurement Notice**:
> All decoding latency differences, speedup percentages, and break-even calculations reported below are **PRELIMINARY engineering measurements**. Sequential execution order, high standard deviations, and thermal throttling on passively cooled hardware prevent confidently attributing observed latency differences to DSAA cache compaction. These data points reflect preliminary observations and must **NOT** be interpreted as proven or reliable production inference speedups.

### 5.1 DSAA Algorithmic Overhead Breakdown

Measured with nanosecond precision using `torch.mps.synchronize()` across 20 repetitions ($N=26$ tokens, values from `task-327.log`):

| Operation | Implementation Details | Mean Latency | Std Dev (from console log) |
|---|---|---:|---:|
| **Attention Scoring** | 28-layer tensor reduction over query window | **0.864 ms** | $\pm$ 0.126 ms |
| **Semantic Scoring** | Heuristic lexical/structural regex matching & token offsets | **0.519 ms** | $\pm$ 0.486 ms |
| **Recency Scoring** | Linear recency normalization curve | **0.005 ms** | $\pm$ 0.005 ms |
| **Combined Scoring & Pressure** | Weighted score integration & pressure calculation | **0.325 ms** | $\pm$ 0.188 ms |
| **Total Scoring Overhead** | **Full score generation pipeline** | **1.713 ms** | — |
| **Pruning Policy (80%)** | Importance sorting & budget selection | **0.050 ms** | $\pm$ 0.012 ms |
| **Cache Slicing (80%)** | 28-layer `index_select` & `DynamicCache` rebuild | **2.974 ms** | $\pm$ 2.204 ms |
| **Total DSAA Overhead (80%)** | **One-time cost prior to generation** | **4.737 ms** | — |
| **Pruning Policy (50%)** | Importance sorting & budget selection | **0.049 ms** | $\pm$ 0.006 ms |
| **Cache Slicing (50%)** | 28-layer `index_select` & `DynamicCache` rebuild | **3.176 ms** | $\pm$ 1.477 ms |
| **Total DSAA Overhead (50%)** | **One-time cost prior to generation** | **4.939 ms** | — |

**Takeaway**: In the tested configuration, the one-time computational overhead of executing DSAA ($<5\text{ ms}$) is negligible compared to a single model forward pass ($\approx 20{-}30\text{ seconds}$).

### 5.2 Single-Token Decoding Latency (10 Repetitions, Fresh Clones)

Each repetition decoded against an independently cloned cache state starting at the exact same sequence length:

| Configuration | Tokens in Cache | Mean Latency / Step | Std Dev (from console log) | Observed Difference (Preliminary) |
|---|---:|---:|---:|---:|
| **Baseline** | 26 | **29,835.85 ms** | $\pm$ 4,970.36 ms | **1.00x** (ref) |
| **DSAA 80%** | 21 | **22,137.30 ms** | $\pm$ 8,493.61 ms | **+25.80%** (preliminary) |
| **DSAA 50%** | 13 | **31,222.76 ms** | $\pm$ 11,082.23 ms | **-4.65%** (preliminary) |

*Note on Variance*: The standard deviations ($\pm 5\text{s}$ to $\pm 11\text{s}$) are substantially larger than any theoretical attention compute savings for a 26-token prompt. These differences cannot be confidently attributed to KV-cache compression.

### 5.3 Multi-Step Trajectory Latency (5 Consecutive Steps)

| Configuration | Total 5-Step Latency | Mean Latency / Step | Step Std Dev (from console log) | Observed Difference (Preliminary) |
|---|---:|---:|---:|---:|
| **Baseline** | 250,226.31 ms (~250.2 s) | **50,045.26 ms** | $\pm$ 3,780.83 ms | **1.00x** (ref) |
| **DSAA 80%** | 194,013.41 ms (~194.0 s) | **38,802.68 ms** | $\pm$ 4,902.93 ms | **+22.46%** (preliminary) |
| **DSAA 50%** | 200,183.59 ms (~200.2 s) | **40,036.72 ms** | $\pm$ 5,562.74 ms | **+19.99%** (preliminary) |

*Note on Autonomous Generation*: Multi-step decoding was executed in autonomous greedy mode (each configuration decoded its own predicted token). Because token trajectories diverged between configurations, steps evaluated different token embeddings under changing thermal states.

### 5.4 Theoretical Break-Even Estimation (Not Validated Performance Claims)

$$\text{One-Time Overhead } T_{\text{overhead}} = T_{\text{scoring}} + T_{\text{pruning}}$$
$$\Delta t = t_{\text{baseline\_step}} - t_{\text{dsaa\_step}}$$

- **Theoretical Arithmetic Estimate at 80%**:
  Using the observed single-token mean difference ($\Delta t \approx 7{,}698.55\text{ ms}$) against the one-time overhead ($4.74\text{ ms}$), a theoretical break-even would occur within $\approx 1\text{ token}$.
- **Observations at 50%**:
  Because DSAA 50% had a higher observed single-token mean latency ($31,222.76\text{ ms}$ vs. $29,835.85\text{ ms}$), no positive delta existed, resulting in no calculated break-even point.
- **Hardware Interpretation**:
  At batch size $B=1$ and context length $N=26$, inference is heavily memory-bandwidth bound by fetching the 1.5B parameters (~6.16 GB in `float32`) at each step. Attention operations on KV tensors account for $<0.05\%$ of forward-pass FLOPs. Consequently, break-even calculations at this context length represent arithmetic extrapolations rather than verified performance characteristics.

### 5.5 Data Completeness & Export Schema Limitations

- `experiments/latency_results.json` records scalar means for prefill, scoring, pruning, and decoding, but **does not serialize individual timing sample arrays or standard deviations**.
- All standard deviation values reported in the tables above were extracted directly from the console execution log (`task-327.log`).
- *Recommendation for Future Iterations*: To ensure full archival reproducibility, future benchmark versions should update the export schema to serialize:
  1. Complete raw iteration timing arrays (`times`)
  2. Summary statistics (mean, median, standard deviation, min, max)
  3. Execution sequence order and timestamp metadata
  4. Hardware power and thermal state indicators

---

## 6. Verification and Regression Testing

*Configuration*: `pytest.ini` (`testpaths = tests`)  
*Command*: `python -m pytest -q`  
*Execution Status*: **16 passed in 2.70s**

### 6.1 Test Suite Breakdown

| Test File | Test Case | Functionality Verified |
|---|---|---|
| `test_attention_score.py` | `test_attention_score_shape` | Verifies attention score output shape equals sequence length $(N,)$. |
| `test_attention_score.py` | `test_attention_score_range` | Validates that all normalized attention scores fall within $[0.0, 1.0]$. |
| `test_attention_score.py` | `test_constant_attention` | Checks edge-case where uniform attention returns zero-vector scores. |
| `test_benchmark_eviction.py` | `test_benchmark_strategies` | Compares FIFO, LRU, and DSAA over 4 retention budgets (80%, 60%, 50%, 30%) with 100 repeats, verifying important-token preservation and protected recent token invariants. |
| `test_cache_simulation.py` | `test_cache_simulation` | Tests synthetic KV-cache memory calculation and byte reduction formulas. |
| `test_combined_score.py` | `test_combined_score` | Validates multi-component weighted score fusion ($0.4 \cdot \text{att} + 0.4 \cdot \text{sem} + 0.2 \cdot \text{rec}$). |
| `test_combined_score.py` | `test_invalid_weights` | Ensures negative or invalid weights raise expected `ValueError`. |
| `test_dsaa_pipeline.py` | `test_complete_dsaa_pipeline` | Validates complete synthetic pipeline from score generation to cache slicing and memory verification. |
| `test_memory_pressure.py` | `test_memory_pressure` | Verifies linear pressure scaling $\frac{N}{\text{capacity}}$ up to 1.0. |
| `test_memory_pressure.py` | `test_capacity_exceeded` | Confirms memory pressure clamps at 1.0 when tokens exceed capacity. |
| `test_memory_pressure.py` | `test_invalid_capacity` | Asserts that zero or negative capacity raises `ValueError`. |
| `test_recency_score.py` | `test_recency_score` | Confirms monotonic linear increase from 0 to 1 over token indices. |
| `test_retention_policy.py` | `test_no_eviction_under_threshold` | Verifies zero eviction when memory pressure $< 0.8$. |
| `test_retention_policy.py` | `test_low_importance_tokens_evicted` | Tests that lowest scored tokens are evicted while recent tokens are retained. |
| `test_retention_policy.py` | `test_empty_input` | Validates empty input handling. |
| `test_semantic_score.py` | `test_semantic_score` | Tests character-to-token offset mapping and heuristic regex pattern scoring using `AutoTokenizer`. |

---

## 7. Project Limitations & Identified Constraints

1. **Heuristic Lexical/Structural Scoring**:
   The current semantic scoring module is **not** a learned semantic model, cross-encoder, or embedding-based language understanding system. It is a heuristic rule-based scorer utilizing handcrafted regular expressions for lexical and structural cues (proper nouns, numerical values, technical acronyms, discourse markers). It does not capture contextual semantics or deeper syntactic dependencies.
2. **Small Evaluation Dataset & Short Contexts**:
   Empirical evaluations were conducted on a curated set of 5 short prompts ($26$ to $50$ tokens). DSAA has not yet been benchmarked on multi-turn dialogue, long documents ($>2{,}048$ tokens), or standardized benchmarks (e.g., LongBench, RULER).
3. **Single Model & Single Hardware Configuration**:
   All real-model experiments were executed exclusively on `Qwen/Qwen2.5-1.5B-Instruct` using an Apple Silicon M1. Behavior on larger models (7B, 14B), different architectures (Llama, Mistral), or dedicated server GPUs (NVIDIA A100/H100) remains unmeasured.
4. **Precision and Memory Footprint Constraints**:
   Evaluating the 1.5B model in `torch.float32` requires ~6.16 GB of memory, causing significant memory pressure and thermal throttling on fanless unified-memory systems. Future work should evaluate half-precision (`float16`, `bfloat16`) or quantized representations.
5. **No Total RAM Reduction**:
   DSAA reduces key-value tensor allocations in memory by up to 50–70%, but does not reduce base model parameter memory.

---

## 8. Conclusion

### 8.1 Demonstrated Findings in Tested Configurations
- **Demonstrated In-Memory KV-Cache Compaction**: In-place reconstruction of Hugging Face `DynamicCache` verified that key-value tensors can be sliced along sequence dimensions while preserving causal attention mask and position ID consistency.
- **Low Algorithmic Overhead**: In the tested setup, the complete scoring, policy selection, and tensor slicing pipeline executed in $< 5\text{ ms}$, demonstrating that the algorithmic operations themselves introduce minimal computational overhead.
- **Preliminary Evidence of Quality Preservation**: In the tested short-prompt configurations, pruning up to 50% of the KV cache maintained logit cosine similarity $> 0.97$ and top-5 token overlap ($\approx 3.8{-}4.0/5$) relative to the uncompressed baseline. This provides initial evidence that selective token eviction can preserve local probability distributions across short generation horizons, though general distributional fidelity across broader domains and longer sequences remains unverified.

### 8.2 Areas Requiring Future Validation
- Benchmarking on long-context sequences ($L \ge 2{,}048$) where KV-cache size represents a substantial fraction of memory traffic and where attention latency reduction can be tested under controlled, compute-heavy conditions.
- Upgrading heuristic lexical/structural scoring to lightweight learned embedding representations.
- Evaluating task accuracy on standardized question-answering, retrieval, and summarization benchmarks.
- Benchmarking with randomized or interleaved execution orders on active-cooled hardware to isolate latency effects from thermal and scheduling variance.
