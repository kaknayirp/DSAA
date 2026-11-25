# Dynamic Semantic-Aware Attention (DSAA)

Dynamic Semantic-Aware Attention (DSAA) is an intelligent, post-hoc key-value (KV) cache compression framework designed for autoregressive Large Language Models (LLMs). DSAA reduces KV-cache memory consumption during inference by dynamically evaluating token importance across multiple signals and evicting low-utility tokens under memory pressure, preserving model continuation quality without requiring fine-tuning or retraining.

---

## 1. Overview & Problem Statement

In modern autoregressive transformers, the generation phase requires caching the key and value representations of all prior tokens in the sequence (the **KV cache**) to avoid redundant computation during self-attention.

- **The Problem**: As sequence length $L$ expands, KV-cache storage scales linearly with sequence length ($O(L)$ per layer/head), while full self-attention computation scales quadratically ($O(L^2)$). For long-context sequences, high-concurrency serving, or memory-constrained edge hardware, KV-cache memory quickly exceeds accelerator capacity.
- **DSAA Approach**: Rather than relying on rigid chronological eviction policies such as First-In-First-Out (**FIFO**) or access-time Least-Recently-Used (**LRU**)—which indiscriminately evict critical prompt instructions or named entities—DSAA dynamically evaluates token importance using a combination of transformer attention weights, lexical/structural importance heuristics, and temporal recency. Tokens are selectively pruned only when memory pressure exceeds safe operational thresholds.

---

## 2. Key Features

- **Multi-Factor Importance Scoring**:
  - **Attention Importance**: Aggregates attention weights across layers and attention heads over a trailing query window.
  - **Heuristic Semantic Scoring**: Fast regex-based structural pattern matching that boosts proper names, numbers, domain keywords, and causal connectives.
  - **Temporal Recency Scoring**: A normalized linear recency curve ensuring recent tokens remain accessible.
  - **Unified Importance Fusion**: Blends attention, semantic, and recency scores into a balanced importance score per token.
- **Pressure-Aware Retention Policy**:
  - Eviction is triggered only when sequence length exceeds a defined memory pressure threshold (`pressure_threshold=0.8`).
  - Strict preservation of recent tokens (`protected_recent_tokens=4`) to maintain generation flow and attention sinks.
  - Retained KV-cache tokens strictly preserve their original relative chronological order.
- **Native Hugging Face Integration**:
  - Implements tensor slicing and cache reconstruction directly on Hugging Face's `DynamicCache` architecture across all 28 layers of `Qwen/Qwen2.5-1.5B-Instruct`.
- **Rigorous Verification & Benchmarking**:
  - Comprehensive synthetic multi-policy cache eviction simulations.
  - Real-model KV-cache compaction, continuation quality evaluations, and profiling suites.
  - 16 unit and regression tests passing under `pytest`.

---

## 3. Algorithm Details & Defaults

All default parameters and logic are verified directly against the implementation in `dsaa_algorithm/`.

```
           ┌────────────────────────────────────────┐
           │      Pre-eviction Attention Cache      │
           └───────────────────┬────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌─────────────────┐
│  Attention   │       │   Semantic   │       │     Recency     │
│ Importance   │       │  Importance  │       │   Importance    │
│ (Window = 4) │       │ (Regex/Span) │       │ (Linear Ramp)   │
└──────┬───────┘       └──────┬───────┘       └────────┬────────┘
       │                      │                        │
       │ (w = 0.4)            │ (w = 0.4)              │ (w = 0.2)
       └──────────────────────┼────────────────────────┘
                              ▼
               ┌──────────────────────────────┐
               │   Combined Importance (S)    │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │    Memory Pressure Check     │
               │  (Pressure > Threshold 0.8)  │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Retention Policy Selection  │
               │  (Protect 4 Recent Tokens)   │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Chronologically Sliced KV   │
               └──────────────────────────────┘
```

### 3.1 Attention Importance (`dsaa_algorithm/attention_score.py`)
Extracts multi-head attention tensors with shape `[layers, batch, heads, query, key]`. Attention weights are averaged across layers, batches, and heads. To reflect current contextual focus, scores are computed over the trailing query positions (default `query_window=4`) and min-max normalized to $[0, 1]$:
$$\text{AttentionScore}(j) = \text{Normalize}\left(\frac{1}{W} \sum_{q=-W}^{-1} \bar{A}_{q, j}\right)$$

### 3.2 Heuristic Semantic Scoring (`dsaa_algorithm/semantic_score.py`)
A fast tokenizer-aligned rule-based heuristic that matches key character spans against patterns:
- Proper nouns / Title-case entities (`r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"`): Base score $0.9$
- Numerical values & percentages (`r"\b\d+(?:\.\d+)?%?\b"`): Base score $0.9$
- Technical / Domain keywords (`AI|ML|NLP|LLM|Python|PyTorch`): Base score $0.8$
- Causal connectives (`important|because|therefore|must|should|requires`): Base score $0.7$
- Default token baseline: $0.1$  
Token scores are mapped via tokenizer offset spans and min-max normalized to $[0, 1]$.

### 3.3 Recency Scoring (`dsaa_algorithm/recency_score.py`)
Assigns a linear ramp prioritizing newer tokens:
$$\text{RecencyScore}(i) = \frac{i}{N - 1}, \quad i \in \{0, \dots, N-1\}$$
Handled edge cases: $N=0 \to \emptyset$, $N=1 \to [1.0]$.

### 3.4 Combined Scoring (`dsaa_algorithm/combined_score.py`)
Fuses individual scores using the verified default weights:
$$\text{CombinedScore} = 0.4 \cdot \text{Attention} + 0.4 \cdot \text{Semantic} + 0.2 \cdot \text{Recency}$$
Weights are validated non-negative and automatically normalized if their sum differs from $1.0$.

### 3.5 Memory Pressure & Retention Policy (`dsaa_algorithm/retention_policy.py`)
- **Memory Pressure** (`dsaa_algorithm/memory_pressure.py`):
  $$\text{Pressure} = \min\left(\frac{N}{\text{max\_capacity}}, 1.0\right)$$
- **Eviction Gating**: If $\text{Pressure} \le \text{pressure\_threshold}$ (default `0.8`), **zero eviction occurs** and all tokens are retained.
- **Budget Slicing**: When $\text{Pressure} > 0.8$, the eviction fraction scales linearly up to `max_eviction_fraction=0.5`.
- **Protected Recent Tokens**: The trailing `protected_recent_tokens=4` tokens are exempt from eviction.
- **Chronological Invariant**: Retained token indices are strictly sorted in their original sequential order before KV-cache slicing.

---

## 4. Repository Structure

```
DSAA/
├── dsaa_algorithm/              # Core algorithmic package (pure Python/PyTorch)
│   ├── attention_score.py       # Query-window multi-head attention reduction
│   ├── semantic_score.py        # Tokenizer-aligned heuristic lexical scorer
│   ├── recency_score.py         # Linear recency scoring
│   ├── combined_score.py        # Weighted score fusion (0.4 / 0.4 / 0.2)
│   ├── memory_pressure.py       # Capacity ratio calculation
│   └── retention_policy.py      # Threshold gating, recent token protection & selection
├── evaluation/                  # Synthetic cache simulation & baseline comparison
│   ├── benchmark_eviction.py    # FIFO vs. LRU vs. DSAA eviction benchmarking
│   ├── simulate_cache.py        # Synthetic KV tensor allocation & measurement
│   └── run_dsaa_simulation.py   # End-to-end synthetic pipeline simulation
├── experiments/                 # Real-model evaluation with Qwen2.5-1.5B-Instruct
│   ├── apply_dsaa_to_real_cache.py          # DynamicCache tensor extraction & pruning
│   ├── benchmark_retention_levels.py        # Single-prompt retention sweep (80%, 60%, 50%, 30%)
│   ├── benchmark_multi_prompt.py            # Multi-prompt single-step evaluation (5 domains)
│   ├── benchmark_multistep.py               # Full multi-step autoregressive benchmark
│   ├── benchmark_multistep_quick.py         # Scoped 28-pass quick multi-step validation
│   ├── benchmark_latency.py                 # Algorithmic overhead & decoding profiling
│   └── latency_results.json                 # Raw JSON output from latency benchmark
├── tests/                       # Unit and regression test suite
│   ├── test_attention_score.py
│   ├── test_semantic_score.py
│   ├── test_recency_score.py
│   ├── test_combined_score.py
│   ├── test_memory_pressure.py
│   ├── test_retention_policy.py
│   ├── test_cache_simulation.py
│   ├── test_benchmark_eviction.py
│   └── test_dsaa_pipeline.py
├── docs/                        # Project documentation
│   ├── architecture.md          # Initial high-level design specification
│   └── experimental_evaluation.md # Consolidated scientific evaluation report
├── pytest.ini                   # Pytest configuration restricting discovery to tests/
└── requirements.txt             # Project dependencies
```

> [!NOTE]
> Roadmap directories (`backend/`, `frontend/`, `llm_engine/`) currently contain placeholder files. Model loading and KV-cache manipulation utilities currently reside inside the standalone scripts in `experiments/`.

---

## 5. Experimental Results

All reported values are verified against experimental runs and documented in [`docs/experimental_evaluation.md`](docs/experimental_evaluation.md).

### 5.1 Synthetic Eviction Benchmark
*Source*: `evaluation/benchmark_eviction.py`, `tests/test_benchmark_eviction.py`  
Evaluated on a 4-layer synthetic KV cache ($N=100$ tokens, 10 designated important tokens, 100 repetitions):

| Retention Budget | Target Retained | FIFO Important Retained | LRU Important Retained | DSAA Important Retained |
|---|---:|---:|---:|---:|
| **80%** | 80 tokens | 7 / 10 (70%) | 9 / 10 (90%) | **8 / 10 (80%)** |
| **60%** | 60 tokens | 5 / 10 (50%) | 6 / 10 (60%) | **6 / 10 (60%)** |
| **50%** | 50 tokens | 4 / 10 (40%) | 4 / 10 (40%) | **4 / 10 (40%)** |
| **30%** | 30 tokens | 2 / 10 (20%) | 0 / 10 (0%) | **4 / 10 (40%)** |

*Takeaway*: At extreme memory pressure (30% retention budget), traditional LRU dropped to 0% retention of important tokens, while DSAA preserved 40% due to multi-signal scoring and protected trailing tokens.

---

### 5.2 Real Qwen2.5-1.5B KV-Cache Tensor Storage Savings
*Source*: `experiments/apply_dsaa_to_real_cache.py`, `experiments/benchmark_retention_levels.py`  
Tested on `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, 12 key/value heads, head dimension 128, `float32`) with prompt length $N=26$ tokens:

$$\text{Original KV Cache Tensor Size} = 28 \text{ layers} \times 2 \times (1 \times 12 \times 26 \times 128) \times 4 \text{ bytes} = 1{,}490{,}944\text{ bytes} \approx 1.42\text{ MB}$$

| Configuration | Retained Tokens | Evicted Tokens | Compacted KV Size (Bytes) | Compacted KV Size (MB) | KV Tensor Storage Saved |
|---|---:|---:|---:|---:|---:|
| **Baseline** | 26 | 0 | 1,490,944 | 1.42 MB | **0.00%** (ref) |
| **DSAA 80%** | 21 | 5 | 1,204,224 | 1.15 MB | **19.23%** |
| **DSAA 60%** | 16 | 10 | 917,504 | 0.87 MB | **38.46%** |
| **DSAA 50%** | 13 | 13 | 745,472 | 0.71 MB | **50.00%** |
| **DSAA 30%** | 8 | 18 | 458,752 | 0.44 MB | **69.23%** |

> [!IMPORTANT]
> **Storage Clarification**: These savings represent **KV-cache tensor allocations** inside attention layers. They do **NOT** represent total system RAM or process resident memory reduction, because model weights ($\approx 6.16\text{ GB}$ in `float32`) remain resident in memory.

---

### 5.3 Continuation Quality Evaluation

#### A. Single-Step Multi-Prompt Benchmark
*Source*: `experiments/benchmark_multi_prompt.py`  
The single-step multi-prompt benchmark script was implemented to evaluate probability distribution preservation across 5 distinct prompt categories (`Technical`, `Programming`, `General Knowledge`, `Reasoning`, `AI/ML`; $N \approx 35{-}45$ tokens) across multiple retention targets (80%, 60%, 50%, 30%).

> [!NOTE]
> **Status of Multi-Prompt Results**:
> Although the benchmark script exists and was implemented, its original output was printed directly to console `stdout` and not archived as a persistent, reproducible on-disk artifact (such as a structured JSON record). Numerical results for this benchmark are therefore excluded from documented tables until a structured execution run with JSON archival is completed.

#### B. Scoped Multi-Step Quick Validation (3 Autoregressive Steps)
*Source*: `experiments/benchmark_multistep_quick.py`  
*Confirmed Execution Log*: `task-269.log`  
Completed multi-step validation across 2 domains (`Technical`, `Programming`) over 3 autoregressive steps (28 total forward passes):

| Prompt Domain | Target Retention | Actual Retention | KV Saved | Cosine Similarity | Top-1 Agreement | Top-5 Overlap |
|---|---:|---:|---:|---:|---:|---:|
| **Technical** | 80% | 80.65% | 19.35% | 0.95778 | 66.7% | 3.33 / 5 |
| **Technical** | 50% | 51.61% | 48.39% | 0.95906 | 66.7% | 3.67 / 5 |
| **Programming** | 80% | 81.48% | 18.52% | 0.99824 | 33.3% | 4.33 / 5 |
| **Programming** | 50% | 51.85% | 48.15% | 0.98575 | 33.3% | 4.33 / 5 |
| **Aggregate Mean** | **80%** | **81.06%** | **18.94%** | **0.97801** | **50.0%** | **3.83 / 5** |
| **Aggregate Mean** | **50%** | **51.73%** | **48.27%** | **0.97240** | **50.0%** | **4.00 / 5** |

> [!WARNING]
> **Limitations of Quick-Validation Results**:
> This validation was restricted to 2 prompts and 3 continuation steps as a small, scoped engineering check. These preliminary results do **not** establish general generation-quality equivalence across broader tasks or long sequence lengths.

---

### 5.4 Algorithmic Overhead & Latency Evaluation Status
*Source*: `experiments/benchmark_latency.py`, `experiments/latency_results.json`

#### One-Time DSAA Algorithmic Overhead (20 Repetitions, $N=26$ Tokens)
Measured with `torch.mps.synchronize()` on Apple Silicon MPS:

| Component | Operation | Mean Latency |
|---|---|---:|
| **Attention Scoring** | 28-layer tensor reduction over query window ($W=4$) | **0.864 ms** |
| **Semantic Scoring** | Heuristic regex pattern matching & offset mapping | **0.519 ms** |
| **Recency Scoring** | Linear recency normalization curve | **0.005 ms** |
| **Combined & Pressure** | Weighted score integration & capacity ratio | **0.325 ms** |
| **Total Scoring Overhead** | **Full score generation pipeline** | **1.713 ms** |
| **Pruning Policy (80%)** | Importance ranking & budget selection | **0.050 ms** |
| **Cache Slicing (80%)** | 28-layer tensor `index_select` & `DynamicCache` rebuild | **2.974 ms** |
| **Total DSAA Overhead (80%)** | **One-time pruning execution** | **4.737 ms** |

*Takeaway*: The computational overhead of running DSAA is $< 5\text{ ms}$, which is negligible compared to model inference.

> [!WARNING]
> **Status of Decoding Latency Measurements (Preliminary)**:  
> In the evaluated hardware setup (Apple Silicon M1, 8 GB RAM, PyTorch MPS backend in `float32`), single-token forward passes take tens of seconds per step ($\approx 22{-}31\text{ s}$) because inference at batch size 1 is memory-bandwidth bound by loading the $1.54\text{B}$ model parameters ($\approx 6.16\text{ GB}$).  
> Measured decoding latency exhibited high standard deviations ($\pm 5\text{ s}$ to $\pm 11\text{ s}$) due to hardware thermal throttling and sequential non-interleaved execution order. Consequently, **observed decoding differences are preliminary and cannot be confidently attributed to KV-cache compaction**. DSAA does not currently claim validated end-to-end decoding speedups.

---

## 6. Installation & Usage

### 6.1 Prerequisites & Setup
- Python 3.10+ (tested under Python 3.12.7 on macOS Apple Silicon)
- PyTorch 2.5+
- Hugging Face `transformers` and `accelerate`

Clone the repository and enter the project directory:
```bash
git clone https://github.com/kaknayirp/DSAA.git
cd DSAA/DSAA
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

*(Optional)* If executing in an offline or sandboxed environment with cached Hugging Face weights:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 6.2 Running Tests
The repository includes a comprehensive regression suite configured via `pytest.ini` (`testpaths = tests`):
```bash
python -m pytest -q
```
*Expected output*: `16 passed in ~2.5s`

### 6.3 Running Core Experiments
The project is structured as a collection of runnable modules and scripts:

```bash
# 1. Run synthetic multi-policy eviction benchmark (FIFO vs. LRU vs. DSAA)
PYTHONPATH=. python tests/test_benchmark_eviction.py

# 2. Run real Qwen2.5-1.5B KV-cache extraction and tensor compaction
PYTHONPATH=. python experiments/apply_dsaa_to_real_cache.py

# 3. Run multi-prompt single-step continuation benchmark
PYTHONPATH=. python experiments/benchmark_multi_prompt.py

# 4. Run scoped 28-pass multi-step continuation validation
PYTHONPATH=. python experiments/benchmark_multistep_quick.py

# 5. Run algorithmic overhead profiling
PYTHONPATH=. python experiments/benchmark_latency.py
```

---

## 7. Current Limitations

1. **Heuristic Semantic Scorer**: The current semantic scoring component relies on regex patterns (proper nouns, numbers, domain keywords) rather than learned embeddings or transformer representations.
2. **Short Prompt Evaluation**: Validation has been conducted on short prompt sequences ($N = 26{-}45$ tokens). Long-context sequences ($L \ge 2{,}048$) where KV-cache size represents a significant fraction of total memory bandwidth remain untested.
3. **Single Model Architecture**: Experiments are currently standardized on `Qwen/Qwen2.5-1.5B-Instruct`. Multi-architecture generalization (Llama, Mistral, Gemma) has not yet been benchmarked.
4. **Single-Device Evaluation**: Profiling was conducted on Apple Silicon MPS using `float32`. CUDA hardware, quantized precisions (`float16`, `bfloat16`, `int4`), and optimized inference engines (FlashAttention, vLLM) have not been evaluated.
5. **Preliminary Latency Attribution**: As detailed in Section 5.4, decoding speedups are preliminary engineering observations subject to high variance and thermal throttling.
6. **Task-Specific Accuracy**: Evaluation has focused on continuation metrics (logit cosine similarity, top-1 agreement, top-5 overlap). Downstream task benchmarks (MMLU, GSM8K, summarization) remain to be evaluated.

---

## 8. Future Roadmap

- **Long-Context Evaluation**: Benchmark DSAA on long-document retrieval and long-sequence generation tasks ($L = 4\text{K} - 32\text{K}$ tokens).
- **Learned Semantic Scoring**: Replace regex heuristics with lightweight learned token representations or cross-layer projection heads.
- **Inference Engine Integration**: Adapt DSAA's retention policy for PagedAttention kernels and vLLM / TensorRT-LLM runtimes.
- **Modular Engine Package**: Consolidate model extraction and cache slicing utilities into the standalone `llm_engine` module.
- **Interactive UI**: Develop the planned FastAPI backend and React frontend dashboard for interactive KV-cache visualization.

---

## 9. Citation & Artifact Traceability

| Claim / Metric | Primary Source File | Associated Log / Result Artifact |
|---|---|---|
| Synthetic Eviction Results | `evaluation/benchmark_eviction.py` | `tests/test_benchmark_eviction.py` |
| Real KV-Cache Tensor Storage | `experiments/apply_dsaa_to_real_cache.py` | `docs/experimental_evaluation.md` (Sec. 3) |
| Multi-Prompt Continuation Benchmark | `experiments/benchmark_multi_prompt.py` | Script implemented; numerical table excluded (unpreserved output artifact) |
| Multi-Step Quick Validation (28 Passes) | `experiments/benchmark_multistep_quick.py` | `task-269.log` (`docs/experimental_evaluation.md` Sec. 4.3 B) |
| Algorithmic Overhead ($<5\text{ ms}$) | `experiments/benchmark_latency.py` | `experiments/latency_results.json` |
| Consolidated Experimental Evaluation | `docs/experimental_evaluation.md` | Full methodology & findings |
