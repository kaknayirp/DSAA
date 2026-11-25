import time
import json
import statistics
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from dsaa_algorithm.attention_score import calculate_attention_importance
from dsaa_algorithm.semantic_score import calculate_semantic_importance
from dsaa_algorithm.recency_score import calculate_recency_importance
from dsaa_algorithm.combined_score import calculate_combined_score
from dsaa_algorithm.memory_pressure import calculate_memory_pressure
from dsaa_algorithm.retention_policy import select_tokens_to_retain


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CAPACITY = 20

WARMUP_REPS = 3
MEASURE_REPS = 10
SCORING_REPS = 20
GEN_STEPS = 5

PROMPT = """Explain how TCP establishes reliable communication between two computers. Discuss handshakes, acknowledgements, retransmission and error detection."""


def sync_device(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def cache_size_bytes(cache):
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def clone_cache(cache, model):
    cache_data = []
    for layer in cache.layers:
        cache_data.append((layer.keys.clone(), layer.values.clone()))
    return DynamicCache(ddp_cache_data=cache_data, config=model.config)


def build_pruned_cache(original_cache, model, indices, device):
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    cache_data = []
    for layer in original_cache.layers:
        keys = layer.keys.index_select(-2, index_tensor)
        values = layer.values.index_select(-2, index_tensor)
        cache_data.append((keys, values))
    return DynamicCache(ddp_cache_data=cache_data, config=model.config)


def benchmark_scoring(outputs, text, num_tokens, tokenizer, device, reps=SCORING_REPS):
    # Warmup
    att_raw = calculate_attention_importance(outputs.attentions, query_window=4)
    sem_raw = calculate_semantic_importance(text, tokenizer)
    rec_raw = calculate_recency_importance(num_tokens)
    calculate_combined_score(att_raw, sem_raw, rec_raw)
    calculate_memory_pressure(num_tokens, MAX_CAPACITY)

    # 1. Attention Scoring
    att_times = []
    for _ in range(reps):
        sync_device(device)
        t0 = time.perf_counter_ns()
        att = calculate_attention_importance(outputs.attentions, query_window=4)
        sync_device(device)
        t1 = time.perf_counter_ns()
        att_times.append((t1 - t0) / 1e6)

    # 2. Semantic Scoring
    sem_times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        sem = calculate_semantic_importance(text, tokenizer)
        t1 = time.perf_counter_ns()
        sem_times.append((t1 - t0) / 1e6)

    # 3. Recency Scoring
    rec_times = []
    for _ in range(reps):
        sync_device(device)
        t0 = time.perf_counter_ns()
        rec = calculate_recency_importance(num_tokens)
        sync_device(device)
        t1 = time.perf_counter_ns()
        rec_times.append((t1 - t0) / 1e6)

    # 4. Combined Scoring + Memory Pressure
    comb_times = []
    for _ in range(reps):
        sync_device(device)
        t0 = time.perf_counter_ns()
        comb = calculate_combined_score(att, sem, rec)
        press = calculate_memory_pressure(num_tokens, MAX_CAPACITY)
        sync_device(device)
        t1 = time.perf_counter_ns()
        comb_times.append((t1 - t0) / 1e6)

    return {
        "attention": {"mean": statistics.mean(att_times), "std": statistics.stdev(att_times), "scores": att},
        "semantic": {"mean": statistics.mean(sem_times), "std": statistics.stdev(sem_times), "scores": sem},
        "recency": {"mean": statistics.mean(rec_times), "std": statistics.stdev(rec_times), "scores": rec},
        "combined_pressure": {"mean": statistics.mean(comb_times), "std": statistics.stdev(comb_times), "combined": comb, "pressure": press}
    }


def benchmark_pruning(combined_scores, memory_pressure, num_tokens, retention, original_cache, model, device, reps=SCORING_REPS):
    target_count = max(4, min(num_tokens, round(num_tokens * retention)))
    eviction_count = num_tokens - target_count
    norm_p = max(0.0, min(1.0, (memory_pressure - 0.8) / 0.2))
    max_evict = 0.0 if norm_p == 0 else min(1.0, (eviction_count + 0.001) / (num_tokens * norm_p))

    # Warmup
    select_tokens_to_retain(combined_scores, memory_pressure, pressure_threshold=0.8,
                            max_eviction_fraction=max_evict, protected_recent_tokens=4)
    dummy_indices = list(range(target_count))
    build_pruned_cache(original_cache, model, dummy_indices, device)

    # 1. Policy Token Selection
    sel_times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        retained = select_tokens_to_retain(
            combined_scores,
            memory_pressure,
            pressure_threshold=0.8,
            max_eviction_fraction=max_evict,
            protected_recent_tokens=4
        )
        t1 = time.perf_counter_ns()
        sel_times.append((t1 - t0) / 1e6)

    # 2. KV-Cache Slicing & DynamicCache Reconstruction
    recon_times = []
    for _ in range(reps):
        sync_device(device)
        t0 = time.perf_counter_ns()
        pruned_cache = build_pruned_cache(original_cache, model, retained, device)
        sync_device(device)
        t1 = time.perf_counter_ns()
        recon_times.append((t1 - t0) / 1e6)

    return {
        "selection": {"mean": statistics.mean(sel_times), "std": statistics.stdev(sel_times), "retained": retained},
        "reconstruction": {"mean": statistics.mean(recon_times), "std": statistics.stdev(recon_times), "cache": pruned_cache}
    }


def benchmark_single_token_decoding(cache_template, cache_len, next_token, position_ids, model, device, warmup=WARMUP_REPS, reps=MEASURE_REPS):
    mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)

    # Warmup with fresh clones
    for _ in range(warmup):
        fresh_cache = clone_cache(cache_template, model)
        with torch.inference_mode():
            _ = model(
                input_ids=next_token,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=fresh_cache,
                use_cache=True,
                return_dict=True
            )
        sync_device(device)

    # Timed repetitions: each uses a fresh independent cloned cache
    step_times = []
    for _ in range(reps):
        fresh_cache = clone_cache(cache_template, model)
        assert fresh_cache.get_seq_length() == cache_len, "Cache length desynchronized"

        sync_device(device)
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            step_out = model(
                input_ids=next_token,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=fresh_cache,
                use_cache=True,
                return_dict=True
            )
        sync_device(device)
        t1 = time.perf_counter_ns()
        step_times.append((t1 - t0) / 1e6)

        # Validate that cache grew by exactly 1 token
        assert step_out.past_key_values.get_seq_length() == cache_len + 1

    return {
        "mean_ms": statistics.mean(step_times),
        "std_ms": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "times": step_times
    }


def benchmark_multi_step_decoding(cache_template, start_seq_len, next_token, start_pos, model, device, num_steps=GEN_STEPS):
    cache = clone_cache(cache_template, model)
    curr_token = next_token
    total_time_ms = 0.0
    step_latencies = []

    for s in range(num_steps):
        pos = torch.tensor([[start_pos + s]], dtype=torch.long, device=device)
        mask = torch.ones((1, cache.get_seq_length() + 1), dtype=torch.long, device=device)

        sync_device(device)
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            out = model(
                input_ids=curr_token,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True
            )
        sync_device(device)
        t1 = time.perf_counter_ns()

        latency = (t1 - t0) / 1e6
        total_time_ms += latency
        step_latencies.append(latency)

        cache = out.past_key_values
        curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    return {
        "total_ms": total_time_ms,
        "mean_step_ms": statistics.mean(step_latencies),
        "std_step_ms": statistics.stdev(step_latencies) if len(step_latencies) > 1 else 0.0,
        "step_latencies": step_latencies
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 65, flush=True)
    print("      DSAA LATENCY & INFERENCE OVERHEAD BENCHMARK", flush=True)
    print("=" * 65, flush=True)
    print(f"Device:           {device}", flush=True)
    print(f"Model:            {MODEL_NAME}", flush=True)
    print(f"Warmup Reps:      {WARMUP_REPS}", flush=True)
    print(f"Measurement Reps: {MEASURE_REPS}", flush=True)
    print(f"Decoding Steps:   {GEN_STEPS}", flush=True)
    print("-" * 65, flush=True)

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    num_tokens = inputs["input_ids"].shape[1]
    print(f"Input Prompt Length: {num_tokens} tokens\n", flush=True)

    # 1. Prefill
    sync_device(device)
    t_pref_0 = time.perf_counter_ns()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )
    sync_device(device)
    t_pref_1 = time.perf_counter_ns()
    prefill_time_ms = (t_pref_1 - t_pref_0) / 1e6

    original_cache = outputs.past_key_values
    original_bytes = cache_size_bytes(original_cache)

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    initial_position = torch.tensor([[num_tokens]], dtype=torch.long, device=device)

    print(f"Prefill Latency: {prefill_time_ms:.2f} ms")
    print(f"Original KV-Cache Size: {original_bytes:,} bytes ({original_bytes / 1024 / 1024:.2f} MB)", flush=True)

    # 2. Benchmark Scoring Overhead
    print("\n[1/4] Benchmarking Scoring Overhead...", flush=True)
    scoring_results = benchmark_scoring(outputs, PROMPT, num_tokens, tokenizer, device)
    total_scoring_mean = (
        scoring_results["attention"]["mean"] +
        scoring_results["semantic"]["mean"] +
        scoring_results["recency"]["mean"] +
        scoring_results["combined_pressure"]["mean"]
    )

    combined_scores = scoring_results["combined_pressure"]["combined"]
    memory_pressure = scoring_results["combined_pressure"]["pressure"]

    # 3. Benchmark Pruning Overhead
    print("[2/4] Benchmarking Pruning Overhead (80% and 50%)...", flush=True)
    prune_80 = benchmark_pruning(combined_scores, memory_pressure, num_tokens, 0.80, original_cache, model, device)
    prune_50 = benchmark_pruning(combined_scores, memory_pressure, num_tokens, 0.50, original_cache, model, device)

    total_pruning_80 = prune_80["selection"]["mean"] + prune_80["reconstruction"]["mean"]
    total_pruning_50 = prune_50["selection"]["mean"] + prune_50["reconstruction"]["mean"]

    cache_80 = prune_80["reconstruction"]["cache"]
    bytes_80 = cache_size_bytes(cache_80)
    retained_80 = len(prune_80["selection"]["retained"])

    cache_50 = prune_50["reconstruction"]["cache"]
    bytes_50 = cache_size_bytes(cache_50)
    retained_50 = len(prune_50["selection"]["retained"])

    # 4. Benchmark Single-Token Decoding Latency
    print("[3/4] Benchmarking Single-Token Decoding Latency (Fresh Caches)...", flush=True)
    decode_base = benchmark_single_token_decoding(original_cache, num_tokens, next_token, initial_position, model, device)
    decode_80 = benchmark_single_token_decoding(cache_80, retained_80, next_token, initial_position, model, device)
    decode_50 = benchmark_single_token_decoding(cache_50, retained_50, next_token, initial_position, model, device)

    # 5. Benchmark Multi-Step Sequential Decoding
    print("[4/4] Benchmarking Multi-Step Decoding (5 Steps)...", flush=True)
    multi_base = benchmark_multi_step_decoding(original_cache, num_tokens, next_token, num_tokens, model, device)
    multi_80 = benchmark_multi_step_decoding(cache_80, retained_80, next_token, num_tokens, model, device)
    multi_50 = benchmark_multi_step_decoding(cache_50, retained_50, next_token, num_tokens, model, device)

    # ----------------------------------------------------
    # RESULTS REPORTING
    # ----------------------------------------------------
    print("\n" + "=" * 65, flush=True)
    print("                 PER-COMPONENT TIMING (ms)", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Component':<35}{'Mean (ms)':<15}{'Std (ms)':<15}")
    print("-" * 65)
    print(f"{'Attention Scoring':<35}{scoring_results['attention']['mean']:<15.3f}{scoring_results['attention']['std']:<15.3f}")
    print(f"{'Semantic Scoring':<35}{scoring_results['semantic']['mean']:<15.3f}{scoring_results['semantic']['std']:<15.3f}")
    print(f"{'Recency Scoring':<35}{scoring_results['recency']['mean']:<15.3f}{scoring_results['recency']['std']:<15.3f}")
    print(f"{'Combined Scoring & Pressure':<35}{scoring_results['combined_pressure']['mean']:<15.3f}{scoring_results['combined_pressure']['std']:<15.3f}")
    print(f"{'Total Scoring Overhead':<35}{total_scoring_mean:<15.3f}{'-':<15}")
    print("-" * 65)
    print(f"{'Pruning: Token Selection (80%)':<35}{prune_80['selection']['mean']:<15.3f}{prune_80['selection']['std']:<15.3f}")
    print(f"{'Pruning: Cache Slicing (80%)':<35}{prune_80['reconstruction']['mean']:<15.3f}{prune_80['reconstruction']['std']:<15.3f}")
    print(f"{'Total DSAA Overhead (80%)':<35}{total_scoring_mean + total_pruning_80:<15.3f}{'-':<15}")
    print("-" * 65)
    print(f"{'Pruning: Token Selection (50%)':<35}{prune_50['selection']['mean']:<15.3f}{prune_50['selection']['std']:<15.3f}")
    print(f"{'Pruning: Cache Slicing (50%)':<35}{prune_50['reconstruction']['mean']:<15.3f}{prune_50['reconstruction']['std']:<15.3f}")
    print(f"{'Total DSAA Overhead (50%)':<35}{total_scoring_mean + total_pruning_50:<15.3f}{'-':<15}")

    print("\n" + "=" * 65, flush=True)
    print("            KV-CACHE STORAGE SAVINGS", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Configuration':<20}{'Tokens':<12}{'KV Bytes':<15}{'Savings (%)':<15}")
    print("-" * 65)
    print(f"{'Baseline':<20}{num_tokens:<12}{original_bytes:<15}{'0.00%':<15}")
    saved_80_pct = (1.0 - bytes_80 / original_bytes) * 100
    saved_50_pct = (1.0 - bytes_50 / original_bytes) * 100
    print(f"{'DSAA (80% target)':<20}{retained_80:<12}{bytes_80:<15}{f'{saved_80_pct:.2f}%':<15}")
    print(f"{'DSAA (50% target)':<20}{retained_50:<12}{bytes_50:<15}{f'{saved_50_pct:.2f}%':<15}")
    print("* Note: Storage reflects raw KV-cache tensors, not total process RAM.")

    print("\n" + "=" * 65, flush=True)
    print("         PER-TOKEN DECODING LATENCY & SPEEDUP", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Configuration':<20}{'Mean (ms)':<15}{'Std (ms)':<15}{'Speedup (%)':<15}")
    print("-" * 65)
    base_mean = decode_base["mean_ms"]
    print(f"{'Baseline':<20}{base_mean:<15.2f}{decode_base['std_ms']:<15.2f}{'1.00x (ref)':<15}")

    sp_80 = (base_mean - decode_80["mean_ms"]) / base_mean * 100
    print(f"{'DSAA (80%)':<20}{decode_80['mean_ms']:<15.2f}{decode_80['std_ms']:<15.2f}{f'{sp_80:+.2f}%':<15}")

    sp_50 = (base_mean - decode_50["mean_ms"]) / base_mean * 100
    print(f"{'DSAA (50%)':<20}{decode_50['mean_ms']:<15.2f}{decode_50['std_ms']:<15.2f}{f'{sp_50:+.2f}%':<15}")

    print("\n" + "=" * 65, flush=True)
    print("        MULTI-STEP DECODING LATENCY (5 Steps)", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Configuration':<20}{'Total (ms)':<15}{'Per-Step (ms)':<15}{'Per-Step Std':<15}")
    print("-" * 65)
    print(f"{'Baseline':<20}{multi_base['total_ms']:<15.2f}{multi_base['mean_step_ms']:<15.2f}{multi_base['std_step_ms']:<15.2f}")
    print(f"{'DSAA (80%)':<20}{multi_80['total_ms']:<15.2f}{multi_80['mean_step_ms']:<15.2f}{multi_80['std_step_ms']:<15.2f}")
    print(f"{'DSAA (50%)':<20}{multi_50['total_ms']:<15.2f}{multi_50['mean_step_ms']:<15.2f}{multi_50['std_step_ms']:<15.2f}")

    print("\n" + "=" * 65, flush=True)
    print("            END-TO-END BREAK-EVEN ANALYSIS", flush=True)
    print("=" * 65, flush=True)
    for name, dec_res, ovh in [("DSAA 80%", decode_80, total_scoring_mean + total_pruning_80),
                               ("DSAA 50%", decode_50, total_scoring_mean + total_pruning_50)]:
        delta_t = base_mean - dec_res["mean_ms"]
        print(f"--- {name} ---")
        print(f"  One-time DSAA Overhead:  {ovh:.2f} ms")
        print(f"  Per-token Decoding Time: {dec_res['mean_ms']:.2f} ms vs Baseline {base_mean:.2f} ms")
        print(f"  Delta per token:         {delta_t:+.3f} ms")

        if delta_t > 0.05:
            breakeven = ovh / delta_t
            print(f"  Break-even Point:        {breakeven:.1f} generated tokens")
            print(f"  Status: Break-even achievable after ~{round(breakeven)} generation steps.")
        else:
            print("  Break-even Point:        None (No per-token decoding speedup)")
            print("  Status: No break-even point exists under measured conditions.")
            print("          Decoding latency is memory-bandwidth bound at batch_size=1,")
            print("          so KV-cache size reduction saves tensor memory without speeding")
            print("          up single-token weight-fetching forward passes.")

    # Save structured results
    summary_data = {
        "model": MODEL_NAME,
        "device": device,
        "prompt_tokens": num_tokens,
        "prefill_time_ms": prefill_time_ms,
        "scoring_overhead_ms": {
            "attention": scoring_results["attention"]["mean"],
            "semantic": scoring_results["semantic"]["mean"],
            "recency": scoring_results["recency"]["mean"],
            "combined_pressure": scoring_results["combined_pressure"]["mean"],
            "total": total_scoring_mean
        },
        "pruning_overhead_ms": {
            "80_selection": prune_80["selection"]["mean"],
            "80_reconstruction": prune_80["reconstruction"]["mean"],
            "80_total": total_pruning_80,
            "50_selection": prune_50["selection"]["mean"],
            "50_reconstruction": prune_50["reconstruction"]["mean"],
            "50_total": total_pruning_50
        },
        "kv_cache_bytes": {
            "baseline": original_bytes,
            "dsaa_80": bytes_80,
            "dsaa_50": bytes_50
        },
        "single_token_decoding_ms": {
            "baseline": decode_base["mean_ms"],
            "dsaa_80": decode_80["mean_ms"],
            "dsaa_50": decode_50["mean_ms"]
        },
        "multi_step_decoding_ms": {
            "baseline_per_step": multi_base["mean_step_ms"],
            "dsaa_80_per_step": multi_80["mean_step_ms"],
            "dsaa_50_per_step": multi_50["mean_step_ms"]
        }
    }

    with open("experiments/latency_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nResults saved to experiments/latency_results.json")
    print("Latency benchmark completed successfully!", flush=True)


if __name__ == "__main__":
    main()
