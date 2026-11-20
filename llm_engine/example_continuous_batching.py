"""
Standalone demonstration for Continuous Batching and Request Scheduling.

Demonstrates:
1. Initializing RequestScheduler with max_batch_size=2.
2. Submitting requests with varying prompt lengths and varying generation lengths.
3. Stepping through scheduling iterations to show requests dynamically entering and leaving batches.
4. Demonstrating request isolation, cache cleanup, and Prompt 2 CacheReport integration.
"""

import sys
import logging
from llm_engine.config import ModelConfig
from llm_engine.generation_config import GenerationConfig
from llm_engine.scheduler_config import SchedulerConfig
from llm_engine.scheduler import RequestScheduler
from llm_engine.inference_engine import InferenceEngine
from llm_engine.model_loader import ModelArtifacts
from llm_engine.test_inference_engine import StatefulCausalLM, MockTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_continuous_batching_demo():
    print("=" * 80)
    print("  DSAA CONTINUOUS BATCHING DEMONSTRATION")
    print("=" * 80)

    # 1. Setup simulated model matching Qwen2.5-1.5B specs
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

    # 2. Configure Scheduler: max_batch_size = 2
    scheduler_cfg = SchedulerConfig(
        max_batch_size=2,
        max_active_requests=4,
        max_queued_requests=10,
        scheduling_policy="fcfs",
    )
    scheduler = RequestScheduler(engine, config=scheduler_cfg)

    # 3. Submit 4 requests with varying prompt lengths and generation targets:
    # Req A: prompt len 3, target 2 tokens (completes quickly)
    # Req B: prompt len 6, target 5 tokens (longer generation)
    # Req C: prompt len 8, target 3 tokens (waits in queue initially)
    # Req D: prompt len 4, target 4 tokens (waits in queue initially)
    print("\n[1] Submitting 4 Requests to Scheduler Queue:")
    id_a = scheduler.submit([1, 2, 3], GenerationConfig(max_new_tokens=2, eos_token_id=999))
    id_b = scheduler.submit([10, 20, 30, 40, 50, 60], GenerationConfig(max_new_tokens=5, eos_token_id=999))
    id_c = scheduler.submit([100, 101, 102, 103, 104, 105, 106, 107], GenerationConfig(max_new_tokens=3, eos_token_id=999))
    id_d = scheduler.submit([200, 201, 202, 203], GenerationConfig(max_new_tokens=4, eos_token_id=999))

    label_map = {id_a: "Req A (len=3, gen=2)", id_b: "Req B (len=6, gen=5)", id_c: "Req C (len=8, gen=3)", id_d: "Req D (len=4, gen=4)"}
    for rid, lbl in label_map.items():
        print(f"    - ID '{rid}': {lbl}")

    print(f"\nInitial State: Queued={scheduler.queued_count}, Active={scheduler.active_count}")
    print("-" * 80)
    print("Executing Scheduler Step-by-Step (Continuous Batching Trace):")
    print("-" * 80)

    step_idx = 0
    while scheduler.queued_count > 0 or scheduler.active_count > 0:
        step_idx += 1
        report = scheduler.step()

        admitted_labels = [label_map.get(rid, rid) for rid in report.admitted_requests]
        completed_labels = [label_map.get(rid, rid) for rid in report.completed_requests]
        active_labels = [label_map.get(rid, rid) for rid in report.active_requests]

        print(f"Step {step_idx:2d}:")
        print(f"   Admitted to Batch: {admitted_labels if admitted_labels else 'None'}")
        print(f"   Completed & Freed: {completed_labels if completed_labels else 'None'}")
        print(f"   Active Decoding:   {active_labels if active_labels else 'Empty'}")
        print(f"   Waiting in Queue:  {report.queued_count} requests")
        print()

    print("=" * 80)
    print("  ALL REQUESTS FINISHED. RESULTS SUMMARY:")
    print("=" * 80)
    stats = scheduler.stats()
    print(f"Total Steps Executed:   {stats['step_count']}")
    print(f"Status Breakdown:       {stats['status_breakdown']}")

    for rid, lbl in label_map.items():
        res = scheduler.get_result(rid)
        print(f"\nResult for {lbl} [ID: {rid}]:")
        print(f"   Prompt Tokens:    {res.num_prompt_tokens}")
        print(f"   Generated Tokens: {res.num_generated_tokens}")
        print(f"   Finish Reason:    {res.finish_reason}")
        if res.final_cache_report:
            print(f"   KV Cache Memory:  {res.final_cache_report.total_kb:.2f} KB ({res.final_cache_report.seq_len} tokens in cache)")


if __name__ == "__main__":
    run_continuous_batching_demo()
