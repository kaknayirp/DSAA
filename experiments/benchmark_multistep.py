import torch
import statistics

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
STEPS = 5

RETENTION_LEVELS = [0.80, 0.60, 0.50, 0.30]

PROMPTS = {
    "Technical": """
    Explain how TCP establishes reliable communication between
    two computers. Discuss handshakes, acknowledgements,
    retransmission and error detection.
    """,

    "Programming": """
    Explain how to find the second largest element in a Python
    array. Discuss duplicates, edge cases and time complexity.
    """,

    "General Knowledge": """
    Explain the importance of solar and wind energy.
    Discuss environmental benefits and limitations.
    """,

    "Reasoning": """
    A student has four hours to study. Mathematics requires
    two hours, programming one hour and artificial intelligence
    two hours. Suggest a practical schedule and explain
    the trade-offs.
    """,

    "AI/ML": """
    Explain supervised and unsupervised learning.
    Discuss classification, regression and clustering
    with practical examples.
    """
}


def cache_size_bytes(cache):
    total = 0

    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()

    return total


def build_pruned_cache(original_cache, model, indices, device):

    index_tensor = torch.tensor(
        indices,
        dtype=torch.long,
        device=device
    )

    cache_data = []

    for layer in original_cache.layers:
        keys = layer.keys.index_select(-2, index_tensor)
        values = layer.values.index_select(-2, index_tensor)
        cache_data.append((keys, values))

    return DynamicCache(
        ddp_cache_data=cache_data,
        config=model.config
    )


def evaluate_retention(
    text,
    retention,
    tokenizer,
    model,
    device
):

    inputs = tokenizer(text, return_tensors="pt").to(device)
    num_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )

    original_cache = outputs.past_key_values
    original_bytes = cache_size_bytes(original_cache)

    # Calculate importance scores
    attention = calculate_attention_importance(
        outputs.attentions,
        query_window=4
    )

    semantic = calculate_semantic_importance(text, tokenizer)
    recency = calculate_recency_importance(num_tokens)

    combined = calculate_combined_score(
        attention,
        semantic,
        recency
    )

    if len(combined) != num_tokens:
        raise ValueError("Score length mismatch")

    pressure = calculate_memory_pressure(
        num_tokens,
        MAX_CAPACITY
    )

    # Calculate target eviction budget
    target_count = round(num_tokens * retention)
    target_count = max(4, min(num_tokens, target_count))

    eviction_count = num_tokens - target_count

    normalized_pressure = max(
        0.0,
        min(1.0, (pressure - 0.8) / 0.2)
    )

    if normalized_pressure == 0:
        max_eviction_fraction = 0.0
    else:
        max_eviction_fraction = min(
            1.0,
            (eviction_count + 0.001)
            / (num_tokens * normalized_pressure)
        )

    retained_indices = select_tokens_to_retain(
        combined,
        pressure,
        pressure_threshold=0.8,
        max_eviction_fraction=max_eviction_fraction,
        protected_recent_tokens=4
    )

    pruned_cache = build_pruned_cache(
        original_cache,
        model,
        retained_indices,
        device
    )

    # Record compacted cache size immediately after pruning, before generation
    pruned_bytes = cache_size_bytes(pruned_cache)
    memory_saved = (1.0 - pruned_bytes / original_bytes) * 100.0
    actual_retention = (len(retained_indices) / num_tokens) * 100.0
    target_retention = retention * 100.0

    # Initial next token from the original prefill
    shared_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    baseline_cache = original_cache
    dsaa_cache = pruned_cache

    baseline_mask = torch.ones(
        (1, num_tokens + 1),
        dtype=torch.long,
        device=device
    )

    dsaa_mask = torch.ones(
        (1, len(retained_indices) + 1),
        dtype=torch.long,
        device=device
    )

    position = torch.tensor(
        [[num_tokens]],
        dtype=torch.long,
        device=device
    )

    # Start both caches with the same first generated token
    with torch.inference_mode():

        baseline_step = model(
            input_ids=shared_token,
            attention_mask=baseline_mask,
            position_ids=position,
            past_key_values=baseline_cache,
            use_cache=True,
            return_dict=True
        )

        dsaa_step = model(
            input_ids=shared_token,
            attention_mask=dsaa_mask,
            position_ids=position,
            past_key_values=dsaa_cache,
            use_cache=True,
            return_dict=True
        )

    baseline_cache = baseline_step.past_key_values
    dsaa_cache = dsaa_step.past_key_values

    cosine_scores = []
    top1_matches = []
    top5_overlaps = []

    # Compare distributions and feed shared baseline tokens
    for step in range(STEPS):

        # Verify cache lengths remain valid at every step
        assert baseline_cache.get_seq_length() == num_tokens + 1 + step, (
            f"Baseline cache length mismatch at step {step}: "
            f"expected {num_tokens + 1 + step}, got {baseline_cache.get_seq_length()}"
        )
        assert dsaa_cache.get_seq_length() == len(retained_indices) + 1 + step, (
            f"DSAA cache length mismatch at step {step}: "
            f"expected {len(retained_indices) + 1 + step}, got {dsaa_cache.get_seq_length()}"
        )

        baseline_logits = baseline_step.logits[:, -1, :].float()
        dsaa_logits = dsaa_step.logits[:, -1, :].float()

        cosine = torch.nn.functional.cosine_similarity(
            baseline_logits,
            dsaa_logits,
            dim=-1
        ).item()

        baseline_top1 = torch.argmax(
            baseline_logits,
            dim=-1
        )

        dsaa_top1 = torch.argmax(
            dsaa_logits,
            dim=-1
        )

        baseline_top5 = torch.topk(
            baseline_logits,
            k=5,
            dim=-1
        ).indices[0].tolist()

        dsaa_top5 = torch.topk(
            dsaa_logits,
            k=5,
            dim=-1
        ).indices[0].tolist()

        overlap = len(set(baseline_top5) & set(dsaa_top5))

        cosine_scores.append(cosine)
        top1_matches.append(
            baseline_top1.item() == dsaa_top1.item()
        )
        top5_overlaps.append(overlap / 5)

        if step < STEPS - 1:
            # Use baseline's prediction as the shared next input
            shared_token = baseline_top1.reshape(1, 1)

            absolute_position = num_tokens + 1 + step

            position = torch.tensor(
                [[absolute_position]],
                dtype=torch.long,
                device=device
            )

            baseline_mask = torch.ones(
                (1, baseline_cache.get_seq_length() + 1),
                dtype=torch.long,
                device=device
            )

            dsaa_mask = torch.ones(
                (1, dsaa_cache.get_seq_length() + 1),
                dtype=torch.long,
                device=device
            )

            # Verify mask lengths match cache length + 1
            assert baseline_mask.shape[-1] == baseline_cache.get_seq_length() + 1
            assert dsaa_mask.shape[-1] == dsaa_cache.get_seq_length() + 1

            with torch.inference_mode():

                baseline_step = model(
                    input_ids=shared_token,
                    attention_mask=baseline_mask,
                    position_ids=position,
                    past_key_values=baseline_cache,
                    use_cache=True,
                    return_dict=True
                )

                dsaa_step = model(
                    input_ids=shared_token,
                    attention_mask=dsaa_mask,
                    position_ids=position,
                    past_key_values=dsaa_cache,
                    use_cache=True,
                    return_dict=True
                )

            baseline_cache = baseline_step.past_key_values
            dsaa_cache = dsaa_step.past_key_values

    return {
        "target": target_retention,
        "actual": actual_retention,
        "saved": memory_saved,
        "cosine": statistics.mean(cosine_scores),
        "top1": statistics.mean(top1_matches) * 100,
        "top5": statistics.mean(top5_overlaps) * 5
    }


def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)

    model.eval()

    results = {
        ratio: {
            "target": [],
            "actual": [],
            "saved": [],
            "cosine": [],
            "top1": [],
            "top5": []
        }
        for ratio in RETENTION_LEVELS
    }

    for name, prompt in PROMPTS.items():

        print(f"\n========== {name} ==========")

        for ratio in RETENTION_LEVELS:

            result = evaluate_retention(
                prompt,
                ratio,
                tokenizer,
                model,
                device
            )

            print(
                f"Retention {ratio * 100:.0f}% | "
                f"Actual {result['actual']:.2f}% | "
                f"Saved {result['saved']:.2f}% | "
                f"Cosine {result['cosine']:.5f} | "
                f"Top1 {result['top1']:.1f}% | "
                f"Top5 {result['top5']:.2f}/5"
            )

            for metric in results[ratio]:
                results[ratio][metric].append(result[metric])

    print("\n========== MULTI-STEP AGGREGATE RESULTS ==========")

    print(
        f"{'Target':<10}"
        f"{'Actual':<12}"
        f"{'Saved':<12}"
        f"{'Cosine':<12}"
        f"{'Top1':<12}"
        f"{'Top5':<12}"
    )

    for ratio in RETENTION_LEVELS:

        values = results[ratio]

        print(
            f"{ratio * 100:.0f}%"
            f"{'':<6}"
            f"{statistics.mean(values['actual']):.2f}%"
            f"{'':<5}"
            f"{statistics.mean(values['saved']):.2f}%"
            f"{'':<5}"
            f"{statistics.mean(values['cosine']):.5f}"
            f"{'':<5}"
            f"{statistics.mean(values['top1']):.1f}%"
            f"{'':<5}"
            f"{statistics.mean(values['top5']):.2f}/5"
        )

    print("\nMulti-step benchmark completed!")


if __name__ == "__main__":
    main()
