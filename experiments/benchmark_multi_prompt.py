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

PROMPTS = {
    "Technical": """
    Explain how a computer network uses the TCP protocol to establish
    a reliable connection between two devices. Discuss the handshake,
    acknowledgements, retransmission, and error detection.
    """,

    "Programming": """
    Write a Python function to find the second largest element
    in an array. Explain how the algorithm works, discuss its
    time complexity, and handle duplicate values and edge cases.
    """,

    "General Knowledge": """
    Explain the importance of renewable energy sources such as
    solar and wind power. Discuss their environmental benefits,
    limitations, and role in reducing dependence on fossil fuels.
    """,

    "Reasoning": """
    A student studies mathematics, programming, and artificial
    intelligence every day. Mathematics takes two hours, programming
    takes one hour, and artificial intelligence takes two hours.
    If the student has only four hours available, suggest a practical
    study schedule and explain the trade-offs.
    """,

    "AI/ML": """
    Explain the difference between supervised learning and
    unsupervised learning. Describe classification, regression,
    clustering, and give practical examples of applications
    where each technique is useful.
    """
}

MAX_CAPACITY = 20

TARGET_RETENTION_LEVELS = [0.80, 0.60, 0.50, 0.30]


def cache_size_bytes(cache):
    total = 0

    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()

    return total


def build_pruned_cache(
    original_cache,
    model,
    retained_indices,
    device
):
    index_tensor = torch.tensor(
        retained_indices,
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


def evaluate_prompt(text, tokenizer, model, device):

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

    # Calculate DSAA scores
    attention_scores = calculate_attention_importance(
        outputs.attentions,
        query_window=4
    )

    semantic_scores = calculate_semantic_importance(
        text,
        tokenizer
    )

    recency_scores = calculate_recency_importance(num_tokens)

    combined_scores = calculate_combined_score(
        attention_scores,
        semantic_scores,
        recency_scores
    )

    pressure = calculate_memory_pressure(
        num_tokens,
        MAX_CAPACITY
    )

    # Use the original model's next token for both paths
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    position = torch.tensor(
        [[num_tokens]],
        dtype=torch.long,
        device=device
    )

    baseline_mask = torch.ones(
        (1, num_tokens + 1),
        dtype=torch.long,
        device=device
    )

    # Build all pruned caches before running continuations
    experiments = []

    for target_ratio in TARGET_RETENTION_LEVELS:

        target_count = round(num_tokens * target_ratio)
        target_count = max(4, min(num_tokens, target_count))

        eviction_count = num_tokens - target_count

        normalized_pressure = (
            (pressure - 0.8) / 0.2
        )

        if normalized_pressure <= 0:
            max_eviction_fraction = 0.0
        else:
            max_eviction_fraction = min(
                1.0,
                (eviction_count + 0.001)
                / (num_tokens * normalized_pressure)
            )

        retained_indices = select_tokens_to_retain(
            combined_scores,
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

        experiments.append({
            "target": target_ratio,
            "indices": retained_indices,
            "cache": pruned_cache
        })

    # Baseline continuation
    with torch.inference_mode():
        baseline_output = model(
            input_ids=next_token,
            attention_mask=baseline_mask,
            position_ids=position,
            past_key_values=original_cache,
            use_cache=True,
            return_dict=True
        )

    baseline_logits = baseline_output.logits[:, -1, :].float()

    baseline_top1 = torch.argmax(
        baseline_logits,
        dim=-1
    ).item()

    baseline_top5 = torch.topk(
        baseline_logits,
        k=5,
        dim=-1
    ).indices[0].tolist()

    results = []

    # Test each pruned cache
    for experiment in experiments:

        indices = experiment["indices"]
        pruned_cache = experiment["cache"]

        pruned_mask = torch.ones(
            (1, len(indices) + 1),
            dtype=torch.long,
            device=device
        )

        with torch.inference_mode():
            pruned_output = model(
                input_ids=next_token,
                attention_mask=pruned_mask,
                position_ids=position,
                past_key_values=pruned_cache,
                use_cache=True,
                return_dict=True
            )

        pruned_logits = pruned_output.logits[:, -1, :].float()

        cosine = torch.nn.functional.cosine_similarity(
            baseline_logits,
            pruned_logits,
            dim=-1
        ).item()

        pruned_top1 = torch.argmax(
            pruned_logits,
            dim=-1
        ).item()

        pruned_top5 = torch.topk(
            pruned_logits,
            k=5,
            dim=-1
        ).indices[0].tolist()

        top1_match = baseline_top1 == pruned_top1

        top5_overlap = len(
            set(baseline_top5) & set(pruned_top5)
        )

        actual_ratio = len(indices) / num_tokens

        pruned_bytes = cache_size_bytes(pruned_cache)

        memory_saved = (
            1 - pruned_bytes / original_bytes
        ) * 100

        results.append({
            "target": experiment["target"],
            "actual": actual_ratio * 100,
            "saved": memory_saved,
            "cosine": cosine,
            "top1": int(top1_match),
            "top5": top5_overlap / 5
        })

    return num_tokens, results


def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)

    model.eval()

    all_results = {
        ratio: {
            "actual": [],
            "saved": [],
            "cosine": [],
            "top1": [],
            "top5": []
        }
        for ratio in TARGET_RETENTION_LEVELS
    }

    print("\n========== MULTI-PROMPT BENCHMARK ==========")

    for prompt_name, prompt_text in PROMPTS.items():

        print(f"\nTesting: {prompt_name}")

        num_tokens, results = evaluate_prompt(
            prompt_text,
            tokenizer,
            model,
            device
        )

        print("Input tokens:", num_tokens)

        for result in results:

            ratio = result["target"]

            print(
                f"Retention {ratio * 100:.0f}% | "
                f"Actual {result['actual']:.2f}% | "
                f"Saved {result['saved']:.2f}% | "
                f"Cosine {result['cosine']:.5f} | "
                f"Top1 {bool(result['top1'])} | "
                f"Top5 {result['top5'] * 5:.0f}/5"
            )

            for metric in all_results[ratio]:
                all_results[ratio][metric].append(
                    result[metric]
                )

    # Aggregate results across prompts
    print("\n========== AGGREGATE RESULTS ==========")

    print(
        f"{'Target':<10}"
        f"{'Actual':<12}"
        f"{'Saved':<12}"
        f"{'Cosine':<12}"
        f"{'Top1':<12}"
        f"{'Top5':<12}"
    )

    for ratio in TARGET_RETENTION_LEVELS:

        values = all_results[ratio]

        avg_actual = statistics.mean(values["actual"])
        avg_saved = statistics.mean(values["saved"])
        avg_cosine = statistics.mean(values["cosine"])
        avg_top1 = statistics.mean(values["top1"]) * 100
        avg_top5 = statistics.mean(values["top5"]) * 5

        print(
            f"{ratio * 100:.0f}%"
            f"{'':<6}"
            f"{avg_actual:.2f}%"
            f"{'':<5}"
            f"{avg_saved:.2f}%"
            f"{'':<5}"
            f"{avg_cosine:.5f}"
            f"{'':<5}"
            f"{avg_top1:.1f}%"
            f"{'':<5}"
            f"{avg_top5:.2f}/5"
        )

    print("\nMulti-prompt benchmark completed!")


if __name__ == "__main__":
    main()
