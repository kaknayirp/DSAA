
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

TEXT = """Rahul is an AI engineering student.
He is developing a smart waste management system.
He enjoys machine learning and deep learning."""

MAX_CAPACITY = 20

TARGET_RETENTION_LEVELS = [0.80, 0.60, 0.50, 0.30]


def cache_size_bytes(cache):
    total = 0

    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()

    return total


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

    inputs = tokenizer(TEXT, return_tensors="pt").to(device)
    num_tokens = inputs["input_ids"].shape[1]

    print("\nInput tokens:", num_tokens)

    # ------------------------------------------------
    # 1. ORIGINAL MODEL PREFILL
    # ------------------------------------------------

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )

    original_cache = outputs.past_key_values
    original_bytes = cache_size_bytes(original_cache)

    print("Original cache bytes:", original_bytes)

    # ------------------------------------------------
    # 2. CALCULATE DSAA SCORES
    # ------------------------------------------------

    attention_scores = calculate_attention_importance(
        outputs.attentions,
        query_window=4
    )

    semantic_scores = calculate_semantic_importance(
        TEXT,
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

    # ------------------------------------------------
    # 3. BASELINE NEXT-TOKEN PREDICTION
    # ------------------------------------------------

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

    # ------------------------------------------------
    # 4. BUILD PRUNED CACHES BEFORE BASELINE DECODE
    # ------------------------------------------------

    experiments = []

    for target_ratio in TARGET_RETENTION_LEVELS:

        target_count = round(num_tokens * target_ratio)
        target_count = max(4, min(num_tokens, target_count))

        eviction_count = num_tokens - target_count

        # At pressure 1.0, the policy's eviction fraction
        # is the configured maximum eviction fraction.
        max_eviction_fraction = (
            (eviction_count + 0.001) / num_tokens
        )

        retained_indices = select_tokens_to_retain(
            combined_scores,
            pressure,
            pressure_threshold=0.8,
            max_eviction_fraction=max_eviction_fraction,
            protected_recent_tokens=4
        )

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

        pruned_cache = DynamicCache(
            ddp_cache_data=cache_data,
            config=model.config
        )

        experiments.append({
            "target": target_ratio,
            "retained_indices": retained_indices,
            "cache": pruned_cache,
            "bytes": cache_size_bytes(pruned_cache)
        })

    # ------------------------------------------------
    # 5. BASELINE CONTINUATION
    # ------------------------------------------------

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
    )

    baseline_top5 = torch.topk(
        baseline_logits,
        k=5,
        dim=-1
    ).indices

    # ------------------------------------------------
    # 6. TEST EACH RETENTION LEVEL
    # ------------------------------------------------

    print("\n========== RETENTION EXPERIMENT RESULTS ==========")

    print(
        f"{'Target':<10}"
        f"{'Actual':<10}"
        f"{'Saved':<10}"
        f"{'Cosine':<12}"
        f"{'Top1':<10}"
        f"{'Top5':<10}"
    )

    for experiment in experiments:

        pruned_cache = experiment["cache"]
        retained_indices = experiment["retained_indices"]

        pruned_mask = torch.ones(
            (1, len(retained_indices) + 1),
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
        )

        pruned_top5 = torch.topk(
            pruned_logits,
            k=5,
            dim=-1
        ).indices

        top1_match = (
            baseline_top1.item() == pruned_top1.item()
        )

        top5_overlap = len(
            set(baseline_top5[0].tolist())
            & set(pruned_top5[0].tolist())
        )

        actual_ratio = len(retained_indices) / num_tokens

        memory_saved = (
            1 - experiment["bytes"] / original_bytes
        ) * 100

        print(
            f"{experiment['target'] * 100:.0f}%"
            f"{'':<5}"
            f"{actual_ratio * 100:.2f}%"
            f"{'':<3}"
            f"{memory_saved:.2f}%"
            f"{'':<4}"
            f"{cosine:.5f}"
            f"{'':<5}"
            f"{str(top1_match):<10}"
            f"{top5_overlap}/5"
        )

    print("\nBaseline next token:", tokenizer.decode(next_token[0]))
    print("\nRetention benchmark completed!")


if __name__ == "__main__":
    main()
