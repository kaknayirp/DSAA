
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


def get_cache_size_bytes(cache):
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

    # --------------------------------------------------
    # 1. PREFILL: Generate original attention and KV cache
    # --------------------------------------------------

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )

    original_cache = outputs.past_key_values
    original_bytes = get_cache_size_bytes(original_cache)

    print("Original cache length:", original_cache.get_seq_length())
    print("Original cache size:", original_bytes, "bytes")

    # --------------------------------------------------
    # 2. DSAA IMPORTANCE SCORING
    # --------------------------------------------------

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

    print("\nScore lengths:")
    print("Attention:", len(attention_scores))
    print("Semantic:", len(semantic_scores))
    print("Recency:", len(recency_scores))
    print("Combined:", len(combined_scores))

    if len(combined_scores) != num_tokens:
        raise ValueError("Score length does not match input token count")

    # --------------------------------------------------
    # 3. DSAA RETENTION POLICY
    # --------------------------------------------------

    pressure = calculate_memory_pressure(
        num_tokens,
        MAX_CAPACITY
    )

    retained_indices = select_tokens_to_retain(
        combined_scores,
        pressure,
        pressure_threshold=0.8,
        max_eviction_fraction=0.5,
        protected_recent_tokens=4
    )

    print("\nMemory pressure:", pressure)
    print("Retained tokens:", len(retained_indices))
    print("Evicted tokens:", num_tokens - len(retained_indices))
    print("Retained indices:", retained_indices)

    # --------------------------------------------------
    # 4. COMPACT EACH REAL KV CACHE LAYER
    # --------------------------------------------------

    index_tensor = torch.tensor(
        retained_indices,
        dtype=torch.long,
        device=device
    )

    compacted_cache_data = []

    for layer in original_cache.layers:
        compacted_keys = layer.keys.index_select(
            -2, index_tensor
        )

        compacted_values = layer.values.index_select(
            -2, index_tensor
        )

        compacted_cache_data.append(
            (compacted_keys, compacted_values)
        )

    # Construct a new DynamicCache using the actual
    # Transformers 5.22.2 API discovered in our inspection.

    pruned_cache = DynamicCache(
        ddp_cache_data=compacted_cache_data,
        config=model.config
    )

    pruned_bytes = get_cache_size_bytes(pruned_cache)

    print("\n--- RECONSTRUCTED PRUNED CACHE ---")
    print("Cache type:", type(pruned_cache))
    print("Cache layers:", len(pruned_cache.layers))
    print("Cache length:", pruned_cache.get_seq_length())
    print("Cache size:", pruned_bytes, "bytes")

    print(
        "Memory reduction:",
        round((1 - pruned_bytes / original_bytes) * 100, 2),
        "%"
    )

    # --------------------------------------------------
    # 5. PREDICT THE FIRST NEXT TOKEN
    # --------------------------------------------------

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    print("\nToken to feed:", tokenizer.decode(next_token[0]))

    # --------------------------------------------------
    # 6. BASELINE CONTINUATION
    # --------------------------------------------------

    original_mask = torch.ones(
        (1, num_tokens + 1),
        dtype=torch.long,
        device=device
    )

    next_position = torch.tensor(
        [[num_tokens]],
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():
        baseline_output = model(
            input_ids=next_token,
            attention_mask=original_mask,
            position_ids=next_position,
            past_key_values=original_cache,
            use_cache=True,
            return_dict=True
        )

    print("\n--- BASELINE CONTINUATION ---")
    print("Successful: Yes")
    print(
        "Updated cache length:",
        baseline_output.past_key_values.get_seq_length()
    )

    # --------------------------------------------------
    # 7. DSAA PRUNED CACHE CONTINUATION
    # --------------------------------------------------

    pruned_mask = torch.ones(
        (1, len(retained_indices) + 1),
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():
        pruned_output = model(
            input_ids=next_token,
            attention_mask=pruned_mask,
            position_ids=next_position,
            past_key_values=pruned_cache,
            use_cache=True,
            return_dict=True
        )

    print("\n--- DSAA PRUNED CONTINUATION ---")
    print("Successful: Yes")
    print(
        "Updated cache length:",
        pruned_output.past_key_values.get_seq_length()
    )

    following_token = torch.argmax(
        pruned_output.logits[:, -1, :],
        dim=-1
    )

    print(
        "Following predicted token:",
        tokenizer.decode(following_token[0])
    )

    print("\nDSAA PRUNED CACHE CONTINUATION TEST PASSED!")

    # --------------------------------------------------
    # 8. CONTROLLED MULTI-STEP LOGIT COMPARISON
    # --------------------------------------------------

    print("\n--- CONTROLLED MULTI-STEP COMPARISON ---")

    baseline_cache = baseline_output.past_key_values
    dsaa_cache = pruned_output.past_key_values

    baseline_step = baseline_output
    dsaa_step = pruned_output

    baseline_predictions = []
    dsaa_predictions = []
    shared_tokens = []

    cosine_scores = []
    top1_matches = []
    top5_overlaps = []

    for step in range(5):

        # Compare distributions under the same input history
        baseline_logits = baseline_step.logits[:, -1, :].float()
        dsaa_logits = dsaa_step.logits[:, -1, :].float()

        cosine = torch.nn.functional.cosine_similarity(
            baseline_logits,
            dsaa_logits,
            dim=-1
        ).item()

        baseline_top1 = torch.argmax(
            baseline_logits, dim=-1
        )

        dsaa_top1 = torch.argmax(
            dsaa_logits, dim=-1
        )

        baseline_top5 = torch.topk(
            baseline_logits, k=5, dim=-1
        ).indices

        dsaa_top5 = torch.topk(
            dsaa_logits, k=5, dim=-1
        ).indices

        overlap = len(
            set(baseline_top5[0].tolist())
            & set(dsaa_top5[0].tolist())
        )

        cosine_scores.append(cosine)
        top1_matches.append(
            baseline_top1.item() == dsaa_top1.item()
        )
        top5_overlaps.append(overlap / 5)

        baseline_predictions.append(baseline_top1.item())
        dsaa_predictions.append(dsaa_top1.item())

        print(f"\nComparison step {step + 1}")
        print("Cosine similarity:", round(cosine, 5))
        print("Top-1 match:", top1_matches[-1])
        print("Top-5 overlap:", overlap, "/ 5")

        # Use the baseline's greedy prediction as the
        # shared next input for BOTH caches.
        shared_token = baseline_top1.reshape(1, 1)
        shared_tokens.append(shared_token.item())

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

    print("\n--- FINAL COMPARISON SUMMARY ---")

    print(
        "Average cosine similarity:",
        round(sum(cosine_scores) / len(cosine_scores), 5)
    )

    print(
        "Top-1 agreement:",
        sum(top1_matches), "/", len(top1_matches)
    )

    print(
        "Average top-5 overlap:",
        round(sum(top5_overlaps) / len(top5_overlaps), 3)
    )

    print("\nBaseline predictions:")
    print(tokenizer.decode(baseline_predictions))

    print("\nDSAA predictions:")
    print(tokenizer.decode(dsaa_predictions))

    print("\nShared continuation:")
    print(tokenizer.decode(shared_tokens))

    print("\nControlled comparison completed.")
if __name__ == "__main__":
    main()
