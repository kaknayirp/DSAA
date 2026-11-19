
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
).to(device)

model.eval()

text = """Rahul is an AI engineering student.
He is developing a smart waste management system.
He enjoys machine learning and deep learning."""

inputs = tokenizer(text, return_tensors="pt").to(device)
num_tokens = inputs["input_ids"].shape[1]

print("\nInput tokens:", num_tokens)

with torch.inference_mode():

    # Step 1: Prefill the model and create the original cache
    outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True
    )

    original_cache = outputs.past_key_values

    print("\nOriginal cache type:", type(original_cache))
    print("Original cache length:", original_cache.get_seq_length())

    # Step 2: Predict the next token from the prefill logits
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    print("\nFirst predicted token:",
          tokenizer.decode(next_token[0]))

    # Step 3: Feed that token back using the original cache
    attention_mask = torch.ones(
        (1, num_tokens + 1),
        dtype=torch.long,
        device=device
    )

    position_ids = torch.tensor(
        [[num_tokens]],
        dtype=torch.long,
        device=device
    )

    continuation = model(
        input_ids=next_token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=original_cache,
        use_cache=True,
        return_dict=True
    )

    print("\nContinuation successful!")

    print("Updated cache length:",
          continuation.past_key_values.get_seq_length())

    following_token = torch.argmax(
        continuation.logits[:, -1, :],
        dim=-1
    )

    print("Following predicted token:",
          tokenizer.decode(following_token))

print("\nBASELINE CACHE CONTINUATION TEST PASSED!")
