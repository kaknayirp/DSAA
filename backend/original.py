from threading import Event, Thread
from time import perf_counter

import psutil

from .studio_schemas import CacheEntry, CacheMetrics, InferenceResult


def load_original(path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).to(device).eval()
    if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
        raise ValueError("The Original adapter requires a Qwen2/Qwen2.5 model with full attention.")
    process = psutil.Process()

    def run(request, _settings):
        messages = [{"role": "user", "content": request.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
        context_tokens = inputs.input_ids.shape[-1]
        context_limit = min(model.config.max_position_embeddings, 8192)
        if context_tokens + request.max_new_tokens > context_limit:
            raise ValueError(f"Prompt plus generation exceeds the {context_limit}-token limit.")

        # generate() streams the input first, then each generated token.
        class FirstTokenClock:
            first_token = None
            received_input = False

            def put(self, _tokens):
                if self.received_input and self.first_token is None:
                    self.first_token = perf_counter()
                self.received_input = True

            def end(self):
                pass

        stop = Event()
        peak = process.memory_info().rss

        def sample_ram():
            nonlocal peak
            while not stop.wait(0.02):
                peak = max(peak, process.memory_info().rss)

        sampler = Thread(target=sample_ram, daemon=True)
        sampler.start()
        clock = FirstTokenClock()
        memory_pressure = psutil.virtual_memory().percent / 100
        try:
            start = perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=request.max_new_tokens, do_sample=False,
                    use_cache=True, return_dict_in_generate=True, streamer=clock,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            elapsed = perf_counter() - start
            peak = max(peak, process.memory_info().rss)
        finally:
            stop.set()
            sampler.join()

        cache = output.past_key_values
        if hasattr(cache, "layers"):
            pairs = [(layer.keys, layer.values) for layer in cache.layers]
        elif hasattr(cache, "key_cache"):
            pairs = list(zip(cache.key_cache, cache.value_cache))
        else:
            pairs = cache
        cached_tokens = pairs[0][0].shape[-2]
        kv_bytes = sum(tensor.numel() * tensor.element_size() for pair in pairs for tensor in pair)
        sequence = output.sequences[0].tolist()
        completion = sequence[context_tokens:]
        cached_ids = sequence[:cached_tokens]
        return InferenceResult(
            context_tokens=context_tokens,
            memory_pressure=memory_pressure,
            metrics=CacheMetrics(
                kv_cache_bytes=kv_bytes, cached_tokens=cached_tokens, peak_ram_bytes=peak,
                first_token_ms=(clock.first_token - start) * 1000 if clock.first_token else None,
                tokens_per_second=len(completion) / elapsed,
                answer=tokenizer.decode(completion, skip_special_tokens=True),
            ),
            entries=[
                CacheEntry(position=i, token=token, token_id=token_id, retained=True,
                           reason="Original cache policy keeps all processed positions.", saved_bytes=0)
                for i, (token_id, token) in enumerate(zip(cached_ids, tokenizer.convert_ids_to_tokens(cached_ids)))
            ],
        )

    return run
