"""
Inference Engine with complete KV Cache Integration for DSAA.

Module Owner: Person 1 — LLM & KV Cache Engineer.

This module provides the core autoregressive token generation engine,
seamlessly orchestrating:
1. Prompt prefill stage (processes full sequence once and initializes the KV Cache).
2. Incremental autoregressive decoding stage (processes single tokens using existing KV Cache).
3. KV Cache lifecycle management (independent state per request, automatic cleanup).
4. Direct integration with Prompt 2's KV Cache inspection utilities (extract_kv_cache, CacheReport).
5. Robust sampling (Greedy, Temperature, Top-K, Top-P, Repetition penalty, Stop tokens).
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .exceptions import (
    InferenceEngineError,
    InvalidInputError,
    CacheStateError,
    LLMEngineError,
)
from .config import ModelConfig
from .generation_config import GenerationConfig
from .model_loader import ModelLoader, ModelArtifacts
from .kv_extractor import extract_kv_cache, CacheReport

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Represents an isolated inference request submitted to the engine."""
    input_ids: Any
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    attention_mask: Optional[Any] = None
    prompt_text: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.input_ids is None:
            raise InvalidInputError("InferenceRequest 'input_ids' cannot be None.")


@dataclass
class GenerationState:
    """
    Maintains the isolated execution and KV Cache state for an individual generation request.
    Prevents cross-request cache contamination and ensures clean resource reclamation.
    """
    request: InferenceRequest
    past_key_values: Any = None
    generated_token_ids: List[int] = field(default_factory=list)
    prompt_length: int = 0
    current_seq_len: int = 0
    prefill_completed: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None
    prefill_latency_sec: float = 0.0
    decode_latencies_sec: List[float] = field(default_factory=list)

    def cleanup(self) -> None:
        """Release KV Cache tensors and reset references to prevent memory leaks."""
        if self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None


@dataclass
class InferenceResult:
    """Structured result returned upon completion of token generation."""
    request_id: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    output_token_ids: List[int]
    decoded_text: Optional[str]
    finish_reason: str
    num_prompt_tokens: int
    num_generated_tokens: int
    total_time_seconds: float
    prefill_time_seconds: float
    decode_time_seconds: float
    tokens_per_second: float
    final_cache_report: Optional[CacheReport] = None

    def summary(self) -> str:
        """Return formatted summary of the generation run."""
        lines = [
            "--------------------------------------------------------------------------------",
            f" Inference Request ID: {self.request_id}",
            f" Finish Reason:        {self.finish_reason}",
            f" Prompt Tokens:        {self.num_prompt_tokens}",
            f" Generated Tokens:     {self.num_generated_tokens}",
            f" Total Tokens:         {len(self.output_token_ids)}",
            f" Total Time:           {self.total_time_seconds:.4f} s",
            f" Prefill Latency:      {self.prefill_time_seconds:.4f} s",
            f" Decode Latency:       {self.decode_time_seconds:.4f} s",
            f" Generation Speed:     {self.tokens_per_second:.2f} tokens/s",
            "--------------------------------------------------------------------------------",
        ]
        if self.decoded_text:
            lines.append(f" Decoded Output:\n'{self.decoded_text}'")
            lines.append("--------------------------------------------------------------------------------")
        if self.final_cache_report:
            lines.append(f" Final KV Cache Memory: {self.final_cache_report.total_mb:.4f} MB ({self.final_cache_report.total_bytes:,} bytes)")
            lines.append(f" KV Cache Sequence Len: {self.final_cache_report.seq_len} tokens")
            lines.append("--------------------------------------------------------------------------------")
        return "\n".join(lines)


class InferenceEngine:
    """
    Core autoregressive inference engine integrated with Hugging Face Causal LMs
    and DSAA KV Cache extraction utilities.
    """

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        artifacts: Optional[ModelArtifacts] = None,
        config: Optional[ModelConfig] = None,
    ) -> None:
        """
        Initialize the inference engine by reusing Prompt 1's ModelLoader or artifacts.
        """
        if artifacts is not None:
            self._artifacts = artifacts
            self._loader = None
        else:
            self._loader = model_loader or ModelLoader(config or ModelConfig())
            self._artifacts = None

    @property
    def is_model_loaded(self) -> bool:
        """Return True if model and tokenizer artifacts are loaded and ready."""
        if self._artifacts is not None:
            return True
        return self._loader.is_loaded if self._loader else False

    def get_artifacts(self) -> ModelArtifacts:
        """Retrieve active ModelArtifacts, triggering load if necessary."""
        if self._artifacts is None:
            if self._loader is None:
                raise InferenceEngineError("No ModelLoader or ModelArtifacts provided to InferenceEngine.")
            self._artifacts = self._loader.load()
        return self._artifacts

    @property
    def model(self) -> Any:
        return self.get_artifacts().model

    @property
    def tokenizer(self) -> Any:
        return self.get_artifacts().tokenizer

    @property
    def device(self) -> str:
        return self.get_artifacts().device

    def prefill(self, request: InferenceRequest) -> GenerationState:
        """
        Execute the prompt prefill stage:
        1. Encodes or prepares the full prompt token sequence.
        2. Executes a single forward pass with use_cache=True.
        3. Initializes past_key_values in the GenerationState.
        4. Selects the first generated token.
        """
        input_ids = request.input_ids
        attention_mask = request.attention_mask

        # Validate input sequence
        prompt_len, tensor_input_ids, tensor_attention_mask = self._normalize_inputs(input_ids, attention_mask)
        if prompt_len <= 0:
            raise InvalidInputError("Prompt sequence is empty (0 tokens). Cannot execute prefill.")

        state = GenerationState(
            request=request,
            prompt_length=prompt_len,
            current_seq_len=prompt_len,
        )

        t0 = time.time()
        try:
            # Model forward pass for the complete prompt sequence
            outputs = self._call_model(
                input_ids=tensor_input_ids,
                attention_mask=tensor_attention_mask,
                past_key_values=None,
                use_cache=True,
            )
        except Exception as exc:
            state.cleanup()
            raise InferenceEngineError(f"Model forward pass failed during prefill: {exc}") from exc

        state.prefill_latency_sec = time.time() - t0
        state.prefill_completed = True

        # Extract and verify the KV Cache
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            state.cleanup()
            raise CacheStateError(
                "Model did not return past_key_values during prefill. "
                "Ensure model.config.use_cache=True is supported."
            )
        state.past_key_values = past_key_values

        # Extract logits for the final prompt position
        logits = self._extract_last_token_logits(outputs)

        # Select first generated token
        first_token = self._sample_next_token(logits, request.generation_config, state.generated_token_ids)
        state.generated_token_ids.append(first_token)
        state.current_seq_len += 1

        # Check stopping condition on first token
        if first_token in request.generation_config.stop_token_set:
            state.is_finished = True
            state.finish_reason = "eos"
        elif self._ends_with_stop_sequence(state.generated_token_ids, request.generation_config):
            state.is_finished = True
            state.finish_reason = "stop_sequence"
        elif len(state.generated_token_ids) >= request.generation_config.max_new_tokens:
            state.is_finished = True
            state.finish_reason = "max_tokens"

        return state

    def decode_step(self, state: GenerationState) -> Tuple[int, bool]:
        """
        Execute a single autoregressive decoding step:
        1. Passes only the last generated token to the model.
        2. Supplies the existing past_key_values so previous tokens are not recomputed.
        3. Updates state.past_key_values incrementally with new key/value representations.
        4. Selects the next token and checks stopping criteria.
        """
        if not state.prefill_completed:
            raise CacheStateError("Cannot execute decode_step before prefill has completed.")
        if state.is_finished:
            return state.generated_token_ids[-1], True

        config = state.request.generation_config

        # Check cache capacity limit if configured
        if config.max_cache_capacity is not None and state.current_seq_len >= config.max_cache_capacity:
            state.is_finished = True
            state.finish_reason = "capacity_exceeded"
            return state.generated_token_ids[-1], True

        last_token = state.generated_token_ids[-1]
        next_input_tensor = self._prepare_single_token_tensor(last_token)

        # Build position IDs and updated attention mask if appropriate
        position_ids = self._prepare_position_ids(state.current_seq_len - 1)

        t0 = time.time()
        try:
            outputs = self._call_model(
                input_ids=next_input_tensor,
                past_key_values=state.past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
        except Exception as exc:
            state.cleanup()
            raise InferenceEngineError(f"Model forward pass failed during decode step: {exc}") from exc

        step_latency = time.time() - t0
        state.decode_latencies_sec.append(step_latency)

        # Update cache reference (incrementally updated by the model)
        updated_cache = getattr(outputs, "past_key_values", None)
        if updated_cache is None:
            state.cleanup()
            raise CacheStateError("Model did not return updated past_key_values during decoding.")
        state.past_key_values = updated_cache

        # Extract logits and select next token
        logits = self._extract_last_token_logits(outputs)
        next_token = self._sample_next_token(logits, config, state.generated_token_ids)

        state.generated_token_ids.append(next_token)
        state.current_seq_len += 1

        # Check stopping criteria
        if next_token in config.stop_token_set:
            state.is_finished = True
            state.finish_reason = "eos"
        elif self._ends_with_stop_sequence(state.generated_token_ids, config):
            state.is_finished = True
            state.finish_reason = "stop_sequence"
        elif len(state.generated_token_ids) >= config.max_new_tokens:
            state.is_finished = True
            state.finish_reason = "max_tokens"

        return next_token, state.is_finished

    def generate(
        self,
        prompt: Union[str, List[int], Any],
        generation_config: Optional[GenerationConfig] = None,
        inspect_cache: bool = True,
    ) -> InferenceResult:
        """
        Execute full autoregressive generation from prompt to completion.

        Args:
            prompt: Text string or list of input token IDs.
            generation_config: Optional decoding configuration (defaults applied if None).
            inspect_cache: If True, uses Prompt 2's extract_kv_cache to include a CacheReport.

        Returns:
            InferenceResult containing generated token IDs, decoded text, latency, and cache statistics.
        """
        cfg = generation_config or GenerationConfig()

        # If eos_token_id is not specified in config, resolve from tokenizer if possible
        if not cfg.stop_token_set and self.tokenizer is not None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                cfg.stop_token_set.add(int(eos_id))

        # Tokenize if prompt is a string
        prompt_text = None
        if isinstance(prompt, str):
            prompt_text = prompt
            if not prompt.strip():
                raise InvalidInputError("Prompt text cannot be empty or purely whitespace.")
            if self.tokenizer is None:
                raise InferenceEngineError("Tokenizer is required to encode string prompt.")
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = prompt
            attention_mask = None

        request = InferenceRequest(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=cfg,
            prompt_text=prompt_text,
        )

        state: Optional[GenerationState] = None
        t_start = time.time()
        try:
            # 1. Prefill stage
            state = self.prefill(request)

            # 2. Decode stage
            while not state.is_finished:
                self.decode_step(state)

            total_time = time.time() - t_start
            decode_time = sum(state.decode_latencies_sec)
            num_generated = len(state.generated_token_ids)
            tps = (num_generated / total_time) if total_time > 0 else 0.0

            # 3. Optional KV Cache inspection (Prompt 2 integration)
            cache_report = None
            if inspect_cache and state.past_key_values is not None:
                try:
                    cache_report = extract_kv_cache(state.past_key_values)
                except Exception as exc:
                    logger.warning("Failed to extract KV Cache report: %s", exc)

            # 4. Decode generated tokens to string
            prompt_token_ids = self._to_token_list(request.input_ids)
            output_token_ids = prompt_token_ids + state.generated_token_ids
            decoded_text = None
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                try:
                    decoded_text = self.tokenizer.decode(state.generated_token_ids, skip_special_tokens=True)
                except Exception:
                    pass

            return InferenceResult(
                request_id=request.request_id,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=list(state.generated_token_ids),
                output_token_ids=output_token_ids,
                decoded_text=decoded_text,
                finish_reason=state.finish_reason or "completed",
                num_prompt_tokens=len(prompt_token_ids),
                num_generated_tokens=num_generated,
                total_time_seconds=total_time,
                prefill_time_seconds=state.prefill_latency_sec,
                decode_time_seconds=decode_time,
                tokens_per_second=tps,
                final_cache_report=cache_report,
            )

        finally:
            # Ensure KV Cache references are freed upon request exit
            if state is not None:
                state.cleanup()

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _call_model(self, **kwargs) -> Any:
        """Invoke the loaded model with torch.no_grad() if torch is available."""
        try:
            import torch
            with torch.no_grad():
                return self.model(**kwargs)
        except ImportError:
            return self.model(**kwargs)

    def _normalize_inputs(self, input_ids: Any, attention_mask: Optional[Any]) -> Tuple[int, Any, Optional[Any]]:
        """Normalize input_ids into standard 2D batch tensor and compute prompt token length."""
        try:
            import torch
            if isinstance(input_ids, list):
                if len(input_ids) == 0:
                    return 0, input_ids, attention_mask
                if isinstance(input_ids[0], int):
                    input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                else:
                    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            elif hasattr(input_ids, "dim") and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            # Hugging Face tokenizers usually return CPU tensors even when the
            # model is placed on CUDA. Keep IDs and masks on the model device.
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                if isinstance(attention_mask, list):
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
                elif hasattr(attention_mask, "to"):
                    attention_mask = attention_mask.to(self.device)

            if hasattr(input_ids, "shape"):
                seq_len = input_ids.shape[-1]
            else:
                seq_len = len(input_ids)
            return seq_len, input_ids, attention_mask
        except ImportError:
            # When PyTorch is mocked or not installed
            if isinstance(input_ids, list):
                if len(input_ids) == 0:
                    return 0, input_ids, attention_mask
                if isinstance(input_ids[0], int):
                    return len(input_ids), [input_ids], attention_mask
                return len(input_ids[0]), input_ids, attention_mask
            if hasattr(input_ids, "shape"):
                return input_ids.shape[-1], input_ids, attention_mask
            return len(input_ids), input_ids, attention_mask

    def _prepare_single_token_tensor(self, token_id: int) -> Any:
        """Create a 2D tensor [1, 1] containing the next token ID."""
        try:
            import torch
            return torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        except ImportError:
            return [[token_id]]

    def _prepare_position_ids(self, position: int) -> Any:
        """Create a 2D tensor [1, 1] for positional embedding."""
        try:
            import torch
            return torch.tensor([[position]], dtype=torch.long, device=self.device)
        except ImportError:
            return [[position]]

    @staticmethod
    def _ends_with_stop_sequence(token_ids: List[int], config: GenerationConfig) -> bool:
        """Return True when generated IDs end in a configured token sequence."""
        return any(
            len(token_ids) >= len(sequence) and token_ids[-len(sequence):] == sequence
            for sequence in config.stop_token_sequences
        )

    def _to_token_list(self, input_ids: Any) -> List[int]:
        """Convert input_ids representation into a flat list of integers."""
        if hasattr(input_ids, "tolist"):
            res = input_ids.tolist()
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                return res[0]
            return res
        if isinstance(input_ids, list):
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                return list(input_ids[0])
            return list(input_ids)
        return []

    def _extract_last_token_logits(self, outputs: Any) -> Any:
        """Extract next-token logits from model outputs [batch_size, seq_len, vocab_size]."""
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise InferenceEngineError("Model outputs do not contain 'logits'.")
        if hasattr(logits, "shape") and len(logits.shape) == 3:
            return logits[:, -1, :]
        if isinstance(logits, list) and len(logits) > 0 and isinstance(logits[0], list):
            return logits[0][-1] if isinstance(logits[0][-1], list) else logits[0]
        return logits

    def _sample_next_token(
        self,
        logits: Any,
        config: GenerationConfig,
        generated_tokens: List[int],
    ) -> int:
        """
        Sample the next token ID from logits based on generation configuration:
        - Greedy (argmax) if do_sample is False or temperature == 0.0
        - Sampling with Temperature, Top-K, Top-P, and Repetition penalty if do_sample is True
        """
        try:
            import torch
            if isinstance(logits, torch.Tensor):
                logits_vec = logits[0] if logits.dim() == 2 else logits
                logits_vec = logits_vec.clone()

                # 1. Repetition Penalty
                if config.repetition_penalty != 1.0 and generated_tokens:
                    for token in set(generated_tokens):
                        if token < logits_vec.shape[-1]:
                            if logits_vec[token] > 0:
                                logits_vec[token] /= config.repetition_penalty
                            else:
                                logits_vec[token] *= config.repetition_penalty

                # 2. Greedy selection
                if not config.do_sample or config.temperature == 0.0:
                    return int(torch.argmax(logits_vec).item())

                # 3. Temperature scaling
                logits_vec = logits_vec / max(config.temperature, 1e-5)

                # 4. Top-K filtering
                if config.top_k is not None and config.top_k > 0:
                    top_k = min(config.top_k, logits_vec.size(-1))
                    indices_to_remove = logits_vec < torch.topk(logits_vec, top_k)[0][..., -1, None]
                    logits_vec[indices_to_remove] = float("-inf")

                # 5. Top-P (nucleus) filtering
                if config.top_p is not None and 0.0 < config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_vec, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    # Shift indices to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_vec[indices_to_remove] = float("-inf")

                # 6. Categorical sampling
                probs = torch.softmax(logits_vec, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1).item()
                return int(sampled_token)

        except ImportError:
            pass

        # Fallback for mock environments / pure Python numbers
        import math
        import random

        if isinstance(logits, list):
            vec = list(logits[0]) if (len(logits) > 0 and isinstance(logits[0], list)) else list(logits)
        elif hasattr(logits, "tolist"):
            l = logits.tolist()
            vec = l[0] if (isinstance(l, list) and len(l) > 0 and isinstance(l[0], list)) else l
        else:
            return 0

        # Greedy fallback
        if not config.do_sample or config.temperature == 0.0:
            return int(max(range(len(vec)), key=lambda i: vec[i]))

        # Softmax and sampling in Python
        temp = max(config.temperature, 1e-5)
        scaled = [v / temp for v in vec]
        max_val = max(scaled)
        exp_vals = [math.exp(v - max_val) for v in scaled]
        total_exp = sum(exp_vals)
        probs = [e / total_exp for e in exp_vals]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1
