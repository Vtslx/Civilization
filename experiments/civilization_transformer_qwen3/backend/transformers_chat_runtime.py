"""Model-agnostic local transformers runtime for the v1 production service.

This backend runs any local Hugging Face causal language model through the
shared decision protocol in :mod:`decision_prompt`. It is the general-purpose
local option: no pinned weights, no architecture assertion, no adapter package,
and no hidden-state claim.

Capability boundary: this runtime performs text-only decision decoding. It does
not execute the Civilization Adapter and does not expose hidden states, so it
reports ``adapter_execution=False`` and ``hidden_states_available=False`` in
every trace. Systems that need the audited adapter path must use the validated
adapter runtime instead of this one.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from ..analysis.stage45_adapter_package import (
    Stage45InferenceRequest,
    Stage45InferenceResponse,
)
from .decision_prompt import build_chat_messages, parse_option_id


@dataclass(frozen=True)
class TransformersChatRuntimeConfig:
    model_path: str
    preferred_device: str = "auto"
    max_new_tokens: int = 64
    max_prompt_tokens: int = 2048
    local_files_only: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1")


def device_candidates(preferred_device: str) -> list[str]:
    """Ordered device preference list, independent of any specific model."""

    import torch

    if preferred_device == "cpu":
        return ["cpu"]
    if preferred_device == "cuda":
        return ["cuda"] if torch.cuda.is_available() else ["cpu"]
    if preferred_device == "mps":
        available = getattr(torch.backends, "mps", None)
        return ["mps"] if available is not None and available.is_available() else ["cpu"]
    candidates: list[str] = []
    if torch.cuda.is_available():
        candidates.append("cuda")
    available = getattr(torch.backends, "mps", None)
    if available is not None and available.is_available():
        candidates.append("mps")
    candidates.append("cpu")
    return candidates


class TransformersChatRuntime:
    """Local causal LM runtime implementing the Stage48 ``RuntimePredictor``."""

    def __init__(
        self,
        config: TransformersChatRuntimeConfig,
        *,
        backend_factory: Callable[[], tuple[Any, Any, str]] | None = None,
    ) -> None:
        self.config = config
        self.request_count = 0
        self.device = "cpu"
        if backend_factory is None:
            self._tokenizer, self._model = self._load()
        else:
            self._tokenizer, self._model, self.device = backend_factory()

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        last_error: Exception | None = None
        for device in device_candidates(self.config.preferred_device):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    local_files_only=self.config.local_files_only,
                    trust_remote_code=self.config.trust_remote_code,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    local_files_only=self.config.local_files_only,
                    trust_remote_code=self.config.trust_remote_code,
                    dtype=torch.float32 if device == "cpu" else torch.bfloat16,
                )
                model.to(device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self.device = device
                return tokenizer, model
            except Exception as error:  # noqa: BLE001 - device fallback
                last_error = error
                continue
        raise RuntimeError("unable to load the local transformers model on any device") from last_error

    def _render_prompt(self, request: Stage45InferenceRequest) -> str:
        messages = build_chat_messages(request)
        apply_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return apply_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # noqa: BLE001 - models without a usable chat template
                pass
        return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"

    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        if request.control_mode != "full":
            raise ValueError("local transformers runtime supports only control_mode='full'")
        import torch

        prompt = self._render_prompt(request)
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_tokens,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        started = time.perf_counter()
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=getattr(self._tokenizer, "pad_token_id", None),
            )
        elapsed = time.perf_counter() - started
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        completion = generated[0][prompt_tokens:]
        content = self._tokenizer.decode(completion, skip_special_tokens=True)
        predicted = parse_option_id(content, request.answer_options)
        scores = [0.0] * len(request.answer_options)
        scores[predicted] = 1.0
        self.request_count += 1
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=predicted,
            scores={"api_choice": scores},
            trace={
                "runtime": "transformers_local_chat",
                "production_full_mode": True,
                "local_model_path": self.config.model_path,
                "local_device": self.device,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(completion.shape[-1]),
                "latency_seconds": elapsed,
                "memory_item_count": len(request.memory_items),
                "rule_item_count": len(request.rule_items),
                "adapter_execution": False,
                "hidden_states_available": False,
                "ablation_controls_executed": False,
            },
        )
