from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
import time
from typing import Any

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
EXPECTED_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "vocab_size": 151936,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
}


@dataclass
class Qwen3BackendOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    device: str
    dtype: str
    runtime_trace: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Qwen3Backend:
    def __init__(
        self,
        model_path: str | Path,
        preferred_device: str | None = None,
        expected_sha256: str = EXPECTED_SHA256,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Qwen3 model directory does not exist: {self.model_path}")
        self.weights_path = self.model_path / "model.safetensors"
        self.expected_sha256 = expected_sha256
        self.initial_sha256 = sha256_file(self.weights_path)
        if self.initial_sha256 != expected_sha256:
            raise ValueError(f"Qwen3 weights SHA256 mismatch: {self.initial_sha256}")

        self.runtime_trace: dict[str, Any] = {
            "model_path": str(self.model_path),
            "local_files_only": True,
            "trust_remote_code": False,
            "fallbacks": [],
            "load_attempts": [],
        }
        self.config = AutoConfig.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._validate_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        candidates = self._device_candidates(preferred_device)
        self.model = None
        last_error: Exception | None = None
        for device, dtype in candidates:
            started = time.perf_counter()
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=dtype,
                )
                self.model.config.use_cache = False
                self.model.eval()
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                self.model.to(device)
                self.device = torch.device(device)
                self.dtype = dtype
                self.runtime_trace["load_attempts"].append(
                    {
                        "device": device,
                        "dtype": str(dtype),
                        "success": True,
                        "seconds": time.perf_counter() - started,
                    }
                )
                break
            except Exception as error:
                last_error = error
                self.runtime_trace["load_attempts"].append(
                    {
                        "device": device,
                        "dtype": str(dtype),
                        "success": False,
                        "seconds": time.perf_counter() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                self.runtime_trace["fallbacks"].append(
                    {
                        "from_device": device,
                        "from_dtype": str(dtype),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        if self.model is None:
            raise RuntimeError("unable to load Qwen3 on any supported device") from last_error
        if type(self.model).__name__ != "Qwen3ForCausalLM":
            raise ValueError(f"unexpected model architecture: {type(self.model).__name__}")
        self.runtime_trace["device"] = str(self.device)
        self.runtime_trace["dtype"] = str(self.dtype)
        self.runtime_trace["parameter_count"] = sum(parameter.numel() for parameter in self.model.parameters())
        self.runtime_trace["trainable_parameter_count"] = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )

    def _validate_config(self) -> None:
        if self.config.model_type != "qwen3":
            raise ValueError(f"unexpected model_type: {self.config.model_type}")
        architectures = list(getattr(self.config, "architectures", []))
        if architectures != ["Qwen3ForCausalLM"]:
            raise ValueError(f"unexpected architectures: {architectures}")
        for field, expected in EXPECTED_CONFIG.items():
            actual = getattr(self.config, field)
            if actual != expected:
                raise ValueError(f"unexpected {field}: expected {expected}, got {actual}")

    @staticmethod
    def _device_candidates(preferred_device: str | None) -> list[tuple[str, torch.dtype]]:
        if preferred_device == "cpu":
            return [("cpu", torch.float32)]
        if preferred_device == "cuda":
            if not torch.cuda.is_available():
                return [("cpu", torch.float32)]
            return [
                ("cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                ("cpu", torch.float32),
            ]
        if preferred_device == "mps":
            if not torch.backends.mps.is_available():
                return [("cpu", torch.float32)]
            return [
                ("mps", torch.bfloat16),
                ("mps", torch.float16),
                ("cpu", torch.float32),
            ]
        if preferred_device not in {None, "auto"}:
            raise ValueError(f"unsupported preferred_device: {preferred_device}")
        if torch.cuda.is_available():
            return [
                ("cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                ("cpu", torch.float32),
            ]
        if torch.backends.mps.is_available():
            return [
                ("mps", torch.bfloat16),
                ("mps", torch.float16),
                ("cpu", torch.float32),
            ]
        return [("cpu", torch.float32)]

    def encode(
        self,
        texts: list[str],
        max_length: int = 128,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts must contain at least one non-empty string")
        full = self.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        truncations: list[dict[str, Any]] = []
        for index, ids in enumerate(full["input_ids"]):
            if len(ids) > max_length:
                truncations.append(
                    {
                        "sample_index": index,
                        "original_tokens": len(ids),
                        "truncated_tokens": max_length,
                        "core_logic_may_be_lost": True,
                    }
                )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }, truncations

    def forward_encoded(self, encoded: dict[str, torch.Tensor]) -> Qwen3BackendOutput:
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        started = time.perf_counter()
        rss_before = psutil.Process().memory_info().rss
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if len(output.hidden_states) != 29:
            raise ValueError(f"expected 29 hidden-state layers, got {len(output.hidden_states)}")
        if not torch.isfinite(output.logits).all():
            raise ValueError("Qwen3 logits contain NaN or Inf")
        if any(not torch.isfinite(hidden).all() for hidden in output.hidden_states):
            raise ValueError("Qwen3 hidden states contain NaN or Inf")
        trace = {
            "forward_seconds": time.perf_counter() - started,
            "rss_before": rss_before,
            "rss_after": psutil.Process().memory_info().rss,
            "mps_allocated": torch.mps.current_allocated_memory() if self.device.type == "mps" else 0,
            "logits_shape": list(output.logits.shape),
            "hidden_state_count": len(output.hidden_states),
        }
        return Qwen3BackendOutput(
            logits=output.logits,
            hidden_states=tuple(output.hidden_states),
            attention_mask=attention_mask,
            input_ids=input_ids,
            device=str(self.device),
            dtype=str(self.dtype),
            runtime_trace=trace,
        )

    def inference_forward(self, encoded: dict[str, torch.Tensor]) -> Qwen3BackendOutput:
        return self.forward_encoded(encoded)

    def adapter_training_forward(
        self,
        encoded: dict[str, torch.Tensor],
        adapter_model,
        context,
    ):
        return adapter_model(encoded, context)

    def forward_texts(self, texts: list[str], max_length: int = 128) -> tuple[Qwen3BackendOutput, list[dict[str, Any]]]:
        encoded, truncations = self.encode(texts, max_length=max_length)
        return self.forward_encoded(encoded), truncations

    def verify_weights_unchanged(self) -> bool:
        return sha256_file(self.weights_path) == self.initial_sha256 == self.expected_sha256

    def parameter_fingerprint(self) -> dict[str, float | int]:
        with torch.no_grad():
            total_sum = 0.0
            total_square_sum = 0.0
            tensor_count = 0
            element_count = 0
            for parameter in self.model.parameters():
                # MPS reductions over the full model can vary between calls.
                # Compute the frozen-weight fingerprint on CPU instead.
                values = parameter.detach().to(device="cpu", dtype=torch.float32)
                total_sum += float(values.sum())
                total_square_sum += float(values.square().sum())
                tensor_count += 1
                element_count += parameter.numel()
        return {
            "tensor_count": tensor_count,
            "element_count": element_count,
            "sum": total_sum,
            "square_sum": total_square_sum,
        }

    def chat_sanity_check(self, prompt: str = "Reply with one word: ready") -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                use_cache=False,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        return {
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "generated_tokens": int(new_tokens.shape[0]),
            "text": self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        }
