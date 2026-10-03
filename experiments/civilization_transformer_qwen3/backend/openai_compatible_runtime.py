"""OpenAI-compatible production runtime for the v1 service stack.

This backend adapts any Chat Completions endpoint -- a hosted provider, a
self-hosted gateway, or a local server that speaks the same protocol -- to the
Stage48 ``RuntimePredictor`` contract. The provider is configuration, not code:
base URL, model name, credentials, headers, and provider-specific body fields
all come from config.

It intentionally supports production ``full`` requests only. Chat Completions
providers return text and do not expose transformer hidden states, so this
runtime never claims to execute the Civilization Adapter, its hidden-state
readouts, or its ablation controls.

Reasoning-heavy providers can spend an entire completion budget on hidden
reasoning and return an empty message. Rather than failing the decision, this
runtime walks a bounded retry ladder (JSON mode -> reasoning suppression ->
assistant prefill) and learns which features a deployment accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..analysis.stage45_adapter_package import (
    Stage45InferenceRequest,
    Stage45InferenceResponse,
)
from .decision_prompt import SYSTEM_PROMPT, build_decision_prompt, parse_option_id

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

_RETRY_NUDGE = (
    "\n\nAnswer immediately with only the JSON object. "
    "Do not write reasoning, explanation, or markdown fences."
)
_PREFILL = '{"option_id":'


def _completion_diagnostic(completion: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Non-secret provider diagnostics for an unusable completion."""

    payload = completion.get("payload") or {}
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details")
    return {
        "finish_reason": completion.get("finish_reason"),
        "max_tokens": max_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
    }


@dataclass(frozen=True)
class OpenAICompatibleRuntimeConfig:
    """Provider configuration for a Chat Completions endpoint."""

    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    max_tokens: int = 512
    user_agent: str = "aoneb-civilization-v1/0.00.08"
    allow_insecure_http: bool = False
    empty_content_retries: int = 2
    force_json_object: bool = True
    suppress_reasoning_on_retry: bool = True
    prefill_on_retry: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and not (self.allow_insecure_http or parsed.hostname in _LOOPBACK_HOSTS):
            raise ValueError(
                "base_url must be an absolute HTTPS URL unless allow_insecure_http=True or the host is loopback"
            )
        if not str(self.model).strip():
            raise ValueError("model must be non-empty")
        if self.api_key is None and not str(self.api_key_env).strip():
            raise ValueError("api_key_env must be non-empty unless api_key is provided directly")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.empty_content_retries < 0:
            raise ValueError("empty_content_retries must be >= 0")
        object.__setattr__(self, "headers", {str(name): str(value) for name, value in dict(self.headers).items()})
        object.__setattr__(self, "extra_body", dict(self.extra_body))


class OpenAICompatibleRuntime:
    """Translate one v1 production request into one Chat Completions call."""

    def __init__(
        self,
        config: OpenAICompatibleRuntimeConfig,
        *,
        urlopen_fn: Callable[..., Any] = urlopen,
    ) -> None:
        self.config = config
        self._urlopen = urlopen_fn
        self.request_count = 0
        self.last_request: Stage45InferenceRequest | None = None
        # Learned per deployment: features a provider rejects are dropped for
        # later requests instead of failing every decision.
        self._json_object_supported: bool | None = None
        self._reasoning_effort_supported: bool | None = None
        self._prefill_supported: bool | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _api_key(self) -> str:
        if self.config.api_key is not None:
            value = self.config.api_key
            if not value.strip():
                raise RuntimeError("configured provider API key is empty")
            return value
        value = os.environ.get(self.config.api_key_env, "")
        if not value.strip():
            raise RuntimeError(f"API key environment variable is missing: {self.config.api_key_env}")
        return value

    @staticmethod
    def _prompt(request: Stage45InferenceRequest) -> str:
        return build_decision_prompt(request)

    @staticmethod
    def _parse_option_id(content: str, options: tuple[str, ...]) -> int:
        return parse_option_id(content, options)

    def _attempt_plans(self) -> list[dict[str, Any]]:
        """Bounded ladder of request shapes tried until the provider answers.

        Attempt 1 is the plain production request. Later attempts add an explicit
        instruction, reasoning suppression, and finally an assistant prefill that
        forces the answer to start immediately -- the shapes that measurably stop
        reasoning-heavy providers from returning an empty message.
        """

        reasoning_effort = "none" if self.config.suppress_reasoning_on_retry else None
        plans: list[dict[str, Any]] = [{"nudge": False, "reasoning_effort": None, "prefill": False}]
        if self.config.empty_content_retries >= 1:
            plans.append({"nudge": True, "reasoning_effort": reasoning_effort, "prefill": False})
        if self.config.empty_content_retries >= 2 and self.config.prefill_on_retry:
            plans.append({"nudge": True, "reasoning_effort": reasoning_effort, "prefill": True})
        return plans

    def _completion(self, request: Stage45InferenceRequest, plan: Mapping[str, Any]) -> dict[str, Any]:
        """One Chat Completions call under one attempt plan."""

        prompt = build_decision_prompt(request)
        if plan.get("nudge"):
            prompt += _RETRY_NUDGE
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prefill = bool(plan.get("prefill")) and self._prefill_supported is not False
        if prefill:
            messages.append({"role": "assistant", "content": _PREFILL})
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
        }
        # JSON mode is a standard OpenAI field and keeps reasoning-heavy providers
        # from spending the whole completion budget before emitting the answer.
        if self.config.force_json_object and self._json_object_supported is not False and not prefill:
            body["response_format"] = {"type": "json_object"}
        reasoning_effort = plan.get("reasoning_effort")
        if reasoning_effort and self._reasoning_effort_supported is not False:
            body["reasoning_effort"] = reasoning_effort
        body.update(self.config.extra_body)
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
            **self.config.headers,
        }
        transport_request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._urlopen(transport_request, timeout=self.config.timeout_seconds) as response:
                status_code = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            provider_error = error.read().decode("utf-8", "replace")[:1000]
            if self._rejected_feature(provider_error, prefill=prefill, reasoning_effort=reasoning_effort):
                return self._completion(request, plan)
            raise RuntimeError(f"provider HTTP {error.code}: {provider_error}") from error
        if self.config.force_json_object and self._json_object_supported is None and not prefill:
            self._json_object_supported = True
        if reasoning_effort and self._reasoning_effort_supported is None:
            self._reasoning_effort_supported = True
        if prefill and self._prefill_supported is None:
            self._prefill_supported = True
        choices = payload.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            raise ValueError("provider response does not contain a choice")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str):
            raise ValueError("provider response content is not text")
        raw_content = content
        completed = content if content.lstrip().startswith("{") else _PREFILL + content
        return {
            "status_code": status_code,
            "payload": payload,
            "content": completed,
            "raw_content": raw_content,
            "finish_reason": choice.get("finish_reason"),
            "used_prefill": prefill,
        }

    def _rejected_feature(self, provider_error: str, *, prefill: bool, reasoning_effort: Any) -> bool:
        """Learn from a provider rejection so the next call omits that feature."""

        if self.config.force_json_object and self._json_object_supported is not False and not prefill:
            if "response_format" in provider_error or "json_object" in provider_error:
                self._json_object_supported = False
                return True
        if reasoning_effort and self._reasoning_effort_supported is not False:
            if "reasoning_effort" in provider_error or "reasoning" in provider_error:
                self._reasoning_effort_supported = False
                return True
        if prefill and self._prefill_supported is not False:
            if "assistant" in provider_error or "prefill" in provider_error or "messages" in provider_error:
                self._prefill_supported = False
                return True
        return False

    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        if request.control_mode != "full":
            raise ValueError("OpenAI-compatible production runtime supports only control_mode='full'")
        started = time.perf_counter()
        diagnostics: list[dict[str, Any]] = []
        completion: dict[str, Any] | None = None
        used_plan: dict[str, Any] = {}
        for plan in self._attempt_plans():
            if plan.get("prefill") and self._prefill_supported is False:
                continue
            if plan.get("reasoning_effort") and self._reasoning_effort_supported is False:
                continue
            completion = self._completion(request, plan)
            if completion["raw_content"].strip():
                used_plan = plan
                break
            diagnostics.append(
                {
                    **_completion_diagnostic(completion, self.config.max_tokens),
                    "plan": {name: value for name, value in plan.items() if value not in (None, False)},
                }
            )
        if completion is None or not completion["raw_content"].strip():
            raise ValueError(
                "provider response was empty after "
                f"{len(diagnostics)} attempts: {json.dumps(diagnostics, ensure_ascii=False)}"
            )
        elapsed = time.perf_counter() - started
        payload = completion["payload"]
        predicted = parse_option_id(completion["content"], request.answer_options)
        scores = [0.0] * len(request.answer_options)
        scores[predicted] = 1.0
        self.request_count += 1
        self.last_request = request
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=predicted,
            scores={"api_choice": scores},
            trace={
                "runtime": "openai_compatible_chat_completions",
                "production_full_mode": True,
                "provider_http_status": completion["status_code"],
                "provider_model": payload.get("model"),
                "provider_response_id": payload.get("id"),
                "provider_usage": payload.get("usage", {}),
                "provider_finish_reason": completion["finish_reason"],
                "provider_json_object_mode": self._json_object_supported is not False
                and self.config.force_json_object
                and not used_plan.get("prefill"),
                "provider_reasoning_effort": used_plan.get("reasoning_effort"),
                "provider_prefill_used": bool(used_plan.get("prefill")),
                "provider_attempts": len(diagnostics) + 1,
                "provider_empty_responses": len(diagnostics),
                "latency_seconds": elapsed,
                "memory_item_count": len(request.memory_items),
                "rule_item_count": len(request.rule_items),
                "adapter_execution": False,
                "hidden_states_available": False,
                "ablation_controls_executed": False,
            },
        )
