"""Provider-agnostic runtime layer for the Civilization v1 framework.

The framework is deliberately not tied to one model, one vendor, or one
deployment shape. A runtime is any object that can answer the production
decision protocol; everything else -- where the weights live, who serves them,
whether hidden states exist -- is declared as capability instead of assumed.

Three runtime kinds ship with this SDK:

``provider``
    Any OpenAI-compatible Chat Completions endpoint (hosted API, self-hosted
    gateway, or local server). Text-only: no hidden states, no adapter.

``local_transformers``
    Any local Hugging Face causal language model. Text-only: no hidden states,
    no adapter, no pinned weights.

``local_adapter``
    The audited Civilization Adapter runtime. This is the only kind that
    executes the adapter and reads hidden states; it is validated for the pinned
    local model and package artifacts recorded in the v1 evidence trail.

Adding a backend is a registry entry, not a change to the service chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "AdapterRuntimeConfig",
    "LocalTransformersRuntimeConfig",
    "ProviderRuntimeConfig",
    "ResolvedRuntime",
    "RuntimeCapabilities",
    "RuntimeKind",
    "RUNTIME_REGISTRY",
    "capabilities_for",
    "register_runtime",
    "resolve_runtime",
    "runtime_kinds",
]


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What a runtime can and cannot do, stated instead of implied."""

    kind: str
    label: str
    provider_agnostic: bool
    local_weights: bool
    chat_completions: bool
    hidden_states: bool
    adapter_execution: bool
    ablation_controls: bool = False
    production_full_mode_only: bool = True
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "provider_agnostic": self.provider_agnostic,
            "local_weights": self.local_weights,
            "chat_completions": self.chat_completions,
            "hidden_states": self.hidden_states,
            "adapter_execution": self.adapter_execution,
            "ablation_controls": self.ablation_controls,
            "production_full_mode_only": self.production_full_mode_only,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    """An OpenAI-compatible Chat Completions endpoint."""

    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    max_tokens: int = 512
    allow_insecure_http: bool = False
    user_agent: str = "aoneb-civilization-v1/0.00.08"
    force_json_object: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTransformersRuntimeConfig:
    """A local Hugging Face causal language model."""

    model_path: str
    preferred_device: str = "auto"
    max_new_tokens: int = 64
    max_prompt_tokens: int = 2048
    local_files_only: bool = True


@dataclass(frozen=True)
class AdapterRuntimeConfig:
    """The audited Civilization Adapter runtime over local weights."""

    package_manifest: str
    centroid_bundle: str
    model_path: str
    preferred_device: str = "auto"
    max_length: int = 384


@dataclass(frozen=True)
class ResolvedRuntime:
    """A runtime factory plus the capabilities it is allowed to claim."""

    kind: str
    factory: Callable[[], Any]
    capabilities: RuntimeCapabilities


@dataclass(frozen=True)
class RuntimeKind:
    name: str
    config_type: type
    capabilities: RuntimeCapabilities
    build: Callable[[Any], ResolvedRuntime]


def _build_provider(config: ProviderRuntimeConfig) -> ResolvedRuntime:
    from experiments.civilization_transformer_qwen3.backend.openai_compatible_runtime import (
        OpenAICompatibleRuntime,
        OpenAICompatibleRuntimeConfig,
    )

    runtime_config = OpenAICompatibleRuntimeConfig(
        base_url=config.base_url,
        model=config.model,
        api_key_env=config.api_key_env,
        api_key=config.api_key,
        headers=dict(config.headers),
        timeout_seconds=config.timeout_seconds,
        max_tokens=config.max_tokens,
        allow_insecure_http=config.allow_insecure_http,
        user_agent=config.user_agent,
        force_json_object=config.force_json_object,
        extra_body=dict(config.extra_body),
    )
    capabilities = RuntimeCapabilities(
        kind="provider",
        label=f"openai-compatible:{config.model}",
        provider_agnostic=True,
        local_weights=False,
        chat_completions=True,
        hidden_states=False,
        adapter_execution=False,
        notes=(
            "any OpenAI-compatible Chat Completions endpoint",
            "text-only decisions: hidden states and adapter readouts are not available",
        ),
    )
    return ResolvedRuntime(kind="provider", factory=lambda: OpenAICompatibleRuntime(runtime_config), capabilities=capabilities)


def _build_local_transformers(config: LocalTransformersRuntimeConfig) -> ResolvedRuntime:
    from experiments.civilization_transformer_qwen3.backend.transformers_chat_runtime import (
        TransformersChatRuntime,
        TransformersChatRuntimeConfig,
    )

    runtime_config = TransformersChatRuntimeConfig(
        model_path=config.model_path,
        preferred_device=config.preferred_device,
        max_new_tokens=config.max_new_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        local_files_only=config.local_files_only,
    )
    capabilities = RuntimeCapabilities(
        kind="local_transformers",
        label=f"transformers:{config.model_path}",
        provider_agnostic=True,
        local_weights=True,
        chat_completions=False,
        hidden_states=False,
        adapter_execution=False,
        notes=(
            "model-agnostic local causal LM; weights are not pinned",
            "text-only decisions: no adapter execution and no hidden-state readouts",
        ),
    )
    return ResolvedRuntime(
        kind="local_transformers",
        factory=lambda: TransformersChatRuntime(runtime_config),
        capabilities=capabilities,
    )


def _build_local_adapter(config: AdapterRuntimeConfig) -> ResolvedRuntime:
    from experiments.civilization_transformer_qwen3.analysis.stage48_production_jsonl_inference import (
        build_stage48_runtime,
    )

    capabilities = RuntimeCapabilities(
        kind="local_adapter",
        label=f"civilization-adapter:{config.model_path}",
        provider_agnostic=True,
        local_weights=True,
        chat_completions=False,
        hidden_states=True,
        adapter_execution=True,
        notes=(
            "the only kind that executes the Civilization Adapter and reads hidden states",
            "validated for the pinned local model and the recorded Stage45 package artifacts",
        ),
    )
    return ResolvedRuntime(
        kind="local_adapter",
        factory=lambda: build_stage48_runtime(
            package_manifest=config.package_manifest,
            centroid_bundle=config.centroid_bundle,
            model_path=config.model_path,
            preferred_device=config.preferred_device,
            max_length=config.max_length,
        ),
        capabilities=capabilities,
    )


RUNTIME_REGISTRY: dict[str, RuntimeKind] = {
    "provider": RuntimeKind(
        name="provider",
        config_type=ProviderRuntimeConfig,
        capabilities=RuntimeCapabilities(
            kind="provider",
            label="openai-compatible",
            provider_agnostic=True,
            local_weights=False,
            chat_completions=True,
            hidden_states=False,
            adapter_execution=False,
        ),
        build=_build_provider,
    ),
    "local_transformers": RuntimeKind(
        name="local_transformers",
        config_type=LocalTransformersRuntimeConfig,
        capabilities=RuntimeCapabilities(
            kind="local_transformers",
            label="transformers-local",
            provider_agnostic=True,
            local_weights=True,
            chat_completions=False,
            hidden_states=False,
            adapter_execution=False,
        ),
        build=_build_local_transformers,
    ),
    "local_adapter": RuntimeKind(
        name="local_adapter",
        config_type=AdapterRuntimeConfig,
        capabilities=RuntimeCapabilities(
            kind="local_adapter",
            label="civilization-adapter",
            provider_agnostic=True,
            local_weights=True,
            chat_completions=False,
            hidden_states=True,
            adapter_execution=True,
        ),
        build=_build_local_adapter,
    ),
}


def register_runtime(kind: RuntimeKind) -> None:
    """Register a runtime kind so callers can select it by name."""

    if not kind.name.strip():
        raise ValueError("runtime kind name must be non-empty")
    RUNTIME_REGISTRY[kind.name] = kind


def runtime_kinds() -> tuple[str, ...]:
    return tuple(sorted(RUNTIME_REGISTRY))


def capabilities_for(kind: str) -> RuntimeCapabilities:
    try:
        return RUNTIME_REGISTRY[kind].capabilities
    except KeyError as error:
        raise ValueError(f"unknown runtime kind {kind!r}; registered kinds: {', '.join(runtime_kinds())}") from error


def resolve_runtime(config: Any) -> ResolvedRuntime:
    """Resolve a runtime config into a factory plus declared capabilities."""

    for kind in RUNTIME_REGISTRY.values():
        if isinstance(config, kind.config_type):
            resolved = kind.build(config)
            if not isinstance(resolved, ResolvedRuntime):
                raise TypeError(f"runtime kind {kind.name!r} did not return a ResolvedRuntime")
            return resolved
    raise ValueError(
        "unsupported runtime config; expected one of: "
        + ", ".join(kind.config_type.__name__ for kind in RUNTIME_REGISTRY.values())
    )
