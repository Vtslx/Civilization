"""Optional in-process host for the Civilization v1 production service.

The host is runtime-agnostic: the decision backend is selected by
``EmbeddedConfig.runtime`` and resolved through :mod:`civilization_v1.runtimes`.
A caller can also inject its own runtime factory, which is how the test suite
drives the service with a fake backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .client import CivilizationClient
from .models import CivilizationRequest, Prediction
from .runtimes import (
    AdapterRuntimeConfig,
    LocalTransformersRuntimeConfig,
    ProviderRuntimeConfig,
    ResolvedRuntime,
    RuntimeCapabilities,
    capabilities_for,
    resolve_runtime,
    runtime_kinds,
)

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class EmbeddedConfig:
    """Configuration for an in-process Civilization v1 service.

    ``runtime`` selects the decision backend. ``provider`` (the default) drives
    any OpenAI-compatible Chat Completions endpoint; ``local_transformers``
    runs a local Hugging Face model; ``local_adapter`` runs the audited adapter
    runtime. Provider fields are only validated for the ``provider`` runtime.
    """

    runtime: str = "provider"
    provider_base_url: str = ""
    provider_model: str = ""
    provider_api_key_env: str = "OPENAI_API_KEY"
    provider_api_key: str | None = None
    provider_headers: Mapping[str, str] = field(default_factory=dict)
    provider_user_agent: str = "aoneb-civilization-v1/0.00.08"
    provider_force_json_object: bool = True
    provider_extra_body: Mapping[str, Any] = field(default_factory=dict)
    allow_insecure_http: bool = False
    local_model_path: str = ""
    preferred_device: str = "auto"
    max_new_tokens: int = 64
    package_manifest: str = ""
    centroid_bundle: str = ""
    state_dir: str = "~/.aoneb/civilization-v1"
    bearer_token_env: str | None = "CIVILIZATION_API_TOKEN"
    host: str = "127.0.0.1"
    port: int = 0
    provider_timeout_seconds: float = 120.0
    max_tokens: int = 512
    replay_after_request: bool = True
    replay_min_episodes: int = 2

    def __post_init__(self) -> None:
        if self.runtime not in runtime_kinds():
            raise ValueError(
                f"unknown runtime {self.runtime!r}; registered kinds: {', '.join(runtime_kinds())}"
            )
        if self.runtime == "provider":
            parsed = urlparse(self.provider_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("provider_base_url must be an absolute HTTP(S) URL")
            if parsed.scheme != "https" and not (self.allow_insecure_http or parsed.hostname in _LOOPBACK_HOSTS):
                raise ValueError(
                    "provider_base_url must be an absolute HTTPS URL unless allow_insecure_http=True "
                    "or the host is loopback"
                )
            if not self.provider_model.strip():
                raise ValueError("provider_model must be non-empty")
        if self.runtime in {"local_transformers", "local_adapter"} and not self.local_model_path.strip():
            raise ValueError(f"local_model_path must be non-empty for the {self.runtime!r} runtime")
        if self.runtime == "local_adapter" and not (self.package_manifest.strip() and self.centroid_bundle.strip()):
            raise ValueError("package_manifest and centroid_bundle must be non-empty for the 'local_adapter' runtime")
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("embedded mode binds to loopback only; deploy the service separately for remote access")

    def runtime_config(self) -> Any:
        """Build the runtime-layer config object for the selected runtime."""

        if self.runtime == "provider":
            return ProviderRuntimeConfig(
                base_url=self.provider_base_url,
                model=self.provider_model,
                api_key_env=self.provider_api_key_env,
                api_key=self.provider_api_key,
                headers=dict(self.provider_headers),
                timeout_seconds=self.provider_timeout_seconds,
                max_tokens=self.max_tokens,
                allow_insecure_http=self.allow_insecure_http,
                user_agent=self.provider_user_agent,
                force_json_object=self.provider_force_json_object,
                extra_body=dict(self.provider_extra_body),
            )
        if self.runtime == "local_transformers":
            return LocalTransformersRuntimeConfig(
                model_path=self.local_model_path,
                preferred_device=self.preferred_device,
                max_new_tokens=self.max_new_tokens,
            )
        return AdapterRuntimeConfig(
            package_manifest=self.package_manifest,
            centroid_bundle=self.centroid_bundle,
            model_path=self.local_model_path,
            preferred_device=self.preferred_device,
        )


def _resolve(config: EmbeddedConfig) -> ResolvedRuntime:
    return resolve_runtime(config.runtime_config())


def _write_capabilities(state: Path, capabilities: RuntimeCapabilities) -> None:
    (state / "capabilities.json").write_text(
        json.dumps(capabilities.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_service(
    config: EmbeddedConfig,
    *,
    runtime_factory: Callable[[], Any] | None,
    capabilities: RuntimeCapabilities,
) -> Any:
    from experiments.civilization_transformer_qwen3.analysis.stage49_persistent_inference_service import Stage49ServiceConfig
    from experiments.civilization_transformer_qwen3.analysis.stage51_runtime_management import Stage51RuntimeDescriptor
    from experiments.civilization_transformer_qwen3.analysis.stage59_access_control_audit import Stage59SecurityConfig
    from experiments.civilization_transformer_qwen3.analysis.stage60_queue_rate_limit import Stage60QueueConfig
    from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig
    from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
    from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
    from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
    from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
    from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import Stage66ExportConfig
    from experiments.civilization_transformer_qwen3.analysis.stage67_export_lifecycle import Stage67ExportLifecycleConfig
    from experiments.civilization_transformer_qwen3.analysis.stage68_export_package_delivery import Stage68PackageConfig
    from experiments.civilization_transformer_qwen3.analysis.stage69_streaming_package_delivery import Stage69StreamingConfig
    from experiments.civilization_transformer_qwen3.analysis.stage74_orion_memory_service import Stage74MemoryConfig, Stage74OrionMemoryService

    state = Path(config.state_dir).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    _write_capabilities(state, capabilities)
    if runtime_factory is None:
        resolved = _resolve(config)
        runtime_factory = resolved.factory
        capabilities = resolved.capabilities
        _write_capabilities(state, capabilities)
    bearer_token = os.environ.get(config.bearer_token_env) if config.bearer_token_env else None
    if config.bearer_token_env and bearer_token is not None and not bearer_token.strip():
        raise ValueError(f"bearer token environment variable is empty: {config.bearer_token_env}")
    require_token = bearer_token is not None
    return Stage74OrionMemoryService(
        runtime_factory=runtime_factory,
        descriptor=Stage51RuntimeDescriptor(
            package_manifest=capabilities.label,
            centroid_bundle="not-applicable",
            label=f"civilization-v1:{capabilities.kind}",
        ),
        config=Stage49ServiceConfig(
            host=config.host,
            port=config.port,
            default_controls=("full",),
            capabilities=capabilities.to_dict(),
        ),
        security=Stage59SecurityConfig(
            bearer_token=bearer_token,
            require_token_for_predict=require_token,
            require_token_for_admin=require_token,
            require_token_for_metrics=require_token,
            audit_log_path=str(state / "audit.jsonl"),
        ),
        queue_config=Stage60QueueConfig(),
        job_config=Stage61JobConfig(job_log_path=str(state / "jobs.jsonl")),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state / "jobs-state.json")),
        retention_config=Stage63RetentionConfig(),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(state / "job-results")),
        batch_config=Stage65BatchConfig(),
        export_config=Stage66ExportConfig(export_dir=str(state / "exports")),
        lifecycle_config=Stage67ExportLifecycleConfig(),
        package_config=Stage68PackageConfig(),
        streaming_config=Stage69StreamingConfig(),
        memory_config=Stage74MemoryConfig(
            replay_after_request=config.replay_after_request,
            replay_min_episodes=config.replay_min_episodes,
            global_store_path=str(state / "global-memory.json"),
        ),
    )


def build_embedded_service(config: EmbeddedConfig, *, runtime_factory: Callable[[], Any] | None = None):
    """Build the Stage49-74 service chain with all mutable files under state_dir."""

    capabilities = capabilities_for(config.runtime)
    return _build_service(config, runtime_factory=runtime_factory, capabilities=capabilities)


class EmbeddedCivilization:
    """Own an in-process service and optionally expose its HTTP API."""

    def __init__(self, config: EmbeddedConfig, *, runtime_factory: Callable[[], Any] | None = None) -> None:
        self.config = config
        self.capabilities = capabilities_for(config.runtime)
        self._service = _build_service(config, runtime_factory=runtime_factory, capabilities=self.capabilities)
        self._server = None

    def predict(self, request: CivilizationRequest) -> Prediction:
        rows = self._service.predict_record(request.to_payload())
        if len(rows) != 1:
            raise RuntimeError("embedded inference did not return exactly one row")
        return Prediction.from_row(request, rows[0])

    def start(self) -> CivilizationClient:
        if self._server is None:
            self._server = self._service.start_background()
        host, port = self._server.server_address[:2]
        client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return CivilizationClient(
            f"http://{client_host}:{port}",
            token_env=self.config.bearer_token_env if self.config.bearer_token_env and os.environ.get(self.config.bearer_token_env) else None,
            timeout=self.config.provider_timeout_seconds,
        )

    def shutdown(self) -> None:
        self._service.shutdown()
        self._server = None

    def __enter__(self) -> "EmbeddedCivilization":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()
