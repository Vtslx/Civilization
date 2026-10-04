"""AstreusN Civilization — decision, memory, job, and export service SDK.

Quick start without any configuration:

    python -m civilization demo

Host the service in-process:

    from civilization import EmbeddedCivilization, EmbeddedConfig

    service = EmbeddedCivilization(
        EmbeddedConfig(
            runtime="provider",
            provider_base_url="https://provider.example.com/v1",
            provider_model="any-chat-model",
            provider_api_key_env="PROVIDER_API_KEY",
        )
    )
    with service:
        client = service.start()
        print(client.ready())

Talk to a deployed service:

    from civilization import CivilizationClient, CivilizationRequest

    client = CivilizationClient("https://civilization.example.com", token_env="CIVILIZATION_API_TOKEN")
    print(client.predict(CivilizationRequest(text="Choose.", answer_options=("approve", "reject"))))
"""

from __future__ import annotations

from .client import (
    CivilizationAPIError,
    CivilizationClient,
    CivilizationInferenceError,
    CivilizationSDKError,
    CivilizationTransportError,
)
from .models import CivilizationRequest, Job, MemorySystem, Prediction
from .runtimes import (
    AdapterRuntimeConfig,
    LocalTransformersRuntimeConfig,
    ProviderRuntimeConfig,
    ResolvedRuntime,
    RuntimeCapabilities,
    RuntimeKind,
    capabilities_for,
    register_runtime,
    resolve_runtime,
    runtime_kinds,
)

__version__ = "0.1.0"

__all__ = [
    "AdapterRuntimeConfig",
    "CivilizationAPIError",
    "CivilizationClient",
    "CivilizationInferenceError",
    "CivilizationRequest",
    "CivilizationSDKError",
    "CivilizationTransportError",
    "EmbeddedCivilization",
    "EmbeddedConfig",
    "Job",
    "LocalTransformersRuntimeConfig",
    "MemorySystem",
    "Prediction",
    "ProviderRuntimeConfig",
    "ResolvedRuntime",
    "RuntimeCapabilities",
    "RuntimeKind",
    "__version__",
    "build_embedded_service",
    "capabilities_for",
    "register_runtime",
    "resolve_runtime",
    "runtime_kinds",
]

_LAZY_EXPORTS = {
    "EmbeddedCivilization": ".embedded",
    "EmbeddedConfig": ".embedded",
    "build_embedded_service": ".embedded",
}


def __getattr__(name: str):
    """Import the service host lazily so the SDK stays dependency-free."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)
