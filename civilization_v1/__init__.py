"""Public API for the AstreusN Civilization v1 SDK."""

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
    "build_embedded_service",
    "capabilities_for",
    "register_runtime",
    "resolve_runtime",
    "runtime_kinds",
]


def __getattr__(name: str):
    if name in {"EmbeddedCivilization", "EmbeddedConfig", "build_embedded_service"}:
        from .embedded import EmbeddedCivilization, EmbeddedConfig, build_embedded_service

        exports = {
            "EmbeddedCivilization": EmbeddedCivilization,
            "EmbeddedConfig": EmbeddedConfig,
            "build_embedded_service": build_embedded_service,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
