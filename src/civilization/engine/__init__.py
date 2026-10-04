"""Decision engine: the frozen-base line that backs the Civilization service.

Layout:

- ``model_paths``  resolves the optional local checkpoint (environment-driven).
- ``adapter``      the trainable Civilization Adapter and its context encoder.
- ``backend``      runtime implementations: local transformers, OpenAI-compatible,
                   and the audited adapter runtime.
- ``stages``       the versioned service chain (Stages 44-160): inference service,
                   access control, queueing, jobs, exports, and the Orion memory
                   subsystems.

Nothing in this package is imported by the dependency-free client path; the
engine is loaded only when a service is actually hosted.
"""

from __future__ import annotations

from .model_paths import DEFAULT_MODEL_PATH, MODEL_PATH_ENV, model_available, require_model_path

__all__ = [
    "DEFAULT_MODEL_PATH",
    "MODEL_PATH_ENV",
    "Qwen3Backend",
    "Qwen3BackendOutput",
    "model_available",
    "require_model_path",
]


def __getattr__(name: str):
    """Expose the local backend lazily so importing the engine stays cheap."""

    if name in {"Qwen3Backend", "Qwen3BackendOutput"}:
        from .backend import Qwen3Backend, Qwen3BackendOutput

        return {"Qwen3Backend": Qwen3Backend, "Qwen3BackendOutput": Qwen3BackendOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
