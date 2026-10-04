"""Locate the local Qwen3-0.6B checkpoint used by the real-model stages.

The checkpoint is **not** part of this repository. Point ``CIVILIZATION_MODEL_PATH``
at a local copy, or place one at ``Models/Qwen3-0.6B`` relative to the repository
root. Everything that only needs the model weights resolves the path here, so no
stage hard-codes a machine-specific location.

The pinned SHA-256 in :mod:`civilization.engine.backend.qwen3_backend`
is a validation constant for the audited checkpoint, not a runtime requirement of
the framework: the provider runtime and the generic local transformers runtime do
not need this checkpoint at all.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_PATH_ENV = "CIVILIZATION_MODEL_PATH"
DEFAULT_MODEL_PATH = Path(os.environ.get(MODEL_PATH_ENV, "Models/Qwen3-0.6B")).expanduser()

MISSING_MODEL_REASON = (
    f"local Qwen3-0.6B checkpoint not found at {DEFAULT_MODEL_PATH}; "
    f"place it there or set {MODEL_PATH_ENV} to a local copy"
)


def model_available() -> bool:
    """True when the local checkpoint directory exists."""

    return DEFAULT_MODEL_PATH.is_dir()


def require_model_path() -> Path:
    """Return the checkpoint path, skipping the calling test when it is absent."""

    import pytest

    if not model_available():
        pytest.skip(MISSING_MODEL_REASON)
    return DEFAULT_MODEL_PATH
