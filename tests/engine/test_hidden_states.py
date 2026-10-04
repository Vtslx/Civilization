from pathlib import Path

import torch

from civilization.engine.stages.hidden_states import collect_qwen3_hidden_states
from civilization.engine.backend import Qwen3Backend
from civilization.research.torch_line.analysis.dataset import build_logic_variant_datasets


import pytest
from civilization.engine.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_hidden_state_collection_has_expected_shapes_and_padding_stability() -> None:
    backend = Qwen3Backend(MODEL_PATH, preferred_device="cpu")
    datasets, _ = build_logic_variant_datasets(
        samples_per_label=1,
        max_seq_len=32,
        seed=202,
        variants=("canonical",),
        template_bank="expanded_v1",
    )
    samples = datasets["canonical"][:2]

    short = collect_qwen3_hidden_states(
        backend,
        samples,
        max_length=64,
        batch_size=1,
        selected_layers=(0, 28),
    )
    long = collect_qwen3_hidden_states(
        backend,
        samples,
        max_length=128,
        batch_size=2,
        selected_layers=(0, 28),
    )

    assert short.num_samples == 2
    assert short.representations["mean"][28].shape == (2, 1024)
    assert short.representations["last"][28].shape == (2, 1024)
    assert torch.isfinite(short.representations["mean"][28]).all()
    assert torch.allclose(short.representations["mean"][28], long.representations["mean"][28], atol=1e-5)
    assert torch.allclose(short.representations["last"][28], long.representations["last"][28], atol=1e-5)
