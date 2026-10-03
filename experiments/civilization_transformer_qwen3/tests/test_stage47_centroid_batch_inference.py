from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.stage47_centroid_batch_inference import (
    Stage47CentroidProvider,
    save_centroid_bundle,
)


def test_stage47_centroid_bundle_round_trip(tmp_path: Path) -> None:
    bundle = tmp_path / "centroids.pt"
    manifest = save_centroid_bundle(
        bundle,
        centroids={(202, "operation_decision"): torch.arange(5 * 1024, dtype=torch.float32).reshape(5, 1024)},
        options={"operation_decision": ("a", "b", "c", "d", "e")},
        source_audit=[{"split": "train", "mode": "full", "used": True}],
        metadata={"qwen_frozen": True},
    )
    assert manifest["entry_count"] == 1
    assert not manifest["contains_qwen_weights"]
    provider = Stage47CentroidProvider(bundle)
    matrix = provider.get(202, "operation_decision", 5)
    assert matrix is not None
    assert matrix.shape == (5, 1024)
    assert provider.options(202, "operation_decision") == ("a", "b", "c", "d", "e")
    assert provider.get(303, "operation_decision", 5) is None


def test_stage47_rejects_centroid_option_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        save_centroid_bundle(
            tmp_path / "bad.pt",
            centroids={(202, "task"): torch.zeros(5, 1024)},
            options={"task": ("a", "b")},
            source_audit=[],
            metadata={},
        )


def test_stage47_rejects_bundle_with_model_weights(tmp_path: Path) -> None:
    bundle = tmp_path / "bad.pt"
    torch.save(
        {
            "bundle_version": "stage47_raw_full_hidden_centroids_v1",
            "entries": {},
            "qwen_state_dict": {},
        },
        bundle,
    )
    with pytest.raises(ValueError, match="forbidden model weights"):
        Stage47CentroidProvider(bundle)


def test_stage47_provider_rejects_nonfinite_centroids(tmp_path: Path) -> None:
    bundle = tmp_path / "bad_values.pt"
    matrix = torch.zeros(2, 1024)
    matrix[0, 0] = float("nan")
    torch.save(
        {
            "bundle_version": "stage47_raw_full_hidden_centroids_v1",
            "entries": {
                "202:task": {
                    "seed": 202,
                    "task_name": "task",
                    "options": ["a", "b"],
                    "centroids": matrix,
                }
            },
            "source_audit": [],
            "metadata": {},
        },
        bundle,
    )
    provider = Stage47CentroidProvider(bundle)
    with pytest.raises(ValueError, match="invalid centroid matrix"):
        provider.get(202, "task", 2)
