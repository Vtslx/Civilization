from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import torch

from experiments.civilization_transformer_qwen3.analysis.stage45_adapter_package import (
    EXTERNAL_TASKS,
    Stage45CheckpointRef,
    Stage45InferenceRequest,
    Stage45PackageManifest,
    validate_inference_request,
)
from experiments.civilization_transformer_qwen3.analysis.stage46_runtime_inference import (
    _cosine_scores,
    _select_checkpoint,
)


def _manifest(tmp_path: Path) -> Stage45PackageManifest:
    checkpoints = []
    for role in ("full_hidden_alignment", "combined", *EXTERNAL_TASKS):
        path = tmp_path / f"{role}.pt"
        path.write_bytes(b"checkpoint")
        checkpoints.append(
            Stage45CheckpointRef(
                seed=202,
                scope="local" if role == "full_hidden_alignment" else "external",
                role=role,
                path=str(path),
                sha256="0" * 64,
                bytes=10,
                excludes_qwen_weights=True,
            )
        )
    return Stage45PackageManifest(
        package_version="stage45_v1",
        model_family="Qwen3-0.6B",
        architecture="dual_16_24_path_specific_v2",
        target_layers=(16, 24),
        raw_full_hidden_residual_scale=200.0,
        seeds=(202,),
        source_stage="stage44c_real_task_runner",
        stage44c_summary_path="summary.json",
        stage44c_summary_sha256="1" * 64,
        local_stage_passed=True,
        external_stage_passed=True,
        dataset_fields_cache_ok=True,
        checkpoints=tuple(checkpoints),
        external_dataset_manifest=(
            {"task_name": "glue_rte", "dataset_fields_structured_ok": True, "structure_sources": ["dataset_fields"]},
            {"task_name": "super_glue_cb", "dataset_fields_structured_ok": True, "structure_sources": ["dataset_fields"]},
            {"task_name": "boolq", "dataset_fields_structured_ok": True, "structure_sources": ["dataset_fields"]},
        ),
        inference_contract={
            "required_readouts": ["projected_delta", "raw_full_hidden_fixed_centroid"],
            "diagnostic_readouts": ["projected_full_hidden"],
        },
    )


def test_stage46_selects_local_and_task_specific_checkpoints(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert _select_checkpoint(manifest, 202, "custom").name == "full_hidden_alignment.pt"
    assert _select_checkpoint(manifest, 202, "boolq").name == "boolq.pt"
    external_without_task = tuple(ref for ref in manifest.checkpoints if ref.role != "boolq")
    fallback_manifest = replace(manifest, checkpoints=external_without_task)
    assert _select_checkpoint(fallback_manifest, 202, "boolq").name == "combined.pt"


def test_stage46_cosine_scores_are_finite_and_ordered() -> None:
    vector = torch.tensor([[1.0, 0.0]])
    options = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = _cosine_scores(vector, options)
    assert torch.isfinite(scores).all()
    assert scores.argmax(dim=-1).item() == 0


def test_stage46_projected_full_hidden_is_allowed_by_stage45_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    request = Stage45InferenceRequest(
        text="Evaluate this record.",
        memory_items=("memory",),
        rule_items=("rule",),
        state_values=(1.0, 0.0, 0.5),
        answer_options=("yes", "no"),
        seed=202,
        readouts=("projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"),
    )
    validate_inference_request(request, manifest)


def test_stage46_manifest_fixture_is_json_safe(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = {
        "seeds": list(manifest.seeds),
        "checkpoints": [ref.path for ref in manifest.checkpoints],
        "external_dataset_manifest": list(manifest.external_dataset_manifest),
    }
    assert json.loads(json.dumps(payload))["seeds"] == [202]
