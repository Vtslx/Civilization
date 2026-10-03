from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.stage45_adapter_package import (
    EXTERNAL_TASKS,
    Stage45InferenceEngine,
    Stage45InferenceRequest,
    build_stage45_adapter_package,
    checkpoint_contains_qwen_weights,
    load_stage45_package_manifest,
    run_stage45_eval_harness,
    validate_inference_request,
    validate_stage45_package,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_checkpoint(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": {"scale": torch.tensor([float(seed)])},
            "projector_state_dict": {"weight": torch.eye(2)},
            "training_metadata": {"seed": seed, "adapter_variant": "path_specific_v2"},
        },
        path,
    )


def _fixture_stage44_artifacts(tmp_path: Path, seeds: tuple[int, ...] = (202, 303)) -> tuple[Path, Path, Path]:
    stage44a = tmp_path / "stage44a"
    stage44b6 = tmp_path / "stage44b6"
    dataset_manifest = [
        {
            "task_name": task_name,
            "dataset_fields_structured_ok": True,
            "structure_sources": ["dataset_fields"],
            "cache_path": f"cache/{task_name}.jsonl",
        }
        for task_name in EXTERNAL_TASKS
    ]
    stage44c = tmp_path / "stage44c" / "summary.json"
    _write_json(
        stage44c,
        {
            "stage": "stage44c_real_task_runner",
            "seeds": list(seeds),
            "stage_gates": {
                "local_stage44a_passed": True,
                "external_stage44b6_passed": True,
                "external_dataset_fields_cache": True,
                "stage44c_passed": True,
            },
            "passes_stage_gate": True,
            "external": {"dataset_fields_cache_manifest": dataset_manifest},
        },
    )
    for seed in seeds:
        _write_checkpoint(
            stage44a / f"seed_{seed}" / "checkpoints" / "full_hidden_alignment" / f"stage44a_full_hidden_alignment_seed_{seed}.pt",
            seed,
        )
        _write_checkpoint(
            stage44b6 / f"seed_{seed}" / "checkpoints" / "combined" / f"stage44b4_combined_seed_{seed}.pt",
            seed,
        )
        for task_name in EXTERNAL_TASKS:
            _write_checkpoint(
                stage44b6 / f"seed_{seed}" / "checkpoints" / task_name / f"stage44b4_{task_name}_seed_{seed}.pt",
                seed,
            )
    return stage44c, stage44a, stage44b6


def test_stage45_builds_package_manifest_and_eval_harness(tmp_path: Path) -> None:
    stage44c, stage44a, stage44b6 = _fixture_stage44_artifacts(tmp_path)
    output_dir = tmp_path / "package"
    manifest = build_stage45_adapter_package(
        output_dir=output_dir,
        stage44c_summary_path=stage44c,
        stage44a_root=stage44a,
        stage44b6_root=stage44b6,
        seeds=(202, 303),
    )
    assert manifest.architecture == "dual_16_24_path_specific_v2"
    assert len(manifest.checkpoints) == 2 * (2 + len(EXTERNAL_TASKS))
    assert all(ref.excludes_qwen_weights for ref in manifest.checkpoints)
    loaded = load_stage45_package_manifest(output_dir / "package_manifest.json")
    validation = validate_stage45_package(loaded)
    assert validation["passes_stage_gate"]
    summary = run_stage45_eval_harness(manifest_path=output_dir / "package_manifest.json", output_dir=output_dir)
    assert summary["passes_stage_gate"]
    assert summary["contract_smoke_response"]["status"] == "requires_runtime_backend"


def test_stage45_rejects_non_dataset_fields_cache(tmp_path: Path) -> None:
    stage44c, stage44a, stage44b6 = _fixture_stage44_artifacts(tmp_path, seeds=(202,))
    payload = json.loads(stage44c.read_text(encoding="utf-8"))
    payload["external"]["dataset_fields_cache_manifest"][0]["structure_sources"] = ["parsed_text_tail"]
    _write_json(stage44c, payload)
    with pytest.raises(ValueError, match="dataset-fields"):
        build_stage45_adapter_package(
            output_dir=tmp_path / "package",
            stage44c_summary_path=stage44c,
            stage44a_root=stage44a,
            stage44b6_root=stage44b6,
            seeds=(202,),
        )


def test_stage45_detects_qwen_weight_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"qwen_state_dict": {"embed_tokens.weight": torch.zeros(1)}}, checkpoint)
    assert checkpoint_contains_qwen_weights(checkpoint)


def test_stage45_inference_contract_validates_seed_and_options(tmp_path: Path) -> None:
    stage44c, stage44a, stage44b6 = _fixture_stage44_artifacts(tmp_path, seeds=(202,))
    manifest = build_stage45_adapter_package(
        output_dir=tmp_path / "package",
        stage44c_summary_path=stage44c,
        stage44a_root=stage44a,
        stage44b6_root=stage44b6,
        seeds=(202,),
        include_task_checkpoints=False,
    )
    request = Stage45InferenceRequest(
        text="Evaluate the supplied record.",
        memory_items=("memory evidence",),
        rule_items=("rule evidence",),
        state_values=(1.0, 0.0, 0.5),
        answer_options=("yes", "no"),
        seed=202,
    )
    validate_inference_request(request, manifest)
    response = Stage45InferenceEngine(manifest).predict(request)
    assert response.status == "requires_runtime_backend"
    with pytest.raises(ValueError, match="not packaged"):
        validate_inference_request(
            Stage45InferenceRequest(
                text="Evaluate",
                memory_items=(),
                rule_items=(),
                state_values=(1.0, 0.0, 0.5),
                answer_options=("yes", "no"),
                seed=303,
            ),
            manifest,
        )
