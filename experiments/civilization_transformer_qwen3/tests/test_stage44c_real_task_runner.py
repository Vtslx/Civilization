from __future__ import annotations

import json
from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.stage44c_real_task_runner import (
    _gate_summary,
    _stage44b6_multiseed_summary,
    run_qwen3_stage44c_real_task_runner,
)


def test_stage44c_gate_requires_local_external_and_dataset_fields() -> None:
    local = {"passes_stage_gate": True}
    external = {
        "passes_stage_gate": True,
        "dataset_fields_cache_manifest": [
            {
                "task_name": "glue_rte",
                "structure_sources": ["dataset_fields"],
                "dataset_fields_structured_ok": True,
            }
        ],
    }
    assert _gate_summary(local, external)["stage44c_passed"]
    external["dataset_fields_cache_manifest"][0]["structure_sources"] = ["parsed_text_tail"]
    assert not _gate_summary(local, external)["stage44c_passed"]


def test_stage44c_external_multiseed_summary_requires_all_pass() -> None:
    rows = [
        {"seed": 202, "passes_stage_gate": True},
        {"seed": 303, "passes_stage_gate": False},
    ]
    summary = _stage44b6_multiseed_summary(rows, (202, 303))
    assert summary["completed_seed_count"] == 2
    assert summary["passed_seed_count"] == 1
    assert not summary["passes_stage_gate"]


def test_stage44c_reuse_existing_reads_summaries(monkeypatch, tmp_path: Path) -> None:
    local_dir = tmp_path / "stage44a"
    external_dir = tmp_path / "stage44b6"
    local_dir.mkdir()
    (local_dir / "summary.json").write_text(
        json.dumps({"passes_stage_gate": True, "fixed_centroid_mean": 1.0}),
        encoding="utf-8",
    )
    for seed in (202, 303):
        seed_dir = external_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "summary.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "passes_stage_gate": True,
                    "dataset_fields_cache_manifest": [
                        {
                            "task_name": "boolq",
                            "structure_sources": ["dataset_fields"],
                            "dataset_fields_structured_ok": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    import experiments.civilization_transformer_qwen3.analysis.stage44c_real_task_runner as module

    monkeypatch.setattr(module, "DEFAULT_STAGE44A_DIR", local_dir)
    monkeypatch.setattr(module, "DEFAULT_STAGE44B6_DIR", external_dir)
    summary = run_qwen3_stage44c_real_task_runner(
        output_dir=tmp_path / "out",
        seeds=(202, 303),
        reuse_existing=True,
    )
    assert summary["passes_stage_gate"]
    assert summary["mode"] == "reuse_existing"
    assert (tmp_path / "out" / "summary.json").exists()
