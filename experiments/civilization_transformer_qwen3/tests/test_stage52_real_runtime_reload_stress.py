from __future__ import annotations

from experiments.civilization_transformer_qwen3.analysis.stage52_real_runtime_reload_stress import (
    Stage52ReloadStressConfig,
    run_stage52_fake_reload_stress,
)


def test_stage52_fake_reload_stress_passes_and_writes_artifacts(tmp_path) -> None:
    summary = run_stage52_fake_reload_stress(
        output_dir=tmp_path,
        config=Stage52ReloadStressConfig(reload_rounds=2, controls=("full", "adapter_disabled")),
    )
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["drain_rejects_prediction"]
    assert summary["stage_gates"]["resume_predictions_ok"]
    assert len(summary["events"]) == 2
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "resource_usage.json").exists()
    assert (tmp_path / "reload_events.jsonl").exists()


def test_stage52_reload_versions_increase(tmp_path) -> None:
    summary = run_stage52_fake_reload_stress(
        output_dir=tmp_path,
        config=Stage52ReloadStressConfig(reload_rounds=3, controls=("full",)),
    )
    versions = [event["reload"]["event"]["new_version"] for event in summary["events"]]
    assert versions == sorted(versions)
    assert len(set(versions)) == 3


def test_stage52_drain_blocks_prediction_rows(tmp_path) -> None:
    summary = run_stage52_fake_reload_stress(
        output_dir=tmp_path,
        config=Stage52ReloadStressConfig(reload_rounds=1, controls=("full",)),
    )
    draining_rows = summary["events"][0]["draining_predict"]["rows"]
    assert draining_rows[0]["status"] == "error"
    assert draining_rows[0]["error_type"] == "ServiceDraining"
