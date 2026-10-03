from __future__ import annotations

import json

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage152_helix_feedback_signal import FeedbackKind
from experiments.civilization_transformer_qwen3.analysis.stage159_helix_calibration_gate import (
    CalibrationConfig,
    run_stage159_helix_calibration_gate_smoke,
)


def test_stage159_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage159_helix_calibration_gate"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "calibration_report.json").exists()


def test_stage159_held_out_covers_all_feedback_types(tmp_path) -> None:
    run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    report = json.loads((tmp_path / "calibration_report.json").read_text(encoding="utf-8"))
    required = {kind.value for kind in FeedbackKind}
    assert required.issubset(set(report["feedback_types_covered"]))


def test_stage159_degradation_within_tolerance(tmp_path) -> None:
    summary = run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path, tolerance=0.0)
    assert summary["stage_gates"]["degradation_within_tolerance"] is True


def test_stage159_learned_has_intended_effect(tmp_path) -> None:
    run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["stage_gates"]["learned_has_intended_effect"] is True
    report = summary["report"]
    # Learned ranks the boosted cell strictly higher (lower rank number) than static.
    assert report["metrics"]["learned"]["boosted_cell_rank"] < report["metrics"]["static"]["boosted_cell_rank"]


def test_stage159_negative_results_preserved(tmp_path) -> None:
    run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["stage_gates"]["negative_results_preserved"] is True
    report = summary["report"]
    # Coverage is identical across configs: learning changes ranking, not coverage.
    covs = {report["metrics"][c]["retrieval_coverage"] for c in ("no_learning", "static", "learned")}
    assert len(covs) == 1
    travs = {report["metrics"][c]["traversal_coverage"] for c in ("no_learning", "static", "learned")}
    assert len(travs) == 1


def test_stage159_no_accuracy_claim_and_controlled_fixture_label(tmp_path) -> None:
    summary = run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    assert summary["stage_gates"]["no_accuracy_claim"] is True
    assert summary["stage_gates"]["numbers_labeled_controlled_fixture"] is True


def test_stage159_weights_bounded_in_learned_config(tmp_path) -> None:
    run_stage159_helix_calibration_gate_smoke(output_dir=tmp_path)
    report = json.loads((tmp_path / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["learned"]["weights_bounded"] is True
