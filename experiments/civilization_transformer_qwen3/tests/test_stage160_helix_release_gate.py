from __future__ import annotations

from experiments.civilization_transformer_qwen3.analysis.stage160_helix_release_gate import (
    VERSION_CHAIN,
    run_stage160_helix_release_gate,
)


def test_stage160_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage160_helix_release_gate(output_dir=tmp_path)

    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage160_helix_release_gate"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()


def test_stage160_version_chain_includes_helix() -> None:
    summary = run_stage160_helix_release_gate()
    assert summary["version_chain"] == ["v0.00.04", "v0.00.05", "v0.00.06", "v0.00.07"]
    assert VERSION_CHAIN == ["v0.00.04", "v0.00.05", "v0.00.06", "v0.00.07"]


def test_stage160_upstream_chain_green() -> None:
    summary = run_stage160_helix_release_gate()
    gates = summary["stage_gates"]
    assert gates["lagoon_v0_00_04_green"] is True
    assert gates["eagle_v0_00_05_green"] is True
    assert gates["rosette_v0_00_06_green"] is True


def test_stage160_seven_category_gates_true() -> None:
    summary = run_stage160_helix_release_gate()
    gates = summary["stage_gates"]
    seven = (
        "helix_path_weight",
        "helix_feedback",
        "helix_update_rule",
        "helix_gated_mutation",
        "helix_snapshot",
        "helix_recovery",
        "helix_calibration",
    )
    for category in seven:
        assert gates[category] is True, f"{category} gate failed"


def test_stage160_all_helix_stages_pass_and_set_complete() -> None:
    summary = run_stage160_helix_release_gate()
    gates = summary["stage_gates"]
    assert gates["all_helix_stages_pass"] is True
    assert gates["helix_stage_set_complete"] is True
    # Every Helix stage 151-159 appears in the summaries.
    assert set(summary["helix_stage_summaries"]) == {f"stage{n}" for n in range(151, 160)}


def test_stage160_weight_determinism() -> None:
    summary = run_stage160_helix_release_gate()
    assert summary["stage_gates"]["weight_determinism"] is True
    assert summary["weight_determinism_sha256"]
    assert len(summary["weight_determinism_sha256"]) == 64  # SHA-256 hex


def test_stage160_qwen_frozen_and_artifacts() -> None:
    summary = run_stage160_helix_release_gate()
    assert summary["stage_gates"]["qwen_frozen"] is True
    assert summary["stage_gates"]["all_release_artifacts_present"] is True
