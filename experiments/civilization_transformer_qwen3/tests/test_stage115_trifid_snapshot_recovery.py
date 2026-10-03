from experiments.civilization_transformer_qwen3.analysis.stage115_trifid_snapshot_recovery import (
    run_stage115_trifid_snapshot_recovery_smoke,
)


def test_stage115_recovers_valid_snapshots_and_fails_closed_for_invalid_ones(tmp_path) -> None:
    summary = run_stage115_trifid_snapshot_recovery_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
