from experiments.civilization_transformer_qwen3.analysis.stage82_orion_global_maintenance import run_stage82_orion_global_maintenance_smoke


def test_stage82_conflict_and_stale_cells_are_auditable(tmp_path) -> None:
    summary = run_stage82_orion_global_maintenance_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["maintenance"]["conflict_pairs"]
