from civilization.engine.stages.stage84_orion_conflict_resolution import run_stage84_orion_resolution_smoke


def test_stage84_resolution_preserves_conflict_evidence(tmp_path) -> None:
    summary = run_stage84_orion_resolution_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["resolution"]["winner_cell_id"]
