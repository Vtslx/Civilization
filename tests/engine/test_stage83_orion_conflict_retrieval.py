from civilization.engine.stages.stage83_orion_conflict_retrieval import run_stage83_orion_conflict_smoke


def test_stage83_conflicts_require_resolution(tmp_path) -> None:
    summary = run_stage83_orion_conflict_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert all(row["requires_resolution"] for row in summary["rows"])
