from civilization.engine.stages.stage136_eagle_skill_retrieval import run_stage136_eagle_skill_retrieval_smoke


def test_stage136_retrieves_eagle_skill_with_trace_provenance(tmp_path) -> None:
    summary = run_stage136_eagle_skill_retrieval_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
