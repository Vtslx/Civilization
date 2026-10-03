from experiments.civilization_transformer_qwen3.analysis.stage135_eagle_approved_skill import run_stage135_eagle_approved_skill_smoke


def test_stage135_consolidates_approved_eagle_skill_with_provenance(tmp_path) -> None:
    summary = run_stage135_eagle_approved_skill_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
