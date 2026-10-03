from experiments.civilization_transformer_qwen3.analysis.stage137_eagle_skill_conflict_guard import run_stage137_eagle_skill_conflict_smoke


def test_stage137_marks_divergent_failed_strategy_without_overwrite(tmp_path) -> None:
    summary = run_stage137_eagle_skill_conflict_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
