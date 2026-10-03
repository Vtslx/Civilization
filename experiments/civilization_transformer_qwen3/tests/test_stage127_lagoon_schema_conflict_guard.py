from experiments.civilization_transformer_qwen3.analysis.stage127_lagoon_schema_conflict_guard import run_stage127_lagoon_conflict_smoke


def test_stage127_marks_only_matching_context_conflicts_and_preserves_memory(tmp_path) -> None:
    summary = run_stage127_lagoon_conflict_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
