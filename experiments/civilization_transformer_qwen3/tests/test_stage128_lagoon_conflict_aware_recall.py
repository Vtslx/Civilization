from experiments.civilization_transformer_qwen3.analysis.stage128_lagoon_conflict_aware_recall import run_stage128_lagoon_conflict_aware_recall_smoke


def test_stage128_returns_conflict_aware_schema_recall_without_mutation(tmp_path) -> None:
    summary = run_stage128_lagoon_conflict_aware_recall_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
