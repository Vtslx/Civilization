from experiments.civilization_transformer_qwen3.analysis.stage125_lagoon_schema_retrieval import run_stage125_lagoon_schema_retrieval_smoke


def test_stage125_preserves_lagoon_schema_provenance_boundary(tmp_path) -> None:
    summary = run_stage125_lagoon_schema_retrieval_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
