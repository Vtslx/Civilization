from experiments.civilization_transformer_qwen3.analysis.stage129_lagoon_conflict_review import run_stage129_lagoon_conflict_review_smoke


def test_stage129_builds_read_only_provenance_verified_conflict_review(tmp_path) -> None:
    summary = run_stage129_lagoon_conflict_review_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
