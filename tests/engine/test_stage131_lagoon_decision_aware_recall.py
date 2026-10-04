from civilization.engine.stages.stage131_lagoon_decision_aware_recall import run_stage131_lagoon_decision_aware_recall_smoke


def test_stage131_recall_surfaces_approved_preserve_both_pair(tmp_path) -> None:
    summary = run_stage131_lagoon_decision_aware_recall_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
