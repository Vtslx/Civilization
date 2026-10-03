from experiments.civilization_transformer_qwen3.analysis.stage130_lagoon_conflict_decision import run_stage130_lagoon_conflict_decision_smoke


def test_stage130_records_approved_preserve_both_decision(tmp_path) -> None:
    summary = run_stage130_lagoon_conflict_decision_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
