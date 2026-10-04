from civilization.engine.stages.stage110_trifid_semantic_conflict_guard import (
    run_stage110_trifid_semantic_conflict_smoke,
)


def test_stage110_marks_outcome_conflicts_without_overwriting_memory(tmp_path) -> None:
    summary = run_stage110_trifid_semantic_conflict_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
