from civilization.engine.stages.stage111_trifid_conflict_review import (
    run_stage111_trifid_conflict_review_smoke,
)


def test_stage111_builds_read_only_conflict_review(tmp_path) -> None:
    summary = run_stage111_trifid_conflict_review_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
