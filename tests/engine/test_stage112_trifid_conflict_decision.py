from civilization.engine.stages.stage112_trifid_conflict_decision import (
    run_stage112_trifid_conflict_decision_smoke,
)


def test_stage112_records_approved_preserve_both_decision(tmp_path) -> None:
    summary = run_stage112_trifid_conflict_decision_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
