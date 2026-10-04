from civilization.engine.stages.stage106_trifid_replay_scheduler import (
    run_stage106_trifid_replay_scheduler_smoke,
)


def test_stage106_schedules_bounded_priority_replay(tmp_path) -> None:
    summary = run_stage106_trifid_replay_scheduler_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
