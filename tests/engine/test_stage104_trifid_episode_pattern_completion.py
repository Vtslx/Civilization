from civilization.engine.stages.stage104_trifid_episode_pattern_completion import (
    run_stage104_trifid_pattern_completion_smoke,
)


def test_stage104_completes_time_safe_episode_and_traces_read(tmp_path) -> None:
    summary = run_stage104_trifid_pattern_completion_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
