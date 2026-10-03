from experiments.civilization_transformer_qwen3.analysis.stage105_trifid_episode_replay import (
    run_stage105_trifid_replay_smoke,
)


def test_stage105_replays_completed_episode_idempotently(tmp_path) -> None:
    summary = run_stage105_trifid_replay_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
