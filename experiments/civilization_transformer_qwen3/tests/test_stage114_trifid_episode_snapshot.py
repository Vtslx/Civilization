from experiments.civilization_transformer_qwen3.analysis.stage114_trifid_episode_snapshot import (
    run_stage114_trifid_episode_snapshot_smoke,
)


def test_stage114_restores_valid_episode_frames_with_monotonic_ids(tmp_path) -> None:
    summary = run_stage114_trifid_episode_snapshot_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
