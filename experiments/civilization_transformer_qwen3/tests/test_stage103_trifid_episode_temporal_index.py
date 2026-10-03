from experiments.civilization_transformer_qwen3.analysis.stage103_trifid_episode_temporal_index import (
    run_stage103_trifid_temporal_index_smoke,
)


def test_stage103_indexes_episode_time_and_rejects_discontinuous_sources(tmp_path) -> None:
    summary = run_stage103_trifid_temporal_index_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
