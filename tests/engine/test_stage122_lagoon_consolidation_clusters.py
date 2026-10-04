from civilization.engine.stages.stage122_lagoon_consolidation_clusters import run_stage122_lagoon_cluster_smoke


def test_stage122_clusters_only_continuous_replayed_episodes(tmp_path) -> None:
    assert run_stage122_lagoon_cluster_smoke(output_dir=tmp_path)["passes_stage_gate"]
