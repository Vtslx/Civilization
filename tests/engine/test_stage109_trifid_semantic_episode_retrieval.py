from civilization.engine.stages.stage109_trifid_semantic_episode_retrieval import (
    run_stage109_trifid_semantic_episode_retrieval_smoke,
)


def test_stage109_recalls_only_explicit_semantic_episode_provenance(tmp_path) -> None:
    summary = run_stage109_trifid_semantic_episode_retrieval_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
