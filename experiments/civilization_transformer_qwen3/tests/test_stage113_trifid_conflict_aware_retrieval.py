from experiments.civilization_transformer_qwen3.analysis.stage113_trifid_conflict_aware_retrieval import (
    run_stage113_trifid_conflict_aware_retrieval_smoke,
)


def test_stage113_exposes_approved_conflicts_with_semantic_recall(tmp_path) -> None:
    summary = run_stage113_trifid_conflict_aware_retrieval_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
