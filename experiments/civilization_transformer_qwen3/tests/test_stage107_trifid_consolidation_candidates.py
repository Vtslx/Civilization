from experiments.civilization_transformer_qwen3.analysis.stage107_trifid_consolidation_candidates import (
    run_stage107_trifid_consolidation_candidate_smoke,
)


def test_stage107_builds_replayed_evidence_candidates_without_consolidating(tmp_path) -> None:
    summary = run_stage107_trifid_consolidation_candidate_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
