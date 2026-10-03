from experiments.civilization_transformer_qwen3.analysis.stage108_trifid_approved_consolidation import (
    run_stage108_trifid_approved_consolidation_smoke,
)


def test_stage108_requires_approval_and_consolidates_candidate_once(tmp_path) -> None:
    summary = run_stage108_trifid_approved_consolidation_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
