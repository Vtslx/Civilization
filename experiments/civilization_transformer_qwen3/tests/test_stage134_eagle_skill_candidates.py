from experiments.civilization_transformer_qwen3.analysis.stage134_eagle_skill_candidates import run_stage134_eagle_skill_candidate_smoke


def test_stage134_builds_evidence_only_skill_candidates(tmp_path) -> None:
    summary = run_stage134_eagle_skill_candidate_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "skill_candidates.json").exists()
    assert (tmp_path / "summary.json").exists()
