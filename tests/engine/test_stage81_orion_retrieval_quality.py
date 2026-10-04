from civilization.engine.stages.stage81_orion_retrieval_quality import run_stage81_orion_retrieval_quality


def test_stage81_global_opt_in_improves_cross_session_hit(tmp_path) -> None:
    summary = run_stage81_orion_retrieval_quality(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["global_opt_in_hit_rate"] > summary["session_only_hit_rate"]
