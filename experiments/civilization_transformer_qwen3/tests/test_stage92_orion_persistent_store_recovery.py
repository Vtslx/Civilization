from experiments.civilization_transformer_qwen3.analysis.stage92_orion_persistent_store_recovery import run_stage92_orion_recovery_smoke


def test_stage92_corruption_is_isolated(tmp_path) -> None:
    assert run_stage92_orion_recovery_smoke(output_dir=tmp_path)["passes_stage_gate"]
