from experiments.civilization_transformer_qwen3.analysis.stage116_trifid_paired_checkpoint import (
    run_stage116_trifid_paired_checkpoint_smoke,
)


def test_stage116_restores_verified_paired_checkpoint(tmp_path) -> None:
    assert run_stage116_trifid_paired_checkpoint_smoke(output_dir=tmp_path)["passes_stage_gate"]
