from experiments.civilization_transformer_qwen3.analysis.stage96_orion_global_store_doctor import run_stage96_orion_doctor_smoke


def test_stage96_restore_is_explicit(tmp_path) -> None:
    assert run_stage96_orion_doctor_smoke(output_dir=tmp_path)["passes_stage_gate"]
