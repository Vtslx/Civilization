from civilization.engine.stages.stage93_orion_global_store_lifecycle import run_stage93_orion_lifecycle_smoke


def test_stage93_explicit_global_store_lifecycle(tmp_path) -> None:
    assert run_stage93_orion_lifecycle_smoke(output_dir=tmp_path)["passes_stage_gate"]
