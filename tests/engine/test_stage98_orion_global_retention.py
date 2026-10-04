from civilization.engine.stages.stage98_orion_global_retention import run_stage98_orion_retention_smoke


def test_stage98_retires_without_deleting(tmp_path) -> None:
    assert run_stage98_orion_retention_smoke(output_dir=tmp_path)["passes_stage_gate"]
