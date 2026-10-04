from civilization.engine.stages.stage99_orion_retention_regression import run_stage99_orion_retention_regression


def test_stage99_retention_persists_across_lifecycle(tmp_path) -> None:
    assert run_stage99_orion_retention_regression(output_dir=tmp_path)["passes_stage_gate"]
