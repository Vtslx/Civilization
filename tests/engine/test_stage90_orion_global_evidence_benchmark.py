from civilization.engine.stages.stage90_orion_global_evidence_benchmark import run_stage90_orion_global_evidence_benchmark


def test_stage90_control_conditions_are_locked(tmp_path) -> None:
    assert run_stage90_orion_global_evidence_benchmark(output_dir=tmp_path)["passes_stage_gate"]
