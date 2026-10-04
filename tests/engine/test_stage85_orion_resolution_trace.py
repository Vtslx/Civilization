from civilization.engine.stages.stage85_orion_resolution_trace import run_stage85_orion_resolution_trace_smoke


def test_stage85_exposes_resolution_without_hiding_loser(tmp_path) -> None:
    summary = run_stage85_orion_resolution_trace_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
