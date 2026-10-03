from experiments.civilization_transformer_qwen3.analysis.stage133_eagle_task_trace import run_stage133_eagle_task_trace_smoke


def test_stage133_normalizes_eagle_execution_traces(tmp_path) -> None:
    summary = run_stage133_eagle_task_trace_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "task_traces.json").exists()
    assert (tmp_path / "summary.json").exists()
