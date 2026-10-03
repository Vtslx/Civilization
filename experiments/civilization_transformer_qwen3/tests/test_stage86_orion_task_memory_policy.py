from experiments.civilization_transformer_qwen3.analysis.stage86_orion_task_memory_policy import run_stage86_orion_task_policy_smoke


def test_stage86_only_admits_winner_global_when_enabled(tmp_path) -> None:
    assert run_stage86_orion_task_policy_smoke(output_dir=tmp_path)["passes_stage_gate"]
