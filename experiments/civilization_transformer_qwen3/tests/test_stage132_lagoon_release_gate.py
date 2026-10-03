from experiments.civilization_transformer_qwen3.analysis.stage132_lagoon_release_gate import run_stage132_lagoon_release_gate


def test_stage132_closes_lagoon_with_complete_release_gate(tmp_path) -> None:
    summary = run_stage132_lagoon_release_gate(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert set(summary["stage_summaries"]) == {f"stage{number}" for number in range(122, 132)}
    assert (tmp_path / "summary.json").exists()
    assert all((tmp_path / stage / "summary.json").exists() for stage in summary["stage_summaries"])
