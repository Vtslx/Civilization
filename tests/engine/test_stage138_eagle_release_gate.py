from civilization.engine.stages.stage138_eagle_release_gate import run_stage138_eagle_release_gate


def test_stage138_closes_eagle_with_lagoon_regression(tmp_path) -> None:
    summary = run_stage138_eagle_release_gate(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert set(summary["eagle_stage_summaries"]) == {f"stage{number}" for number in range(133, 138)}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "lagoon-stage132" / "summary.json").exists()
