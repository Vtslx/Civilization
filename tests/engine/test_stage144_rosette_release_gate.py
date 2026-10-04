from civilization.engine.stages.stage144_rosette_release_gate import run_stage144_rosette_release_gate


def test_stage144_closes_rosette_with_full_version_regression(tmp_path) -> None:
    summary = run_stage144_rosette_release_gate(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert set(summary["rosette_stage_summaries"]) == {f"stage{number}" for number in range(139, 144)}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "lagoon-stage132" / "summary.json").exists()
    assert (tmp_path / "eagle-stage138" / "summary.json").exists()
