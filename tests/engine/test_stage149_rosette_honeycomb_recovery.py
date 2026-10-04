from civilization.engine.stages.stage149_rosette_honeycomb_recovery import run_stage149_rosette_honeycomb_recovery_smoke


def test_stage149_recovers_only_live_valid_honeycomb_graphs(tmp_path) -> None:
    summary = run_stage149_rosette_honeycomb_recovery_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "valid-honeycomb.json").exists()
    assert (tmp_path / "dangling-honeycomb.json").exists()
    assert (tmp_path / "summary.json").exists()
