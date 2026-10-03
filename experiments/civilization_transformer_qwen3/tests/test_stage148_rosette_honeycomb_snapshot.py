from experiments.civilization_transformer_qwen3.analysis.stage148_rosette_honeycomb_snapshot import run_stage148_rosette_honeycomb_snapshot_smoke


def test_stage148_saves_only_validated_honeycomb_graphs(tmp_path) -> None:
    summary = run_stage148_rosette_honeycomb_snapshot_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "honeycomb.json").exists()
    assert (tmp_path / "summary.json").exists()
