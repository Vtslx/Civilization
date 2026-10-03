from experiments.civilization_transformer_qwen3.analysis.stage150_rosette_release_gate import run_stage150_rosette_release_gate


def test_stage150_closes_complete_lagoon_eagle_rosette_chain(tmp_path) -> None:
    summary = run_stage150_rosette_release_gate(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert summary["version_chain"] == ["v0.00.04", "v0.00.05", "v0.00.06"]
    assert set(summary["honeycomb_stage_summaries"]) == {f"stage{number}" for number in range(145, 150)}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "context-stage144" / "summary.json").exists()
