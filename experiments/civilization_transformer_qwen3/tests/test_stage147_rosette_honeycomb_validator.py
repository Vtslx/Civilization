from experiments.civilization_transformer_qwen3.analysis.stage147_rosette_honeycomb_validator import run_stage147_rosette_honeycomb_validator_smoke


def test_stage147_fails_closed_on_invalid_honeycomb_topology(tmp_path) -> None:
    summary = run_stage147_rosette_honeycomb_validator_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
