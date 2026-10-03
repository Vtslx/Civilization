from experiments.civilization_transformer_qwen3.analysis.stage146_rosette_honeycomb_traversal import run_stage146_rosette_honeycomb_traversal_smoke


def test_stage146_traverses_multiple_adjacencies_deterministically(tmp_path) -> None:
    summary = run_stage146_rosette_honeycomb_traversal_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
