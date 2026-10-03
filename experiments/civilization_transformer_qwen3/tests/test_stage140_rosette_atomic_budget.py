from experiments.civilization_transformer_qwen3.analysis.stage140_rosette_atomic_budget import run_stage140_rosette_atomic_budget_smoke


def test_stage140_routes_rosette_groups_atomically_under_budget(tmp_path) -> None:
    summary = run_stage140_rosette_atomic_budget_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
