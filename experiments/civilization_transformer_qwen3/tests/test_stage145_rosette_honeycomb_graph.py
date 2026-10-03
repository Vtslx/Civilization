from experiments.civilization_transformer_qwen3.analysis.stage145_rosette_honeycomb_graph import run_stage145_rosette_honeycomb_graph_smoke


def test_stage145_builds_multi_scale_multi_adjacency_honeycomb_graph(tmp_path) -> None:
    summary = run_stage145_rosette_honeycomb_graph_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
