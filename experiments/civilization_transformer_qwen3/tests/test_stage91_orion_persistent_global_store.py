from experiments.civilization_transformer_qwen3.analysis.stage91_orion_persistent_global_store import run_stage91_orion_persistence_smoke


def test_stage91_persists_global_cells_links_and_metadata(tmp_path) -> None:
    assert run_stage91_orion_persistence_smoke(output_dir=tmp_path)["passes_stage_gate"]
