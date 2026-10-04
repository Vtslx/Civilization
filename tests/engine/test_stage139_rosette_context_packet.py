from civilization.engine.stages.stage139_rosette_context_packet import run_stage139_rosette_context_packet_smoke


def test_stage139_assembles_typed_rosette_context_packet(tmp_path) -> None:
    summary = run_stage139_rosette_context_packet_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
