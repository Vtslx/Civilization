from civilization.engine.stages.stage141_rosette_packet_validator import run_stage141_rosette_packet_validator_smoke


def test_stage141_validates_packet_and_fails_closed(tmp_path) -> None:
    summary = run_stage141_rosette_packet_validator_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
