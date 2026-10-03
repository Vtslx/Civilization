from experiments.civilization_transformer_qwen3.analysis.stage142_rosette_packet_snapshot import run_stage142_rosette_packet_snapshot_smoke


def test_stage142_saves_canonical_atomic_packet_snapshot(tmp_path) -> None:
    summary = run_stage142_rosette_packet_snapshot_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "packet.json").exists()
    assert (tmp_path / "packet-repeat.json").exists()
    assert (tmp_path / "summary.json").exists()
