from experiments.civilization_transformer_qwen3.analysis.stage143_rosette_snapshot_recovery import run_stage143_rosette_snapshot_recovery_smoke


def test_stage143_recovers_valid_snapshot_and_rejects_corruption(tmp_path) -> None:
    summary = run_stage143_rosette_snapshot_recovery_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "valid-packet.json").exists()
    assert (tmp_path / "tampered-packet.json").exists()
    assert (tmp_path / "dangling-packet.json").exists()
    assert (tmp_path / "summary.json").exists()
