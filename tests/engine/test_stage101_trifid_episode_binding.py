from civilization.engine.stages.stage101_trifid_episode_binding import run_stage101_trifid_binding_smoke


def test_stage101_binds_and_recalls_episode(tmp_path) -> None:
    assert run_stage101_trifid_binding_smoke(output_dir=tmp_path)["passes_stage_gate"]
