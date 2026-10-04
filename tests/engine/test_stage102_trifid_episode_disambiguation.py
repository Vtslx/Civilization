from civilization.engine.stages.stage102_trifid_episode_disambiguation import run_stage102_trifid_disambiguation_smoke


def test_stage102_disambiguates_and_rejects_mixed_cues(tmp_path) -> None:
    assert run_stage102_trifid_disambiguation_smoke(output_dir=tmp_path)["passes_stage_gate"]
