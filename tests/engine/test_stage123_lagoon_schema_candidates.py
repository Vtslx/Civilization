from civilization.engine.stages.stage123_lagoon_schema_candidates import run_stage123_lagoon_schema_candidate_smoke


def test_stage123_builds_schema_candidate_without_writing_semantic_memory(tmp_path) -> None:
    assert run_stage123_lagoon_schema_candidate_smoke(output_dir=tmp_path)["passes_stage_gate"]
