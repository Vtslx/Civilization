from __future__ import annotations

import json
from pathlib import Path

import pytest

from civilization.cli import main
from civilization.engine.stages.stage162_orion_accuracy_scorecard import (
    ScorecardError,
    build_longmemeval_scorecard,
    write_scorecard,
)


def _result(path: Path, scores: list[float], *, metadata_overrides: dict | None = None, question_ids: list[str] | None = None) -> Path:
    ids = question_ids or [f"q-{index}" for index in range(len(scores))]
    metadata = {
        "benchmark": "longmemeval",
        "mode": "answerer",
        "all_questions": True,
        "total_questions": len(scores),
        "answerer_model": "gpt-6-luna",
        "judge_model": "gpt-6-luna",
        "provider": "openai",
        "top_k": 200,
        "top_k_cutoffs": ["top_20"],
        "project_name": "scorecard-test",
        "run_id": path.stem,
    }
    metadata.update(metadata_overrides or {})
    cutoff = metadata["top_k_cutoffs"][0]
    types = ("single-session-user", "temporal-reasoning", "multi-session")
    evaluations = []
    for index, (question_id, score) in enumerate(zip(ids, scores, strict=True)):
        memory = f"Conversation date: 2023-05-01\nuser: source fact for {question_id}\nassistant: measurement is 10"
        evaluations.append(
            {
                "question_id": question_id,
                "question_type": types[index % len(types)],
                "question": "private benchmark question must not be copied into scorecards",
                "answer_session_ids": [f"source-{index}"],
                "retrieval": {
                    "search_latency_ms": 20 + index,
                    "total_results": 200,
                    "search_results": [{"memory": memory, "score": 1.0, "id": f"cell-{index}"}],
                },
                "cutoff_results": {cutoff: {"score": score}},
            }
        )
    path.write_text(
        json.dumps({"metadata": metadata, "metrics_by_cutoff": {}, "evaluations": evaluations}),
        encoding="utf-8",
    )
    return path


def _dataset(path: Path, question_ids: list[str]) -> Path:
    rows = []
    for index, question_id in enumerate(question_ids):
        session_id = f"source-{index}"
        rows.append(
            {
                "question_id": question_id,
                "question": "private benchmark question",
                "answer_session_ids": [session_id],
                "haystack_session_ids": [session_id],
                "haystack_dates": ["2023/05/01 (Mon) 21:05"],
                "haystack_sessions": [[
                    {"role": "user", "content": f"source fact for {question_id}", "has_answer": True},
                    {"role": "assistant", "content": "measurement is 10"},
                ]],
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_stage162_computes_strict_97_percent_release_gate(tmp_path: Path) -> None:
    result = _result(tmp_path / "candidate.json", [1.0, 0.0])

    scorecard = build_longmemeval_scorecard(result, expected_questions=2, target_accuracy=0.97)

    candidate = scorecard["candidate"]
    assert candidate["correct_count"] == 1
    assert candidate["accuracy"] == 0.5
    assert candidate["required_correct_for_target"] == 2
    assert candidate["target_met"] is False
    assert candidate["question_types"]["single-session-user"]["correct"] == 1


def test_stage162_preserves_adapter_feature_flags(tmp_path: Path) -> None:
    result = _result(
        tmp_path / "features.json",
        [1.0, 0.0],
        metadata_overrides={
            "count_inventory_enabled": False,
            "answer_verification_enabled": True,
            "preference_verification_enabled": True,
            "count_inventory_policy": "skip_multi_session",
            "temporal_event_resolution_enabled": True,
            "postprocessor_replay_only": True,
            "postprocessor_source_run": "Stage184",
            "subset_replay_only": True,
            "subset_replay_question_type": "single-session-preference",
            "subset_replay_source_run": "Stage188",
        },
    )

    scorecard = build_longmemeval_scorecard(result, expected_questions=2)

    assert scorecard["candidate"]["adapter_features"] == {
        "count_inventory_enabled": False,
        "answer_verification_enabled": True,
        "preference_verification_enabled": True,
        "count_inventory_policy": "skip_multi_session",
        "temporal_event_resolution_enabled": True,
    }
    assert scorecard["candidate"]["result_provenance"] == {
        "postprocessor_replay_only": True,
        "postprocessor_source_run": "Stage184",
        "subset_replay_only": True,
        "subset_replay_question_type": "single-session-preference",
        "subset_replay_source_run": "Stage188",
    }


def test_stage162_marks_top200_as_diagnostic_not_release_gate(tmp_path: Path) -> None:
    result = _result(
        tmp_path / "top200.json",
        [1.0, 1.0],
        metadata_overrides={"top_k_cutoffs": ["top_200"]},
    )

    scorecard = build_longmemeval_scorecard(result, expected_questions=2, cutoff="top_200")

    assert scorecard["candidate"]["correct_count"] == 2
    assert scorecard["candidate"]["release_gate_applies"] is False
    assert scorecard["candidate"]["target_met"] is None


def test_stage162_pair_compares_only_identical_protocols(tmp_path: Path) -> None:
    baseline = _result(tmp_path / "baseline.json", [1.0, 0.0, 0.0])
    candidate = _result(tmp_path / "candidate.json", [1.0, 1.0, 0.0])
    dataset = _dataset(tmp_path / "dataset.json", ["q-0", "q-1", "q-2"])

    report = build_longmemeval_scorecard(
        candidate,
        baseline_file=baseline,
        dataset_file=dataset,
        expected_questions=3,
        target_accuracy=0.6,
    )

    assert report["comparison"]["accuracy_delta_percentage_points"] == pytest.approx(33.33, abs=0.01)
    assert report["comparison"]["paired_outcomes"] == {
        "both_correct": 1,
        "candidate_only_correct": 1,
        "baseline_only_correct": 0,
        "both_wrong": 1,
    }
    assert report["candidate"]["evidence_retrieval"]["answer_source_recall_at_20"] == 1.0
    assert report["candidate"]["evidence_retrieval"]["mean_reciprocal_rank_at_200"] == 1.0
    assert report["candidate"]["evidence_retrieval"]["answer_accuracy_given_source_in_top_20"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "metadata_overrides, expected_error",
    [
        ({"mode": "retrieval"}, "answerer mode"),
        ({"all_questions": False}, "all 2 questions"),
        ({"top_k": 20}, "top_k=200"),
        ({"top_k_cutoffs": ["top_10"]}, "cutoff top_20"),
        ({"answerer_model": ""}, "answerer_model"),
    ],
)
def test_stage162_rejects_noncomparable_or_partial_results(
    tmp_path: Path, metadata_overrides: dict, expected_error: str
) -> None:
    result = _result(tmp_path / "result.json", [1.0, 1.0], metadata_overrides=metadata_overrides)

    with pytest.raises(ScorecardError, match=expected_error):
        build_longmemeval_scorecard(result, expected_questions=2)


def test_stage162_rejects_duplicate_ids_and_missing_scores(tmp_path: Path) -> None:
    duplicates = _result(tmp_path / "duplicates.json", [1.0, 1.0], question_ids=["same", "same"])
    with pytest.raises(ScorecardError, match="duplicate question_id"):
        build_longmemeval_scorecard(duplicates, expected_questions=2)

    missing = _result(tmp_path / "missing.json", [1.0, 0.0])
    payload = json.loads(missing.read_text(encoding="utf-8"))
    del payload["evaluations"][1]["cutoff_results"]["top_20"]["score"]
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScorecardError, match="invalid score"):
        build_longmemeval_scorecard(missing, expected_questions=2)


def test_stage162_scorecard_output_contains_no_question_text(tmp_path: Path) -> None:
    result = _result(tmp_path / "result.json", [1.0, 1.0])
    report = build_longmemeval_scorecard(result, expected_questions=2)
    output = write_scorecard(report, tmp_path / "scorecard.json")
    written = output.read_text(encoding="utf-8")

    assert "private benchmark question" not in written
    assert json.loads(written)["candidate"]["question_count"] == 2


def test_stage162_scorecard_cli_writes_full_report(tmp_path: Path, capsys) -> None:
    result = _result(tmp_path / "result.json", [1.0, 0.0])
    dataset = _dataset(tmp_path / "dataset.json", ["q-0", "q-1"])
    output = tmp_path / "release-scorecard.json"

    exit_code = main(
        [
            "scorecard",
            "--results",
            str(result),
            "--dataset-file",
            str(dataset),
            "--expected-questions",
            "2",
            "--target-accuracy",
            "0.5",
            "--output",
            str(output),
        ]
    )

    console = capsys.readouterr().out
    assert exit_code == 0
    assert "scorecard saved" in console
    assert json.loads(output.read_text(encoding="utf-8"))["candidate"]["target_met"] is True
