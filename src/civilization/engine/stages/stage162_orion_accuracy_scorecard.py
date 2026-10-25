"""Strict scorecards for full LongMemEval answerer runs.

This module scores the benchmark runner's saved JSON. It does not regenerate
answers or judgments, and it keeps per-question text out of the scorecard.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage162_orion_accuracy_scorecard")
_PROTOCOL_FIELDS = (
    "benchmark",
    "answerer_model",
    "judge_model",
    "provider",
    "top_k",
    "top_k_cutoffs",
    "dataset_sha256",
)


class ScorecardError(ValueError):
    """A benchmark result is incomplete or cannot support a comparable score."""


def build_longmemeval_scorecard(
    results_file: str | Path,
    *,
    baseline_file: str | Path | None = None,
    dataset_file: str | Path | None = None,
    expected_questions: int = 500,
    cutoff: str = "top_20",
    target_accuracy: float = 0.97,
) -> dict[str, Any]:
    """Validate and summarize a full LongMemEval answerer result JSON.

    A 97% target on the standard 500-question set requires at least 485 fully
    correct answers. Any baseline comparison is rejected unless the question
    IDs, categories, model, provider, retrieval count, and cutoff list match.
    """

    if expected_questions < 1:
        raise ValueError("expected_questions must be positive")
    if not math.isfinite(target_accuracy) or not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    if cutoff not in {"top_20", "top_200"}:
        raise ScorecardError("supported scorecard cutoffs are top_20 and diagnostic-only top_200")
    if baseline_file is not None and dataset_file is None:
        raise ScorecardError("a dataset file is required for comparable retrieval diagnostics")
    dataset_by_id, dataset_sha256 = _load_dataset(dataset_file) if dataset_file is not None else (None, None)
    candidate, candidate_rows = _load_and_aggregate(
        results_file,
        expected_questions=expected_questions,
        cutoff=cutoff,
        target_accuracy=target_accuracy,
    )
    candidate["dataset_sha256"] = dataset_sha256
    candidate["evidence_retrieval"] = _evidence_retrieval_metrics(candidate_rows, dataset_by_id, dataset_sha256)
    candidate["release_gate_applies"] = cutoff == "top_20"
    if cutoff != "top_20":
        candidate["target_met"] = None
    report: dict[str, Any] = {"candidate": candidate, "cutoff": cutoff, "target_accuracy": target_accuracy}
    if baseline_file is not None:
        baseline, baseline_rows = _load_and_aggregate(
            baseline_file,
            expected_questions=expected_questions,
            cutoff=cutoff,
            target_accuracy=target_accuracy,
        )
        baseline["dataset_sha256"] = dataset_sha256
        baseline["evidence_retrieval"] = _evidence_retrieval_metrics(baseline_rows, dataset_by_id, dataset_sha256)
        baseline["release_gate_applies"] = cutoff == "top_20"
        if cutoff != "top_20":
            baseline["target_met"] = None
        _require_comparable(candidate, baseline, candidate_rows, baseline_rows)
        candidate_score = candidate["score_sum"]
        baseline_score = baseline["score_sum"]
        question_count = candidate["question_count"]
        candidate_types = candidate["question_types"]
        baseline_types = baseline["question_types"]
        both_correct = candidate_only = baseline_only = both_wrong = 0
        for question_id in candidate_rows:
            new_correct = candidate_rows[question_id]["score"] >= 1.0
            old_correct = baseline_rows[question_id]["score"] >= 1.0
            if new_correct and old_correct:
                both_correct += 1
            elif new_correct:
                candidate_only += 1
            elif old_correct:
                baseline_only += 1
            else:
                both_wrong += 1
        report["baseline"] = baseline
        report["comparison"] = {
            "accuracy_delta_percentage_points": round(
                (candidate_score - baseline_score) / question_count * 100,
                2,
            ),
            "paired_outcomes": {
                "both_correct": both_correct,
                "candidate_only_correct": candidate_only,
                "baseline_only_correct": baseline_only,
                "both_wrong": both_wrong,
            },
            "question_type_deltas_percentage_points": {
                name: round(
                    (candidate_types[name]["accuracy"] - baseline_types[name]["accuracy"]) * 100,
                    2,
                )
                for name in sorted(candidate_types)
            },
            "answer_source_recall_at_20_delta_percentage_points": round(
                (
                    candidate["evidence_retrieval"]["answer_source_recall_at_20"]
                    - baseline["evidence_retrieval"]["answer_source_recall_at_20"]
                )
                * 100,
                2,
            ),
            "answer_source_recall_at_200_delta_percentage_points": round(
                (
                    candidate["evidence_retrieval"]["answer_source_recall_at_200"]
                    - baseline["evidence_retrieval"]["answer_source_recall_at_200"]
                )
                * 100,
                2,
            ),
            "answer_source_pair_recall_at_20_delta_percentage_points": (
                round(
                    (
                        candidate["evidence_retrieval"]["answer_source_pair_recall_at_20"]
                        - baseline["evidence_retrieval"]["answer_source_pair_recall_at_20"]
                    )
                    * 100,
                    2,
                )
                if candidate["evidence_retrieval"]["answer_source_pair_recall_at_20"] is not None
                and baseline["evidence_retrieval"]["answer_source_pair_recall_at_20"] is not None
                else None
            ),
            "answer_source_pair_recall_at_200_delta_percentage_points": (
                round(
                    (
                        candidate["evidence_retrieval"]["answer_source_pair_recall_at_200"]
                        - baseline["evidence_retrieval"]["answer_source_pair_recall_at_200"]
                    )
                    * 100,
                    2,
                )
                if candidate["evidence_retrieval"]["answer_source_pair_recall_at_200"] is not None
                and baseline["evidence_retrieval"]["answer_source_pair_recall_at_200"] is not None
                else None
            ),
        }
    return report


def write_scorecard(report: dict[str, Any], output_file: str | Path) -> Path:
    """Atomically write a scorecard without copying benchmark question contents."""

    output = Path(output_file).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _load_and_aggregate(
    results_file: str | Path,
    *,
    expected_questions: int,
    cutoff: str,
    target_accuracy: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(results_file).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScorecardError(f"could not read benchmark result JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ScorecardError("benchmark result root must be an object")
    metadata = payload.get("metadata")
    rows = payload.get("evaluations")
    if not isinstance(metadata, dict) or metadata.get("benchmark") != "longmemeval":
        raise ScorecardError("result metadata must identify the longmemeval benchmark")
    if metadata.get("mode", "answerer") != "answerer":
        raise ScorecardError("the 97% accuracy gate requires answerer mode")
    if metadata.get("all_questions") is not True or metadata.get("total_questions") != expected_questions:
        raise ScorecardError(f"result metadata does not certify all {expected_questions} questions")
    if type(metadata.get("top_k")) is not int or metadata["top_k"] != 200:
        raise ScorecardError("the comparable LongMemEval protocol requires top_k=200")
    for model_field in ("answerer_model", "judge_model"):
        if not isinstance(metadata.get(model_field), str) or not metadata[model_field].strip():
            raise ScorecardError(f"result metadata must identify {model_field}")
    cutoffs = metadata.get("top_k_cutoffs")
    if not isinstance(cutoffs, list) or cutoff not in cutoffs:
        raise ScorecardError(f"result was not scored at cutoff {cutoff}")
    if not isinstance(rows, list) or len(rows) != expected_questions:
        raise ScorecardError(f"expected exactly {expected_questions} evaluation rows")

    scores_by_type: dict[str, list[float]] = {}
    latencies: list[float] = []
    returned_counts: list[int] = []
    row_index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ScorecardError("every evaluation row must be an object")
        question_id = row.get("question_id")
        question_type = row.get("question_type")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ScorecardError("every evaluation row must have a question_id")
        if question_id in row_index:
            raise ScorecardError(f"duplicate question_id: {question_id}")
        if not isinstance(question_type, str) or not question_type.strip():
            raise ScorecardError(f"question {question_id} has no question_type")
        cutoff_results = row.get("cutoff_results")
        cutoff_result = cutoff_results.get(cutoff) if isinstance(cutoff_results, dict) else None
        if not isinstance(cutoff_result, dict):
            raise ScorecardError(f"question {question_id} has no {cutoff} result")
        score = cutoff_result.get("score")
        if isinstance(score, bool) or not isinstance(score, (float, int)):
            raise ScorecardError(f"question {question_id} has an invalid score")
        score_value = float(score)
        if not math.isfinite(score_value) or not 0 <= score_value <= 1:
            raise ScorecardError(f"question {question_id} score must be between 0 and 1")
        scores_by_type.setdefault(question_type, []).append(score_value)
        retrieval = row.get("retrieval", {})
        retrieval = retrieval if isinstance(retrieval, dict) else {}
        row_index[question_id] = {
            "question_type": question_type,
            "score": score_value,
            "retrieval": retrieval,
            "answer_session_ids": row.get("answer_session_ids", []),
            "is_abstention": bool(row.get("is_abstention", False)),
        }
        if isinstance(retrieval, dict):
            latency = retrieval.get("search_latency_ms")
            if isinstance(latency, (int, float)) and math.isfinite(float(latency)) and latency >= 0:
                latencies.append(float(latency))
            result_count = retrieval.get("total_results")
            if isinstance(result_count, int) and result_count >= 0:
                returned_counts.append(result_count)

    score_sum = sum(row["score"] for row in row_index.values())
    correct_count = sum(row["score"] >= 1.0 for row in row_index.values())
    required_correct = math.ceil(expected_questions * target_accuracy - 1e-12)
    question_types = {
        name: {
            "questions": len(scores),
            "correct": sum(score >= 1.0 for score in scores),
            "score_sum": round(sum(scores), 4),
            "accuracy": round(sum(scores) / len(scores), 6),
        }
        for name, scores in sorted(scores_by_type.items())
    }
    report = {
        "benchmark": "longmemeval",
        "result_sha256": _sha256_file(path),
        "answerer_model": _safe_metadata_text(metadata.get("answerer_model")),
        "judge_model": _safe_metadata_text(metadata.get("judge_model")),
        "provider": _safe_metadata_text(metadata.get("provider")),
        "top_k": metadata.get("top_k"),
        "top_k_cutoffs": cutoffs,
        "question_count": len(row_index),
        "correct_count": correct_count,
        "score_sum": round(score_sum, 4),
        "accuracy": round(score_sum / len(row_index), 6),
        "strict_correct_accuracy": round(correct_count / len(row_index), 6),
        "required_correct_for_target": required_correct,
        "target_met": correct_count >= required_correct,
        "abstention": {
            "questions": sum(bool(row.get("is_abstention", False)) for row in rows),
            "correct": sum(
                bool(row.get("is_abstention", False))
                and row_index[str(row["question_id"])]["score"] >= 1.0
                for row in rows
            ),
        },
        "api_error_count": sum(
            bool(
                row.get("error")
                or (row.get("retrieval", {}).get("error") if isinstance(row.get("retrieval"), dict) else None)
            )
            for row in rows
        ),
        "question_types": question_types,
        "retrieval": {
            "mean_results_returned": round(statistics.fmean(returned_counts), 2) if returned_counts else None,
            "mean_search_latency_ms": round(statistics.fmean(latencies), 2) if latencies else None,
            "p95_search_latency_ms": _percentile(latencies, 0.95),
        },
    }
    adapter_features = {
        key: metadata[key]
        for key in (
            "count_inventory_enabled",
            "answer_verification_enabled",
            "preference_verification_enabled",
            "count_completeness_repair_enabled",
            "temporal_event_resolution_enabled",
            "explicit_total_deduplication_enabled",
            "entity_count_prompt_enabled",
            "user_authored_count_evidence_validation_enabled",
        )
        if isinstance(metadata.get(key), bool)
    }
    count_inventory_policy = metadata.get("count_inventory_policy")
    if count_inventory_policy in {"all", "skip_multi_session"}:
        adapter_features["count_inventory_policy"] = count_inventory_policy
    if adapter_features:
        report["adapter_features"] = adapter_features
    result_provenance: dict[str, Any] = {}
    if isinstance(metadata.get("postprocessor_replay_only"), bool):
        result_provenance["postprocessor_replay_only"] = metadata["postprocessor_replay_only"]
    source_run = metadata.get("postprocessor_source_run")
    if isinstance(source_run, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", source_run):
        result_provenance["postprocessor_source_run"] = source_run
    if isinstance(metadata.get("subset_replay_only"), bool):
        result_provenance["subset_replay_only"] = metadata["subset_replay_only"]
    replay_question_type = metadata.get("subset_replay_question_type")
    if replay_question_type in {
        "temporal-reasoning",
        "multi-session",
        "knowledge-update",
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
    }:
        result_provenance["subset_replay_question_type"] = replay_question_type
    subset_source_run = metadata.get("subset_replay_source_run")
    if isinstance(subset_source_run, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", subset_source_run):
        result_provenance["subset_replay_source_run"] = subset_source_run
    subset_scope = metadata.get("subset_replay_scope")
    if isinstance(subset_scope, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", subset_scope):
        result_provenance["subset_replay_scope"] = subset_scope
    frozen_base_run = metadata.get("frozen_base_source_run")
    if isinstance(frozen_base_run, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", frozen_base_run):
        result_provenance["frozen_base_source_run"] = frozen_base_run
    if result_provenance:
        report["result_provenance"] = result_provenance
    return report, row_index


def _require_comparable(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    candidate_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
) -> None:
    mismatches = [field for field in _PROTOCOL_FIELDS if candidate.get(field) != baseline.get(field)]
    if mismatches:
        raise ScorecardError(f"candidate and baseline protocol differs: {', '.join(mismatches)}")
    if candidate_rows.keys() != baseline_rows.keys():
        raise ScorecardError("candidate and baseline question IDs differ")
    if any(candidate_rows[key]["question_type"] != baseline_rows[key]["question_type"] for key in candidate_rows):
        raise ScorecardError("candidate and baseline question categories differ")


def _safe_metadata_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _load_dataset(dataset_file: str | Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = Path(dataset_file).expanduser().resolve(strict=True)
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScorecardError(f"could not read LongMemEval source dataset: {error}") from error
    if not isinstance(dataset, list):
        raise ScorecardError("LongMemEval dataset root must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in dataset:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise ScorecardError("every dataset row must have a question_id")
        question_id = row["question_id"]
        if question_id in indexed:
            raise ScorecardError(f"duplicate dataset question_id: {question_id}")
        indexed[question_id] = row
    return indexed, _sha256_file(path)


def _evidence_retrieval_metrics(
    result_rows: dict[str, dict[str, Any]],
    dataset_rows: dict[str, dict[str, Any]] | None,
    dataset_sha256: str | None,
) -> dict[str, Any]:
    if dataset_rows is None:
        return {
            "available": False,
            "dataset_sha256": None,
            "answer_source_questions": None,
            "answer_source_in_top_20": None,
            "answer_source_recall_at_20": None,
            "answer_source_in_top_200": None,
            "answer_source_recall_at_200": None,
            "answer_correct_with_source_in_top_20": None,
            "answer_errors_with_source_in_top_20": None,
            "answer_accuracy_given_source_in_top_20": None,
            "mean_reciprocal_rank_at_200": None,
            "answer_source_pair_metadata_available": False,
            "answer_source_pair_questions": None,
            "answer_source_pairs": None,
            "answer_source_pairs_in_top_20": None,
            "answer_source_pair_recall_at_20": None,
            "answer_source_pairs_in_top_200": None,
            "answer_source_pair_recall_at_200": None,
            "questions_with_all_answer_pairs_in_top_20": None,
            "questions_with_all_answer_pairs_in_top_200": None,
            "answers_correct_with_all_answer_pairs_in_top_20": None,
            "answers_wrong_with_all_answer_pairs_in_top_20": None,
            "question_types": {},
        }
    if result_rows.keys() != dataset_rows.keys():
        raise ScorecardError("result and source dataset question IDs differ")

    type_counts: dict[str, dict[str, float | int]] = {}
    source_questions = hit20 = hit200 = 0
    source_hit_correct20 = source_hit_error20 = 0
    reciprocal_rank_sum = 0.0
    pair_metadata_available = any(
        isinstance(result.get("retrieval"), dict)
        and any(
            isinstance(retrieved, dict)
            and "source_session_id" in retrieved
            and "source_pair_index" in retrieved
            for retrieved in result["retrieval"].get("search_results", [])
        )
        for result in result_rows.values()
    )
    answer_pair_questions = answer_pairs = pair_hit20 = pair_hit200 = 0
    questions_all_pairs20 = questions_all_pairs200 = 0
    correct_all_pairs20 = error_all_pairs20 = 0
    for question_id, result in result_rows.items():
        source_row = dataset_rows[question_id]
        answer_session_ids = result["answer_session_ids"]
        if result["is_abstention"] or not isinstance(answer_session_ids, list) or not answer_session_ids:
            continue
        expected_chunks = _answer_source_memory_chunks(source_row, set(answer_session_ids))
        retrieval_results = result["retrieval"].get("search_results")
        if not isinstance(retrieval_results, list):
            raise ScorecardError(f"question {question_id} has no retrieval result list")
        source_questions += 1
        category = result["question_type"]
        metrics = type_counts.setdefault(
            category,
            {
                "questions": 0,
                "hit_at_20": 0,
                "hit_at_200": 0,
                "hit_correct_at_20": 0,
                "hit_error_at_20": 0,
                "rr_sum": 0.0,
                "answer_pair_questions": 0,
                "answer_pairs": 0,
                "pair_hit_at_20": 0,
                "pair_hit_at_200": 0,
                "all_pairs_at_20": 0,
                "all_pairs_at_200": 0,
                "correct_all_pairs_at_20": 0,
                "error_all_pairs_at_20": 0,
            },
        )
        metrics["questions"] += 1
        required_pairs = _answer_source_memory_pairs(source_row, set(answer_session_ids))
        if required_pairs:
            answer_pair_questions += 1
            answer_pairs += len(required_pairs)
            metrics["answer_pair_questions"] += 1
            metrics["answer_pairs"] += len(required_pairs)
            if pair_metadata_available:
                present20 = _retrieved_pair_keys(retrieval_results[:20])
                present200 = _retrieved_pair_keys(retrieval_results[:200])
                hit_pairs20 = len(required_pairs & present20)
                hit_pairs200 = len(required_pairs & present200)
                pair_hit20 += hit_pairs20
                pair_hit200 += hit_pairs200
                metrics["pair_hit_at_20"] += hit_pairs20
                metrics["pair_hit_at_200"] += hit_pairs200
                if required_pairs.issubset(present20):
                    questions_all_pairs20 += 1
                    metrics["all_pairs_at_20"] += 1
                    if result["score"] >= 1.0:
                        correct_all_pairs20 += 1
                        metrics["correct_all_pairs_at_20"] += 1
                    else:
                        error_all_pairs20 += 1
                        metrics["error_all_pairs_at_20"] += 1
                if required_pairs.issubset(present200):
                    questions_all_pairs200 += 1
                    metrics["all_pairs_at_200"] += 1
        source_rank = None
        for rank, retrieved in enumerate(retrieval_results[:200], start=1):
            memory = retrieved.get("memory") if isinstance(retrieved, dict) else None
            if isinstance(memory, str) and _normalize_memory_text(memory) in expected_chunks:
                source_rank = rank
                break
        if source_rank is not None:
            reciprocal_rank_sum += 1 / source_rank
            metrics["rr_sum"] += 1 / source_rank
        if source_rank is not None and source_rank <= 20:
            hit20 += 1
            metrics["hit_at_20"] += 1
            if result["score"] >= 1.0:
                source_hit_correct20 += 1
                metrics["hit_correct_at_20"] += 1
            else:
                source_hit_error20 += 1
                metrics["hit_error_at_20"] += 1
        if source_rank is not None:
            hit200 += 1
            metrics["hit_at_200"] += 1

    per_type = {
        name: {
            "answer_source_questions": int(metrics["questions"]),
            "answer_source_in_top_20": int(metrics["hit_at_20"]),
            "answer_source_recall_at_20": round(int(metrics["hit_at_20"]) / int(metrics["questions"]), 6),
            "answer_source_in_top_200": int(metrics["hit_at_200"]),
            "answer_source_recall_at_200": round(int(metrics["hit_at_200"]) / int(metrics["questions"]), 6),
            "answer_correct_with_source_in_top_20": int(metrics["hit_correct_at_20"]),
            "answer_errors_with_source_in_top_20": int(metrics["hit_error_at_20"]),
            "answer_accuracy_given_source_in_top_20": (
                round(int(metrics["hit_correct_at_20"]) / int(metrics["hit_at_20"]), 6)
                if metrics["hit_at_20"]
                else None
            ),
            "mean_reciprocal_rank_at_200": round(float(metrics["rr_sum"]) / int(metrics["questions"]), 6),
            "answer_source_pair_questions": int(metrics["answer_pair_questions"]),
            "answer_source_pairs": int(metrics["answer_pairs"]),
            "answer_source_pairs_in_top_20": int(metrics["pair_hit_at_20"]) if pair_metadata_available else None,
            "answer_source_pair_recall_at_20": (
                round(int(metrics["pair_hit_at_20"]) / int(metrics["answer_pairs"]), 6)
                if pair_metadata_available and metrics["answer_pairs"]
                else None
            ),
            "answer_source_pairs_in_top_200": int(metrics["pair_hit_at_200"]) if pair_metadata_available else None,
            "answer_source_pair_recall_at_200": (
                round(int(metrics["pair_hit_at_200"]) / int(metrics["answer_pairs"]), 6)
                if pair_metadata_available and metrics["answer_pairs"]
                else None
            ),
            "questions_with_all_answer_pairs_in_top_20": int(metrics["all_pairs_at_20"]) if pair_metadata_available else None,
            "questions_with_all_answer_pairs_in_top_200": int(metrics["all_pairs_at_200"]) if pair_metadata_available else None,
            "answers_correct_with_all_answer_pairs_in_top_20": int(metrics["correct_all_pairs_at_20"]) if pair_metadata_available else None,
            "answers_wrong_with_all_answer_pairs_in_top_20": int(metrics["error_all_pairs_at_20"]) if pair_metadata_available else None,
        }
        for name, metrics in sorted(type_counts.items())
    }
    return {
        "available": True,
        "dataset_sha256": dataset_sha256,
        "answer_source_questions": source_questions,
        "answer_source_in_top_20": hit20,
        "answer_source_recall_at_20": round(hit20 / source_questions, 6) if source_questions else None,
        "answer_source_in_top_200": hit200,
        "answer_source_recall_at_200": round(hit200 / source_questions, 6) if source_questions else None,
        "answer_correct_with_source_in_top_20": source_hit_correct20,
        "answer_errors_with_source_in_top_20": source_hit_error20,
        "answer_accuracy_given_source_in_top_20": (
            round(source_hit_correct20 / hit20, 6) if hit20 else None
        ),
        "mean_reciprocal_rank_at_200": round(reciprocal_rank_sum / source_questions, 6) if source_questions else None,
        "answer_source_pair_metadata_available": pair_metadata_available,
        "answer_source_pair_questions": answer_pair_questions if pair_metadata_available else None,
        "answer_source_pairs": answer_pairs if pair_metadata_available else None,
        "answer_source_pairs_in_top_20": pair_hit20 if pair_metadata_available else None,
        "answer_source_pair_recall_at_20": round(pair_hit20 / answer_pairs, 6) if pair_metadata_available and answer_pairs else None,
        "answer_source_pairs_in_top_200": pair_hit200 if pair_metadata_available else None,
        "answer_source_pair_recall_at_200": round(pair_hit200 / answer_pairs, 6) if pair_metadata_available and answer_pairs else None,
        "questions_with_all_answer_pairs_in_top_20": questions_all_pairs20 if pair_metadata_available else None,
        "questions_with_all_answer_pairs_in_top_200": questions_all_pairs200 if pair_metadata_available else None,
        "answers_correct_with_all_answer_pairs_in_top_20": correct_all_pairs20 if pair_metadata_available else None,
        "answers_wrong_with_all_answer_pairs_in_top_20": error_all_pairs20 if pair_metadata_available else None,
        "question_types": per_type,
    }


def _answer_source_memory_chunks(question: dict[str, Any], answer_session_ids: set[Any]) -> set[str]:
    session_ids = question.get("haystack_session_ids")
    dates = question.get("haystack_dates")
    sessions = question.get("haystack_sessions")
    if not isinstance(session_ids, list) or not isinstance(dates, list) or not isinstance(sessions, list):
        raise ScorecardError("source dataset row is missing conversation sessions or dates")
    if not (len(session_ids) == len(dates) == len(sessions)):
        raise ScorecardError("source dataset session IDs, dates, and conversations have different lengths")

    chunks: set[str] = set()
    for session_id, date_text, session in zip(session_ids, dates, sessions, strict=True):
        if session_id not in answer_session_ids or not isinstance(session, list):
            continue
        date_label = _observation_date(date_text)
        for offset in range(0, len(session), 2):
            pair = session[offset : offset + 2]
            if not pair or any(not str(message.get("content", "")).strip() for message in pair if isinstance(message, dict)):
                continue
            if any(not isinstance(message, dict) for message in pair):
                continue
            content = "\n".join(
                f"{message.get('role', 'unknown')}: {message.get('content', '')}"
                for message in pair
            ).strip()
            if date_label:
                content = f"Conversation date: {date_label}\n{content}"
            if content:
                chunks.add(_normalize_memory_text(content))
    return chunks


def _answer_source_memory_pairs(question: dict[str, Any], answer_session_ids: set[Any]) -> set[tuple[str, int]]:
    session_ids = question.get("haystack_session_ids")
    sessions = question.get("haystack_sessions")
    if not isinstance(session_ids, list) or not isinstance(sessions, list):
        raise ScorecardError("source dataset row is missing conversation sessions")
    if len(session_ids) != len(sessions):
        raise ScorecardError("source dataset session IDs and conversations have different lengths")

    answer_pairs: set[tuple[str, int]] = set()
    for session_id, session in zip(session_ids, sessions, strict=True):
        if session_id not in answer_session_ids or not isinstance(session, list):
            continue
        for offset in range(0, len(session), 2):
            pair = session[offset : offset + 2]
            if any(isinstance(message, dict) and bool(message.get("has_answer")) for message in pair):
                answer_pairs.add((str(session_id), offset // 2))
    return answer_pairs


def _retrieved_pair_keys(retrieval_results: list[Any]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for item in retrieval_results:
        if not isinstance(item, dict):
            continue
        session_id = item.get("source_session_id")
        pair_index = item.get("source_pair_index")
        if isinstance(pair_index, str) and pair_index.isdigit():
            pair_index = int(pair_index)
        if isinstance(session_id, str) and isinstance(pair_index, int):
            pairs.add((session_id, pair_index))
    return pairs


def _observation_date(date_text: Any) -> str | None:
    if not isinstance(date_text, str):
        return None
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_text).strip()
    try:
        parsed = datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return datetime.fromtimestamp(parsed.timestamp(), tz=timezone.utc).date().isoformat()


def _normalize_memory_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 2)
