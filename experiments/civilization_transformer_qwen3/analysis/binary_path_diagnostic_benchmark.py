from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import psutil
import torch

from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .binary_path_diagnostic_data import (
    BINARY_BASE_DIAGNOSTIC_MODES,
    BINARY_DIAGNOSTIC_MODES,
    BinaryDiagnosticPair,
    assert_binary_pairs_valid,
    build_combined_binary_path_diagnostic_pairs,
    build_binary_path_diagnostic_pairs,
    split_binary_pairs,
)
from .binary_path_diagnostic_training import run_binary_path_diagnostic_training
from .multilayer_adapter_benchmark import _collect_multilayer_vectors
from ..backend import Qwen3Backend


PROBE_LAYER = 28
EVALUATION_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_state_path",
    "no_rule_path",
    "empty_context",
    "wrong_context",
    "counterfactual_context",
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _samples_for_pairs(pairs: list[BinaryDiagnosticPair]):
    return [sample for pair in pairs for sample in (pair.full_sample, pair.counterfactual_sample)]


def _counterfactual_samples_for_pairs(pairs: list[BinaryDiagnosticPair]):
    return [sample for pair in pairs for sample in (pair.counterfactual_sample, pair.full_sample)]


def _pair_targets(pairs: list[BinaryDiagnosticPair]) -> list[int]:
    result: list[int] = []
    for pair in pairs:
        result.extend((pair.expected_full_option_id, pair.expected_counterfactual_option_id))
    return result


def _centroid_accuracy(
    train_vectors: np.ndarray,
    train_targets: list[int],
    test_vectors: np.ndarray,
    test_targets: list[int],
) -> tuple[float, list[int]]:
    centroids = []
    for option_id in (0, 1):
        rows = train_vectors[np.array(train_targets) == option_id]
        if len(rows) == 0:
            raise ValueError("missing train centroid for binary option")
        centroids.append(rows.mean(axis=0))
    centroid_matrix = np.stack(centroids)
    normalized_test = test_vectors / np.clip(np.linalg.norm(test_vectors, axis=1, keepdims=True), 1e-8, None)
    normalized_centroids = centroid_matrix / np.clip(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1e-8, None)
    predictions = np.argmax(normalized_test @ normalized_centroids.T, axis=1).tolist()
    correct = [prediction == target for prediction, target in zip(predictions, test_targets, strict=True)]
    return sum(correct) / len(correct) if correct else 0.0, predictions


def _load_adapter_checkpoint(model, checkpoint_path: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.adapters.load_state_dict(payload["adapter_state_dict"])


def _evaluate_stage(
    *,
    backend: Qwen3Backend,
    model,
    mode: str,
    stage: str,
    train_pairs: list[BinaryDiagnosticPair],
    test_pairs: list[BinaryDiagnosticPair],
    max_length: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples = _samples_for_pairs(train_pairs)
    test_samples = _samples_for_pairs(test_pairs)
    train_targets = _pair_targets(train_pairs)
    test_targets = _pair_targets(test_pairs)
    full_train_vectors, _train_traces, train_failures = _collect_multilayer_vectors(
        backend,
        model,
        train_samples,
        "full",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=True,
    )
    disabled_test_vectors, _disabled_traces, disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        test_samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=True,
    )
    fixed_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failures = [
        {"mode": mode, "stage": stage, **row}
        for row in (*train_failures, *disabled_failures)
    ]
    full_accuracy = 0.0
    wrong_accuracy = 0.0
    for eval_mode in EVALUATION_MODES:
        eval_samples = _counterfactual_samples_for_pairs(test_pairs) if eval_mode == "counterfactual_context" else test_samples
        collect_mode = "full" if eval_mode == "counterfactual_context" else eval_mode
        vectors, traces, vector_failures = _collect_multilayer_vectors(
            backend,
            model,
            eval_samples,
            collect_mode,
            max_length,
            batch_size,
            (PROBE_LAYER,),
            fail_on_truncation=True,
        )
        failures.extend({"mode": mode, "stage": stage, **row} for row in vector_failures)
        trace_rows.extend({"diagnostic_mode": mode, "stage": stage, "eval_mode": eval_mode, **row} for row in traces)
        readout_vectors = vectors[PROBE_LAYER] - disabled_test_vectors[PROBE_LAYER]
        option_vectors = build_answer_option_vectors(backend, test_pairs[0].answer_options)
        answer = score_answer_options(readout_vectors, test_samples, option_vectors, test_pairs[0].answer_options)
        if eval_mode == "full":
            full_accuracy = answer.accuracy
        if eval_mode == "wrong_context":
            wrong_accuracy = answer.accuracy
        answer_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                "accuracy": answer.accuracy,
                "macro_accuracy": answer.macro_accuracy,
                "mean_margin": answer.mean_margin,
            }
        )
        fixed_accuracy, fixed_predictions = _centroid_accuracy(
            full_train_vectors[PROBE_LAYER],
            train_targets,
            vectors[PROBE_LAYER],
            test_targets,
        )
        fixed_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                "accuracy": fixed_accuracy,
            }
        )
        if eval_mode == "full":
            for pair_index, pair in enumerate(test_pairs):
                first = pair_index * 2
                second = first + 1
                pair_rows.append(
                    {
                        "diagnostic_mode": mode,
                        "stage": stage,
                        "pair_type": pair.pair_type,
                        "pair_id": pair.pair_id,
                        "full_correct": answer.rows[first]["correct"],
                        "counterfactual_correct": answer.rows[second]["correct"],
                        "pair_success": bool(answer.rows[first]["correct"] and answer.rows[second]["correct"]),
                        "full_fixed_correct": fixed_predictions[first] == pair.expected_full_option_id,
                        "counterfactual_fixed_correct": fixed_predictions[second] == pair.expected_counterfactual_option_id,
                    }
                )
    answer_rows.append(
        {
            "diagnostic_mode": mode,
            "stage": stage,
            "eval_mode": "wrong_context_drop",
            "accuracy": full_accuracy - wrong_accuracy,
            "macro_accuracy": "",
            "mean_margin": "",
        }
    )
    return answer_rows, fixed_rows, pair_rows, trace_rows, failures


def run_qwen3_binary_path_diagnostic(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/binary_path_diagnostic",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    pairs_per_mode: int = 120,
    train_pairs: int = 80,
    held_out_pairs: int = 40,
    stage_a_steps: int = 80,
    stage_b_steps: int = 40,
    gradient_accumulation: int = 4,
    max_length: int = 64,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    adapter_variant: str = "baseline_v1",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    all_pairs = build_binary_path_diagnostic_pairs(pairs_per_mode=pairs_per_mode, seed=seed, max_seq_len=max_length)
    all_pairs["combined_binary_diagnostic"] = build_combined_binary_path_diagnostic_pairs(
        all_pairs,
        pairs_per_mode,
    )
    assert_binary_pairs_valid([
        pair
        for mode, pairs in all_pairs.items()
        if mode in BINARY_BASE_DIAGNOSTIC_MODES
        for pair in pairs
    ])

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    binary_pair_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, dict[str, str]] = {}

    for mode in BINARY_DIAGNOSTIC_MODES:
        print(f"binary_path_diagnostic_mode_start mode={mode}", flush=True)
        mode_started = time.perf_counter()
        mode_train, mode_test = split_binary_pairs(all_pairs[mode], train_pairs, held_out_pairs)
        model, _heads, training = run_binary_path_diagnostic_training(
            backend=backend,
            train_pairs=mode_train,
            output_dir=output_path / "checkpoints",
            mode=mode,
            seed=seed,
            stage_a_steps=stage_a_steps,
            stage_b_steps=stage_b_steps,
            gradient_accumulation=gradient_accumulation,
            max_length=max_length,
            adapter_variant=adapter_variant,
        )
        training_row = asdict(training)
        losses = training_row.pop("losses")
        training_rows.append(training_row)
        loss_rows.extend({"diagnostic_mode": mode, **row} for row in losses)
        checkpoint_paths[mode] = {
            "stage_a": training.stage_a_checkpoint_path,
            "stage_b": training.stage_b_checkpoint_path,
        }
        stage_metrics: dict[str, dict[str, float]] = {}
        for stage, checkpoint_path in checkpoint_paths[mode].items():
            _load_adapter_checkpoint(model, checkpoint_path)
            stage_answer, stage_fixed, stage_pairs, stage_trace, stage_failures = _evaluate_stage(
                backend=backend,
                model=model,
                mode=mode,
                stage=stage,
                train_pairs=mode_train,
                test_pairs=mode_test,
                max_length=max_length,
                batch_size=evaluation_batch_size,
                seed=seed,
            )
            answer_rows.extend(stage_answer)
            fixed_rows.extend(stage_fixed)
            binary_pair_rows.extend(stage_pairs)
            trace_rows.extend(stage_trace)
            failure_cases.extend(stage_failures)
            stage_metrics[stage] = {
                "answer_accuracy": _mean(stage_answer, lambda row: row["eval_mode"] == "full"),
                "fixed_accuracy": _mean(stage_fixed, lambda row: row["eval_mode"] == "full"),
                "wrong_context_drop": _mean(stage_answer, lambda row: row["eval_mode"] == "wrong_context_drop"),
            }
        stage_rows.append(
            {
                "diagnostic_mode": mode,
                "stage_a_answer_accuracy": stage_metrics["stage_a"]["answer_accuracy"],
                "stage_b_answer_accuracy": stage_metrics["stage_b"]["answer_accuracy"],
                "answer_regression": stage_metrics["stage_a"]["answer_accuracy"] - stage_metrics["stage_b"]["answer_accuracy"],
                "stage_a_fixed_accuracy": stage_metrics["stage_a"]["fixed_accuracy"],
                "stage_b_fixed_accuracy": stage_metrics["stage_b"]["fixed_accuracy"],
                "fixed_improvement": stage_metrics["stage_b"]["fixed_accuracy"] - stage_metrics["stage_a"]["fixed_accuracy"],
            }
        )
        for eval_mode in ("no_memory_path", "no_state_path", "no_rule_path", "empty_context", "counterfactual_context"):
            full = _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "full")
            ablated = _mean(answer_rows, lambda row, current=mode, current_eval=eval_mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == current_eval)
            ablation_rows.append(
                {
                    "diagnostic_mode": mode,
                    "eval_mode": eval_mode,
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
        full = _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "full")
        wrong = _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "wrong_context")
        wrong_rows.append(
            {
                "diagnostic_mode": mode,
                "full_accuracy": full,
                "wrong_context_accuracy": wrong,
                "wrong_context_drop": full - wrong,
            }
        )
        resource_rows.append(
            {
                "diagnostic_mode": mode,
                "seconds": time.perf_counter() - mode_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        print(f"binary_path_diagnostic_mode_complete mode={mode} seconds={time.perf_counter() - mode_started:.2f}", flush=True)
        del model, _heads
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    drops = {(row["diagnostic_mode"], row["eval_mode"]): row["absolute_drop"] for row in ablation_rows}
    stage_metrics = {row["diagnostic_mode"]: row for row in stage_rows}
    stage_b_answer = {
        mode: _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "full")
        for mode in BINARY_DIAGNOSTIC_MODES
    }
    conflict_pair_success = _mean(
        binary_pair_rows,
        lambda row: row["diagnostic_mode"] == "memory_rule_conflict_diagnostic" and row["stage"] == "stage_b",
        field="pair_success",
    )
    max_hidden_norm = max((float(row["hidden_norm_ratio"]) for row in loss_rows), default=1.0)
    loss_decreased = {row["mode"]: row["loss_decreased"] for row in training_rows}
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            and not row["optimizer_contains_qwen_parameters"]
            for row in training_rows
        ),
        "separate_mode_checkpoints": len({paths["stage_b"] for paths in checkpoint_paths.values()}) == len(BINARY_DIAGNOSTIC_MODES),
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(
            abs(
                _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "adapter_disabled")
                - _mean(answer_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_b" and row["eval_mode"] == "zero_scale")
            )
            <= 1e-9
            for mode in BINARY_DIAGNOSTIC_MODES
        ),
        "hidden_norm_ratio": max_hidden_norm <= 2.0,
        "training_losses": all(values.get("binary_answer_margin_loss", False) for values in loss_decreased.values()),
        "memory_answer_accuracy": stage_b_answer["memory_only_diagnostic"] >= 0.80,
        "memory_path_drop": drops.get(("memory_only_diagnostic", "no_memory_path"), 0.0) >= 0.25,
        "rule_answer_accuracy": stage_b_answer["rule_only_diagnostic"] >= 0.80,
        "rule_path_drop": drops.get(("rule_only_diagnostic", "no_rule_path"), 0.0) >= 0.25,
        "conflict_pair_success": conflict_pair_success >= 0.75,
        "conflict_rule_path_drop": drops.get(("memory_rule_conflict_diagnostic", "no_rule_path"), 0.0) >= 0.20,
        "wrong_context_drop": all(row["wrong_context_drop"] >= 0.20 for row in wrong_rows),
        "stage_b_fixed_improvement": all(row["fixed_improvement"] >= 0.10 for row in stage_rows),
        "stage_b_answer_retention": all(row["answer_regression"] <= 0.05 for row in stage_rows),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "diagnostic_modes": list(BINARY_DIAGNOSTIC_MODES),
        "adapter_variant": adapter_variant,
        "pairs_per_mode": pairs_per_mode,
        "train_pairs": train_pairs,
        "held_out_pairs": held_out_pairs,
        "stage_a_steps": stage_a_steps,
        "stage_b_steps": stage_b_steps,
        "stage_b_answer_accuracy": stage_b_answer,
        "conflict_pair_success": conflict_pair_success,
        "stage_a_vs_stage_b": stage_metrics,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_real_task_remigration": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "binary_pair_metrics.csv", binary_pair_rows)
    _write_csv(output_path / "pair_flip_metrics.csv", binary_pair_rows)
    _write_csv(
        output_path / "binary_diagnostic_metrics.csv",
        [
            *({"metric_type": "answer_option", **row} for row in answer_rows),
            *({"metric_type": "fixed_centroid", **row} for row in fixed_rows),
            *({"metric_type": "path_ablation", **row} for row in ablation_rows),
        ],
    )
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "answer_option_metrics.csv", answer_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "stage_a_vs_stage_b.csv", stage_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _write_csv(
        output_path / "language_preservation.csv",
        [
            {
                "diagnostic_mode": mode,
                "status": "not_evaluated",
                "reason": "stage31 binary path diagnostic does not run generation or language preservation probes",
            }
            for mode in BINARY_DIAGNOSTIC_MODES
        ],
    )
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "dataset_mode": "binary_path_diagnostic_v1",
            "adapter_variant": adapter_variant,
            "modes": list(BINARY_DIAGNOSTIC_MODES),
            "pairs_per_mode": pairs_per_mode,
            "train_pairs": train_pairs,
            "held_out_pairs": held_out_pairs,
        },
    )
    return summary
