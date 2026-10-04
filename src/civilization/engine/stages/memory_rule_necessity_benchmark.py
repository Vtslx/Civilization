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

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, EVALUATION_MODES, _mean
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .evidence_answer_benchmark import (
    LOCAL_MEMORY_TASKS,
    LOCAL_RULE_TASKS,
    LOCAL_STATE_TASKS,
    PROBE_LAYER,
    ROUTES,
    _evaluate_task,
    _route_metric,
)
from .evidence_answer_data import (
    assert_no_logic_label_leakage,
    build_evidence_answer_samples,
)
from .memory_rule_necessity_training import run_memory_rule_necessity_training
from .multilayer_adapter_benchmark import _collect_multilayer_vectors, _language_preservation_rows
from .necessity_alignment_data import (
    NecessityPair,
    assert_necessity_pairs_valid,
    build_necessity_pairs,
)
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
    split_by_surface_group,
)
from .real_task_migration import _balanced_external_split


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


def _evaluate_necessity_pairs(
    *,
    backend: Qwen3Backend,
    model,
    pairs: list[NecessityPair],
    route: str,
    source_type: str,
    max_length: int,
    batch_size: int,
    seed: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    samples = [sample for pair in pairs for sample in (pair.full_sample, pair.counterfactual_sample)]
    disabled_vectors, _disabled_traces, disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    full_vectors, _full_traces, full_failures = _collect_multilayer_vectors(
        backend,
        model,
        samples,
        "full",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    failures = [
        {"route": route, "source_type": source_type, **item}
        for item in (*disabled_failures, *full_failures)
    ]
    for index, pair in enumerate(pairs):
        start = index * 2
        end = start + 2
        delta = full_vectors[PROBE_LAYER][start:end] - disabled_vectors[PROBE_LAYER][start:end]
        option_vectors = build_answer_option_vectors(backend, pair.base_record.answer_options)
        pair_samples = [pair.full_sample, pair.counterfactual_sample]
        answer = score_answer_options(delta, pair_samples, option_vectors, pair.base_record.answer_options)
        row0, row1 = answer.rows
        rows.append(
            {
                "route": route,
                "seed": seed,
                "source_type": source_type,
                "pair_type": pair.pair_type,
                "pair_id": pair.pair_id,
                "required_path": pair.required_path,
                "full_correct": bool(row0["correct"]),
                "counterfactual_correct": bool(row1["correct"]),
                "pair_success": bool(row0["correct"] and row1["correct"]),
                "full_margin": row0["margin"],
                "counterfactual_margin": row1["margin"],
                "delta_distance": float(np.linalg.norm(delta[0] - delta[1])),
            }
        )
    truncations = [item for item in failures if item.get("type") == "truncation"]
    hard_failures = [item for item in failures if item.get("type") != "truncation"]
    return rows, hard_failures, truncations


def _split_and_align_local(
    local_datasets,
    local_train_groups: int,
    seed: int,
):
    local_splits = {}
    local_train_samples = []
    local_test_samples = []
    for task_type, samples in local_datasets.items():
        train, test = split_by_surface_group(samples, local_train_groups, seed)
        aligned_train = [row.sample for row in build_evidence_answer_samples(train, "local_semireal")]
        aligned_test = [row.sample for row in build_evidence_answer_samples(test, "local_semireal")]
        local_splits[task_type] = (aligned_train, aligned_test)
        local_train_samples.extend(aligned_train)
        local_test_samples.extend(aligned_test)
    return local_splits, local_train_samples, local_test_samples


def run_qwen3_memory_rule_necessity_repair(
    output_dir: str | Path = "artifacts/civilization/memory_rule_necessity_repair",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    external_train_per_label: int = 12,
    external_heldout_per_label: int = 12,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "src/civilization/engine/data/external_cache",
    training_steps: int = 120,
    gradient_accumulation: int = 4,
    max_length: int = 64,
    external_max_length: int = 384,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    local_records = build_local_semireal_task_records(
        local_samples_per_label,
        seed=seed,
        context_grounding_mode="grounded_v1",
    )
    local_datasets, _ = records_to_logic_datasets(
        local_records,
        max_seq_len=max_length,
        context_grounding_mode="grounded_v1",
    )
    external_records, external_manifest = load_external_task_records(
        cache_dir=external_cache_dir,
        task_names=external_task_names,
        allow_download=False,
        max_records_per_task=320,
        context_grounding_mode="grounded_v1",
    )
    external_datasets, _ = records_to_logic_datasets(
        external_records,
        max_seq_len=max_length,
        context_grounding_mode="grounded_v1",
    )
    local_splits, local_train_samples, local_test_samples = _split_and_align_local(
        local_datasets,
        local_train_groups,
        seed,
    )
    external_splits = {}
    external_train_samples = []
    external_test_samples = []
    for task_name, samples in external_datasets.items():
        train, test = _balanced_external_split(samples, external_train_per_label, external_heldout_per_label)
        aligned_train = [row.sample for row in build_evidence_answer_samples(train, "external_benchmark")]
        aligned_test = [row.sample for row in build_evidence_answer_samples(test, "external_benchmark")]
        external_splits[task_name] = (aligned_train, aligned_test)
        external_train_samples.extend(aligned_train)
        external_test_samples.extend(aligned_test)

    local_training_records = build_evidence_answer_samples(local_train_samples, "local_semireal")
    external_training_records = build_evidence_answer_samples(external_train_samples, "external_benchmark")
    local_test_records = build_evidence_answer_samples(local_test_samples, "local_semireal")
    external_test_records = build_evidence_answer_samples(external_test_samples, "external_benchmark")
    assert_no_logic_label_leakage(local_training_records + external_training_records)
    local_train_pairs = build_necessity_pairs(local_training_records)
    external_train_pairs = build_necessity_pairs(external_training_records)
    local_test_pairs = build_necessity_pairs(local_test_records)
    external_test_pairs = build_necessity_pairs(external_test_records)
    assert_necessity_pairs_valid(local_train_pairs + external_train_pairs + local_test_pairs + external_test_pairs)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    local_retention_rows: list[dict[str, Any]] = []
    route_comparison: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    route_checkpoints: dict[str, str] = {}

    for route in ROUTES:
        route_started = time.perf_counter()
        print(f"memory_rule_necessity_route_start route={route} training_steps={training_steps}", flush=True)
        train_pairs = list(local_train_pairs)
        if route == "external_few_shot":
            train_pairs.extend(external_train_pairs)
        model, _heads, training = run_memory_rule_necessity_training(
            backend=backend,
            train_pairs=train_pairs,
            output_dir=output_path / "checkpoints",
            route=route,
            seed=seed,
            steps=training_steps,
            gradient_accumulation=gradient_accumulation,
            local_max_length=max_length,
            external_max_length=external_max_length,
        )
        training_row = asdict(training)
        losses = training_row.pop("losses")
        route_checkpoints[route] = training.checkpoint_path
        training_rows.append(training_row)
        loss_rows.extend({"route": route, "seed": seed, **row} for row in losses)

        for task_name, (train, test) in local_splits.items():
            centroid, answer, failures, truncations = _evaluate_task(
                backend=backend,
                model=model,
                train_samples=train,
                test_samples=test,
                modes=evaluation_modes,
                max_length=max_length,
                batch_size=evaluation_batch_size,
                source_type="local_semireal",
                task_name=task_name,
                seed=seed,
                fail_on_truncation=True,
            )
            fixed_rows.extend({"route": route, **row} for row in centroid)
            answer_rows.extend({"route": route, **row} for row in answer)
            failure_cases.extend({"route": route, **row} for row in failures)
            truncation_cases.extend({"route": route, **row} for row in truncations)
        for task_name, (train, test) in external_splits.items():
            centroid, answer, failures, truncations = _evaluate_task(
                backend=backend,
                model=model,
                train_samples=train,
                test_samples=test,
                modes=evaluation_modes,
                max_length=external_max_length,
                batch_size=evaluation_batch_size,
                source_type="external_benchmark",
                task_name=task_name,
                seed=seed,
                fail_on_truncation=False,
            )
            fixed_rows.extend({"route": route, **row} for row in centroid)
            answer_rows.extend({"route": route, **row} for row in answer)
            external_rows.extend({"route": route, **row} for row in answer)
            failure_cases.extend({"route": route, **row} for row in failures)
            truncation_cases.extend({"route": route, **row} for row in truncations)

        local_pair_metrics, local_pair_failures, local_pair_truncations = _evaluate_necessity_pairs(
            backend=backend,
            model=model,
            pairs=local_test_pairs,
            route=route,
            source_type="local_semireal",
            max_length=max_length,
            batch_size=evaluation_batch_size,
            seed=seed,
            fail_on_truncation=True,
        )
        pair_rows.extend(local_pair_metrics)
        failure_cases.extend(local_pair_failures)
        truncation_cases.extend(local_pair_truncations)
        if external_test_pairs:
            external_pair_metrics, external_pair_failures, external_pair_truncations = _evaluate_necessity_pairs(
                backend=backend,
                model=model,
                pairs=external_test_pairs,
                route=route,
                source_type="external_benchmark",
                max_length=external_max_length,
                batch_size=evaluation_batch_size,
                seed=seed,
                fail_on_truncation=False,
            )
            pair_rows.extend(external_pair_metrics)
            failure_cases.extend(external_pair_failures)
            truncation_cases.extend(external_pair_truncations)

        for mode, tasks in (
            ("no_memory_path", LOCAL_MEMORY_TASKS),
            ("no_state_path", LOCAL_STATE_TASKS),
            ("no_rule_path", LOCAL_RULE_TASKS),
        ):
            full = _route_metric(answer_rows, route, "local_semireal", "full", tasks)
            ablated = _route_metric(answer_rows, route, "local_semireal", mode, tasks)
            ablation_rows.append(
                {
                    "route": route,
                    "mode": mode,
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
        for source_type in ("local_semireal", "external_benchmark"):
            full = _route_metric(answer_rows, route, source_type, "full")
            wrong = _route_metric(answer_rows, route, source_type, "wrong_context")
            disabled = _route_metric(answer_rows, route, source_type, "adapter_disabled")
            zero = _route_metric(answer_rows, route, source_type, "zero_scale")
            wrong_rows.append(
                {
                    "route": route,
                    "source_type": source_type,
                    "full_accuracy": full,
                    "wrong_context_accuracy": wrong,
                    "wrong_context_drop": full - wrong,
                    "adapter_disabled_accuracy": disabled,
                    "zero_scale_accuracy": zero,
                    "disabled_zero_equivalent": abs(disabled - zero) <= 1e-9,
                }
            )
        preservation, _generation = _language_preservation_rows(
            backend,
            model,
            "dual_16_24",
            seed,
            max_length,
            evaluation_batch_size,
        )
        preservation_rows.extend({"route": route, **row} for row in preservation)
        resource_rows.append(
            {
                "route": route,
                "seconds": time.perf_counter() - route_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        print(f"memory_rule_necessity_route_complete route={route} seconds={time.perf_counter() - route_started:.2f}", flush=True)
        del model, _heads
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    for task_name in external_splits:
        local_only = _route_metric(answer_rows, "local_only_transfer", "external_benchmark", "full", {task_name})
        few_shot = _route_metric(answer_rows, "external_few_shot", "external_benchmark", "full", {task_name})
        route_comparison.append(
            {
                "task_name": task_name,
                "local_only_accuracy": local_only,
                "few_shot_accuracy": few_shot,
                "few_shot_improvement": few_shot - local_only,
            }
        )
    local_only_local = _route_metric(answer_rows, "local_only_transfer", "local_semireal", "full")
    few_shot_local = _route_metric(answer_rows, "external_few_shot", "local_semireal", "full")
    local_retention_rows.append(
        {
            "local_only_accuracy": local_only_local,
            "few_shot_accuracy": few_shot_local,
            "few_shot_regression": local_only_local - few_shot_local,
            "passes_retention": few_shot_local >= local_only_local - 0.05,
        }
    )
    drops = {(row["route"], row["mode"]): row["absolute_drop"] for row in ablation_rows}
    external_evidence = {}
    for route in ROUTES:
        route_results = []
        for task_name in external_splits:
            full = _route_metric(answer_rows, route, "external_benchmark", "full", {task_name})
            disabled = _route_metric(answer_rows, route, "external_benchmark", "adapter_disabled", {task_name})
            route_results.append({"task_name": task_name, "full_accuracy": full, "adapter_disabled_accuracy": disabled, "improvement": full - disabled})
        external_evidence[route] = route_results
    loss_decreased = {route: next(row["loss_decreased"] for row in training_rows if row["route"] == route) for route in ROUTES}
    local_only_wins = sum(row["improvement"] >= 0.03 for row in external_evidence["local_only_transfer"])
    few_shot_wins = sum(row["improvement"] >= 0.08 for row in external_evidence["external_few_shot"])
    route_improvements = sum(row["few_shot_improvement"] >= 0.05 for row in route_comparison)
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            and not row["optimizer_contains_qwen_parameters"]
            for row in training_rows
        ),
        "separate_route_models": len(set(route_checkpoints.values())) == len(ROUTES),
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(row["disabled_zero_equivalent"] for row in wrong_rows),
        "hidden_norm_ratio": all(
            max(float(loss["hidden_norm_ratio"]) for loss in loss_rows if loss["route"] == route) <= 2.0
            for route in ROUTES
        ),
        "training_losses": all(
            loss_decreased[route].get("memory_necessity_loss", False)
            and loss_decreased[route].get("rule_necessity_loss", False)
            and loss_decreased[route].get("fixed_centroid_group_flip_loss", False)
            for route in ROUTES
        ),
        "positive_answer_margin": all(
            _mean(
                loss_rows,
                lambda row, current=route: row["route"] == current and float(row["step"]) >= training_steps - min(10, training_steps),
                field="correct_option_margin",
            ) > 0.0
            for route in ROUTES
        ),
        "local_answer_accuracy": few_shot_local >= 0.70,
        "local_fixed_centroid_accuracy": _route_metric(fixed_rows, "external_few_shot", "local_semireal", "full") >= 0.50,
        "local_surface_flip": _route_metric(
            fixed_rows,
            "external_few_shot",
            "local_semireal",
            "full",
            field="surface_group_flip_accuracy",
        ) >= 0.50,
        "local_wrong_context": (
            few_shot_local - _route_metric(answer_rows, "external_few_shot", "local_semireal", "wrong_context")
        ) >= 0.10,
        "memory_path_drop": drops.get(("external_few_shot", "no_memory_path"), 0.0) >= 0.10,
        "rule_path_drop": drops.get(("external_few_shot", "no_rule_path"), 0.0) >= 0.10,
        "state_path_drop": drops.get(("external_few_shot", "no_state_path"), 0.0) >= 0.06,
        "local_only_external_evidence": local_only_wins >= min(1, len(external_splits)),
        "few_shot_external_evidence": few_shot_wins >= min(2, len(external_splits)),
        "few_shot_route_improvement": route_improvements >= min(2, len(external_splits)),
        "few_shot_wrong_context": (
            _route_metric(answer_rows, "external_few_shot", "external_benchmark", "full")
            - _route_metric(answer_rows, "external_few_shot", "external_benchmark", "wrong_context")
        ) >= 0.10,
        "local_retention": few_shot_local >= local_only_local - 0.05,
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "routes": list(ROUTES),
        "training_steps": training_steps,
        "local_only_local_answer_accuracy": local_only_local,
        "few_shot_local_answer_accuracy": few_shot_local,
        "external_evidence": external_evidence,
        "route_comparison": route_comparison,
        "external_truncation_count": len(truncation_cases),
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage27b_rerun": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "necessity_pair_metrics.csv", pair_rows)
    _write_csv(output_path / "answer_option_metrics.csv", answer_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "route_comparison.csv", route_comparison)
    _write_csv(output_path / "local_retention.csv", local_retention_rows)
    _write_csv(output_path / "external_task_metrics.csv", external_rows)
    _write_csv(output_path / "language_preservation.csv", preservation_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "local_semireal": local_real_task_manifest(local_records),
            "external_benchmark": external_manifest,
            "necessity_pair_types": ["memory_necessity_pair", "rule_necessity_pair", "memory_rule_conflict_pair"],
        },
    )
    return summary
