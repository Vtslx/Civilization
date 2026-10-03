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

from experiments.civilization_transformer_torch.analysis.dataset import LogicSample

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, EVALUATION_MODES, _mean
from .answer_option_readout import (
    answer_options_for_samples,
    build_answer_option_vectors,
    score_answer_options,
)
from .evidence_answer_data import (
    assert_no_logic_label_leakage,
    build_evidence_answer_samples,
    wrong_context_sample,
)
from .evidence_answer_training import run_evidence_answer_alignment_training
from .multilayer_adapter_benchmark import (
    _collect_multilayer_vectors,
    _language_preservation_rows,
)
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
    split_by_surface_group,
)
from .real_task_migration import (
    _balanced_external_split,
    _generic_centroid_metrics,
    _surface_flip_accuracy,
)


ROUTES = ("local_only_transfer", "external_few_shot")
PROBE_LAYER = 28
LOCAL_MEMORY_TASKS = {"operation_decision", "causal_trace"}
LOCAL_STATE_TASKS = {"condition_check", "negation_constraint"}
LOCAL_RULE_TASKS = {"rule_conflict", "priority_selection"}


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


def _route_metric(
    rows: list[dict[str, Any]],
    route: str,
    source_type: str,
    mode: str,
    tasks: set[str] | None = None,
    field: str = "accuracy",
) -> float:
    return _mean(
        rows,
        lambda row: row["route"] == route
        and row["source_type"] == source_type
        and row["mode"] == mode
        and row["probe_layer"] == PROBE_LAYER
        and (tasks is None or row["task_name"] in tasks),
        field=field,
    )


def _evaluate_task(
    *,
    backend: Qwen3Backend,
    model,
    train_samples: list[LogicSample],
    test_samples: list[LogicSample],
    modes: tuple[str, ...],
    max_length: int,
    batch_size: int,
    source_type: str,
    task_name: str,
    seed: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_vectors, _train_traces, train_failures = _collect_multilayer_vectors(
        backend,
        model,
        train_samples,
        "full",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    disabled_vectors, _disabled_traces, disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        test_samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    train_labels = [sample.label for sample in train_samples]
    test_labels = [sample.label for sample in test_samples]
    options = answer_options_for_samples(train_samples + test_samples)
    option_vectors = build_answer_option_vectors(backend, options)
    fixed_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    failures = [
        {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "train", **item}
        for item in train_failures
    ] + [
        {
            "seed": seed,
            "source_type": source_type,
            "task_name": task_name,
            "split": "test",
            **item,
        }
        for item in disabled_failures
    ]
    for mode in modes:
        mode_samples = test_samples
        collector_mode = mode
        if mode == "wrong_context":
            mode_samples = [
                wrong_context_sample(record)
                for record in build_evidence_answer_samples(test_samples, source_type)
            ]
            collector_mode = "full"
        if mode == "adapter_disabled":
            test_vectors = disabled_vectors
            test_failures: list[dict[str, Any]] = []
        else:
            test_vectors, _test_traces, test_failures = _collect_multilayer_vectors(
                backend,
                model,
                mode_samples,
                collector_mode,
                max_length,
                batch_size,
                (PROBE_LAYER,),
                fail_on_truncation=fail_on_truncation,
            )
        failures.extend(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "split": "test",
                **item,
            }
            for item in test_failures
        )
        centroid = _generic_centroid_metrics(
            train_vectors[PROBE_LAYER],
            train_labels,
            test_vectors[PROBE_LAYER],
            test_labels,
        )
        fixed_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": PROBE_LAYER,
                "accuracy": centroid["accuracy"],
                "macro_accuracy": centroid["macro_accuracy"],
                "mean_distance": centroid["mean_distance"],
                "surface_group_flip_accuracy": _surface_flip_accuracy(
                    test_samples,
                    centroid["predictions"],
                ),
                "centroid_source": "full_context_train",
            }
        )
        adapter_delta = test_vectors[PROBE_LAYER] - disabled_vectors[PROBE_LAYER]
        answer = score_answer_options(adapter_delta, test_samples, option_vectors, options)
        answer_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": PROBE_LAYER,
                "accuracy": answer.accuracy,
                "macro_accuracy": answer.macro_accuracy,
                "mean_margin": answer.mean_margin,
                "options": "|".join(options),
                "representation": "adapter_delta_vs_disabled",
                "delta_norm": float(np.linalg.norm(adapter_delta, axis=1).mean()),
            }
        )
    truncations = [item for item in failures if item.get("type") == "truncation"]
    hard_failures = [item for item in failures if item.get("type") != "truncation"]
    return fixed_rows, answer_rows, hard_failures, truncations


def run_qwen3_evidence_answer_alignment(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/evidence_answer_alignment",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    external_train_per_label: int = 12,
    external_heldout_per_label: int = 12,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "experiments/civilization_transformer_qwen3/data/external_cache",
    training_steps: int = 100,
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

    local_splits: dict[str, tuple[list[LogicSample], list[LogicSample]]] = {}
    local_train_samples: list[LogicSample] = []
    for task_type, samples in local_datasets.items():
        train, test = split_by_surface_group(samples, local_train_groups, seed)
        aligned_train = [row.sample for row in build_evidence_answer_samples(train, "local_semireal")]
        aligned_test = [row.sample for row in build_evidence_answer_samples(test, "local_semireal")]
        local_splits[task_type] = (aligned_train, aligned_test)
        local_train_samples.extend(aligned_train)

    external_splits: dict[str, tuple[list[LogicSample], list[LogicSample]]] = {}
    external_train_samples: list[LogicSample] = []
    for task_name, samples in external_datasets.items():
        train, test = _balanced_external_split(samples, external_train_per_label, external_heldout_per_label)
        aligned_train = [row.sample for row in build_evidence_answer_samples(train, "external_benchmark")]
        aligned_test = [row.sample for row in build_evidence_answer_samples(test, "external_benchmark")]
        external_splits[task_name] = (aligned_train, aligned_test)
        external_train_samples.extend(aligned_train)

    local_training_records = build_evidence_answer_samples(local_train_samples, "local_semireal")
    external_training_records = build_evidence_answer_samples(external_train_samples, "external_benchmark")
    assert_no_logic_label_leakage(local_training_records)
    assert_no_logic_label_leakage(external_training_records)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    context_pair_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    route_checkpoints: dict[str, str] = {}

    for route in ROUTES:
        route_started = time.perf_counter()
        print(
            f"evidence_answer_route_start route={route} "
            f"training_steps={training_steps}",
            flush=True,
        )
        train_records = list(local_training_records)
        if route == "external_few_shot":
            train_records.extend(external_training_records)
        model, _heads, training = run_evidence_answer_alignment_training(
            backend=backend,
            train_records=train_records,
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
        for task_name in (*local_splits.keys(), *external_splits.keys()):
            source_type = "local_semireal" if task_name in local_splits else "external_benchmark"
            full = _route_metric(answer_rows, route, source_type, "full", {task_name})
            wrong = _route_metric(answer_rows, route, source_type, "wrong_context", {task_name})
            context_pair_rows.append(
                {
                    "route": route,
                    "source_type": source_type,
                    "task_name": task_name,
                    "full_accuracy": full,
                    "wrong_context_accuracy": wrong,
                    "absolute_drop": full - wrong,
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
        print(
            f"evidence_answer_route_complete route={route} "
            f"seconds={time.perf_counter() - route_started:.2f}",
            flush=True,
        )
        del model, _heads
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    route_comparison: list[dict[str, Any]] = []
    for task_name in external_splits:
        local_only = _route_metric(
            answer_rows,
            "local_only_transfer",
            "external_benchmark",
            "full",
            {task_name},
        )
        few_shot = _route_metric(
            answer_rows,
            "external_few_shot",
            "external_benchmark",
            "full",
            {task_name},
        )
        route_comparison.append(
            {
                "task_name": task_name,
                "local_only_accuracy": local_only,
                "few_shot_accuracy": few_shot,
                "few_shot_improvement": few_shot - local_only,
            }
        )

    drops = {
        (row["route"], row["mode"]): row["absolute_drop"]
        for row in ablation_rows
    }
    external_evidence: dict[str, list[dict[str, float | str]]] = {}
    for route in ROUTES:
        route_results = []
        for task_name in external_splits:
            full = _route_metric(answer_rows, route, "external_benchmark", "full", {task_name})
            disabled = _route_metric(answer_rows, route, "external_benchmark", "adapter_disabled", {task_name})
            route_results.append(
                {
                    "task_name": task_name,
                    "full_accuracy": full,
                    "adapter_disabled_accuracy": disabled,
                    "improvement": full - disabled,
                }
            )
        external_evidence[route] = route_results

    loss_decreased = {
        route: next(row["loss_decreased"] for row in training_rows if row["route"] == route)
        for route in ROUTES
    }
    local_few_shot_answer = _route_metric(answer_rows, "external_few_shot", "local_semireal", "full")
    local_only_answer = _route_metric(answer_rows, "local_only_transfer", "local_semireal", "full")
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
            max(
                float(loss["hidden_norm_ratio"])
                for loss in loss_rows
                if loss["route"] == route
            )
            <= 2.0
            for route in ROUTES
        ),
        "training_losses": all(
            loss_decreased[route].get("answer_option_margin_loss", False)
            and loss_decreased[route].get("wrong_context_separation_loss", False)
            and loss_decreased[route].get("evidence_to_answer_consistency_loss", False)
            for route in ROUTES
        ),
        "positive_answer_margin": all(
            _mean(
                loss_rows,
                lambda row, current=route: row["route"] == current
                and float(row["step"]) >= training_steps - min(10, training_steps),
                field="correct_option_margin",
            )
            > 0.0
            for route in ROUTES
        ),
        "local_answer_accuracy": local_few_shot_answer >= 0.60,
        "local_fixed_centroid_accuracy": _route_metric(
            fixed_rows, "external_few_shot", "local_semireal", "full"
        )
        >= 0.45,
        "local_surface_flip": _route_metric(
            fixed_rows,
            "external_few_shot",
            "local_semireal",
            "full",
            field="surface_group_flip_accuracy",
        )
        >= 0.40,
        "local_wrong_context": (
            local_few_shot_answer
            - _route_metric(answer_rows, "external_few_shot", "local_semireal", "wrong_context")
        )
        >= 0.10,
        "memory_path_drop": drops.get(("external_few_shot", "no_memory_path"), 0.0) >= 0.08,
        "state_path_drop": drops.get(("external_few_shot", "no_state_path"), 0.0) >= 0.06,
        "rule_path_drop": drops.get(("external_few_shot", "no_rule_path"), 0.0) >= 0.06,
        "local_only_external_evidence": local_only_wins >= min(1, len(external_splits)),
        "few_shot_external_evidence": few_shot_wins >= min(2, len(external_splits)),
        "few_shot_route_improvement": route_improvements >= min(2, len(external_splits)),
        "few_shot_wrong_context": (
            _route_metric(answer_rows, "external_few_shot", "external_benchmark", "full")
            - _route_metric(answer_rows, "external_few_shot", "external_benchmark", "wrong_context")
        )
        >= 0.10,
        "local_retention": local_few_shot_answer >= local_only_answer - 0.05,
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "routes": list(ROUTES),
        "training_steps": training_steps,
        "local_only_local_answer_accuracy": local_only_answer,
        "few_shot_local_answer_accuracy": local_few_shot_answer,
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
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "local_semireal": local_real_task_manifest(local_records),
            "external_benchmark": external_manifest,
            "held_out_surface_groups": {
                task: sorted({sample.surface_group_id for sample in split[1]})
                for task, split in external_splits.items()
            },
        },
    )
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "answer_option_metrics.csv", answer_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "context_pair_metrics.csv", context_pair_rows)
    _write_csv(output_path / "external_task_metrics.csv", external_rows)
    _write_csv(output_path / "route_comparison.csv", route_comparison)
    _write_csv(output_path / "language_preservation.csv", preservation_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    return summary
