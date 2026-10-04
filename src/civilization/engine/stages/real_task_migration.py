from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import psutil
import torch

from civilization.research.torch_line.analysis.dataset import LogicSample

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, EVALUATION_MODES, _mean
from .multilayer_adapter_benchmark import _collect_multilayer_vectors, _language_preservation_rows
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
    split_by_surface_group,
)
from .surface_group_training import run_qwen3_surface_flip_training


PROBE_LAYERS = (18, 26, 28)
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


def _generic_centroid_metrics(
    train_vectors: np.ndarray,
    train_labels: list[str],
    test_vectors: np.ndarray,
    test_labels: list[str],
) -> dict[str, Any]:
    labels = tuple(sorted(set(train_labels)))
    if not labels:
        raise ValueError("centroid metrics require at least one train label")
    if not set(test_labels).issubset(set(labels)):
        raise ValueError("test labels must be present in train labels")
    centroids = {
        label: train_vectors[np.array(train_labels) == label].mean(axis=0)
        for label in labels
    }
    predictions = []
    distances = []
    for vector in test_vectors:
        row_distances = {
            label: float(np.linalg.norm(vector - centroid))
            for label, centroid in centroids.items()
        }
        predicted = min(row_distances, key=row_distances.get)
        predictions.append(predicted)
        distances.append(row_distances[predicted])
    correct = [predicted == truth for predicted, truth in zip(predictions, test_labels, strict=True)]
    per_label = {}
    for label in labels:
        values = [ok for ok, truth in zip(correct, test_labels, strict=True) if truth == label]
        per_label[label] = sum(values) / len(values) if values else 0.0
    return {
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "macro_accuracy": float(np.mean(list(per_label.values()))) if per_label else 0.0,
        "per_label_accuracy": per_label,
        "predictions": predictions,
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
    }


def _balanced_external_split(samples: list[LogicSample], train_per_label: int, heldout_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in sorted({sample.label for sample in samples}):
        rows = [sample for sample in samples if sample.label == label]
        if len(rows) < 2:
            continue
        current_train_count = min(train_per_label, max(1, len(rows) // 2))
        current_heldout_count = min(heldout_per_label, len(rows) - current_train_count)
        train.extend(rows[:current_train_count])
        test.extend(rows[current_train_count : current_train_count + current_heldout_count])
    if {sample.surface_group_id for sample in train} & {sample.surface_group_id for sample in test}:
        raise ValueError("external train/test split leaked surface groups")
    return train, test


def _surface_flip_accuracy(samples: list[LogicSample], predictions: list[str]) -> float:
    groups: dict[str, list[bool]] = {}
    for sample, prediction in zip(samples, predictions, strict=True):
        groups.setdefault(sample.surface_group_id, []).append(sample.label == prediction)
    return sum(all(values) and len(values) > 1 for values in groups.values()) / len(groups) if groups else 0.0


def _build_external_logic_datasets(
    external_cache_dir: str | Path,
    external_task_names: tuple[str, ...],
    max_seq_len: int,
    allow_dataset_download: bool | None,
    max_records_per_task: int,
) -> tuple[dict[str, list[LogicSample]], list[dict[str, Any]]]:
    if not external_task_names:
        return {}, []
    records, manifest = load_external_task_records(
        cache_dir=external_cache_dir,
        task_names=external_task_names,
        allow_download=allow_dataset_download,
        max_records_per_task=max_records_per_task,
    )
    datasets, _tokenizer = records_to_logic_datasets(records, max_seq_len=max_seq_len)
    return datasets, manifest


def run_qwen3_real_task_migration(
    output_dir: str | Path = "artifacts/civilization/real_task_migration",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seeds: tuple[int, ...] = (202, 303, 404),
    local_samples_per_label: int = 40,
    local_train_groups: int = 24,
    external_train_per_label: int = 32,
    external_heldout_per_label: int = 32,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "src/civilization/engine/data/external_cache",
    allow_dataset_download: bool | None = None,
    max_records_per_external_task: int = 320,
    training_steps: int = 140,
    max_length: int = 64,
    external_max_length: int = 256,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    local_records = build_local_semireal_task_records(local_samples_per_label)
    local_datasets, _local_tokenizer = records_to_logic_datasets(local_records, max_seq_len=max_length)
    external_datasets, external_manifest = _build_external_logic_datasets(
        external_cache_dir=external_cache_dir,
        external_task_names=external_task_names,
        max_seq_len=max_length,
        allow_dataset_download=allow_dataset_download,
        max_records_per_task=max_records_per_external_task,
    )
    dataset_manifest = {
        "local_semireal": local_real_task_manifest(local_records),
        "external_benchmark": external_manifest,
        "external_cache_required": bool(external_task_names),
        "external_training_default": "evaluation_only",
    }

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_context_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    generation_sanity: list[dict[str, Any]] = []

    for run_index, seed in enumerate(seeds, start=1):
        print(f"qwen3_real_task_training seed={seed} run={run_index}/{len(seeds)}", flush=True)
        local_train: list[LogicSample] = []
        local_test_by_task: dict[str, list[LogicSample]] = {}
        local_train_by_task: dict[str, list[LogicSample]] = {}
        for task_type, samples in local_datasets.items():
            train, test = split_by_surface_group(samples, local_train_groups, seed)
            local_train.extend(train)
            local_train_by_task[task_type] = train
            local_test_by_task[task_type] = test
        run_started = time.perf_counter()
        model, _heads, training = run_qwen3_surface_flip_training(
            backend=backend,
            train_samples=local_train,
            output_dir=output_path / "checkpoints",
            target_layers=(16, 24),
            seed=seed,
            steps=training_steps,
            max_length=max_length,
            training_mode="surface_group_flip_alignment",
            scenario_weights={task_type: 1.0 for task_type in local_datasets},
        )
        row = asdict(training)
        losses = row.pop("losses")
        training_rows.append(row)
        loss_rows.extend({"seed": seed, **loss} for loss in losses)

        local_reference: dict[str, tuple[dict[int, np.ndarray], list[LogicSample]]] = {}
        external_reference: dict[str, tuple[dict[int, np.ndarray], list[LogicSample]]] = {}
        for mode in evaluation_modes:
            print(f"qwen3_real_task_eval_local seed={seed} mode={mode}", flush=True)
            full_train_vectors, train_traces, train_failures = _collect_multilayer_vectors(
                backend,
                model,
                local_train,
                mode,
                max_length,
                evaluation_batch_size,
                PROBE_LAYERS,
            )
            if mode == "full":
                local_reference["all"] = (full_train_vectors, local_train)
            train_vectors, train_samples = local_reference["all"]
            failure_cases.extend({"seed": seed, "source": "local", "split": "train", **item} for item in train_failures)
            for task_type, test_samples in local_test_by_task.items():
                test_vectors, test_traces, test_failures = _collect_multilayer_vectors(
                    backend,
                    model,
                    test_samples,
                    mode,
                    max_length,
                    evaluation_batch_size,
                    PROBE_LAYERS,
                )
                failure_cases.extend({"seed": seed, "source": "local", "task_type": task_type, **item} for item in test_failures)
                task_train_indices = [
                    index for index, sample in enumerate(train_samples) if sample.variant == task_type
                ]
                labels_train = [train_samples[index].label for index in task_train_indices]
                labels_test = [sample.label for sample in test_samples]
                for layer in PROBE_LAYERS:
                    metrics = _generic_centroid_metrics(
                        train_vectors[layer][task_train_indices],
                        labels_train,
                        test_vectors[layer],
                        labels_test,
                    )
                    row = {
                        "seed": seed,
                        "source_type": "local_semireal",
                        "task_type": task_type,
                        "mode": mode,
                        "probe_layer": layer,
                        "accuracy": metrics["accuracy"],
                        "macro_accuracy": metrics["macro_accuracy"],
                        "mean_distance": metrics["mean_distance"],
                        "surface_group_flip_accuracy": _surface_flip_accuracy(test_samples, metrics["predictions"]),
                    }
                    local_rows.append(row)
                    if mode in {"wrong_context", "empty_context", "adapter_disabled"} and layer == 28:
                        wrong_context_rows.append(row)
            del train_traces

        for mode in evaluation_modes:
            for task_name, samples in external_datasets.items():
                train, test = _balanced_external_split(
                    samples,
                    train_per_label=external_train_per_label,
                    heldout_per_label=external_heldout_per_label,
                )
                if not train or not test:
                    failure_cases.append({"seed": seed, "source": "external", "task_name": task_name, "type": "empty_split"})
                    continue
                print(f"qwen3_real_task_eval_external seed={seed} mode={mode} task={task_name}", flush=True)
                if mode == "full":
                    train_vectors, train_traces, train_failures = _collect_multilayer_vectors(
                        backend,
                        model,
                        train,
                        mode,
                        external_max_length,
                        evaluation_batch_size,
                        PROBE_LAYERS,
                        fail_on_truncation=False,
                    )
                    external_reference[task_name] = (train_vectors, train)
                    for item in train_failures:
                        target = truncation_cases if item.get("type") == "truncation" else failure_cases
                        target.append({"seed": seed, "source": "external", "task_name": task_name, "split": "train", **item})
                train_vectors, train_samples = external_reference[task_name]
                test_vectors, test_traces, test_failures = _collect_multilayer_vectors(
                    backend,
                    model,
                    test,
                    mode,
                    external_max_length,
                    evaluation_batch_size,
                    PROBE_LAYERS,
                    fail_on_truncation=False,
                )
                for item in test_failures:
                    target = truncation_cases if item.get("type") == "truncation" else failure_cases
                    target.append({"seed": seed, "source": "external", "task_name": task_name, "split": "test", **item})
                labels_train = [sample.label for sample in train_samples]
                labels_test = [sample.label for sample in test]
                external_labels = [sample.expected_pattern for sample in test]
                for layer in PROBE_LAYERS:
                    metrics = _generic_centroid_metrics(
                        train_vectors[layer],
                        labels_train,
                        test_vectors[layer],
                        labels_test,
                    )
                    external_rows.append(
                        {
                            "seed": seed,
                            "source_type": "external_benchmark",
                            "task_name": task_name,
                            "mode": mode,
                            "probe_layer": layer,
                            "accuracy": metrics["accuracy"],
                            "macro_accuracy": metrics["macro_accuracy"],
                            "answer_option_accuracy": metrics["accuracy"],
                            "external_labels": "|".join(sorted(set(external_labels))),
                            "mean_distance": metrics["mean_distance"],
                        }
                    )

        preservation, generation = _language_preservation_rows(
            backend,
            model,
            "dual_16_24",
            seed,
            max_length,
            evaluation_batch_size,
        )
        preservation_rows.extend(preservation)
        generation_sanity.append(generation)
        resource_rows.append(
            {
                "seed": seed,
                "seconds": time.perf_counter() - run_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        del model, _heads
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    for mode, tasks in (
        ("no_memory_path", LOCAL_MEMORY_TASKS),
        ("no_state_path", LOCAL_STATE_TASKS),
        ("no_rule_path", LOCAL_RULE_TASKS),
        ("adapter_disabled", set(local_datasets)),
        ("wrong_context", set(local_datasets)),
    ):
        full = _mean(local_rows, lambda row, task_set=tasks: row["mode"] == "full" and row["task_type"] in task_set)
        ablated = _mean(local_rows, lambda row, task_set=tasks, current=mode: row["mode"] == current and row["task_type"] in task_set)
        ablation_rows.append(
            {
                "source_type": "local_semireal",
                "mode": mode,
                "full_accuracy": full,
                "ablated_accuracy": ablated,
                "absolute_drop": full - ablated,
            }
        )
    external_improvements = []
    for task_name in external_datasets:
        full = _mean(external_rows, lambda row, current=task_name: row["mode"] == "full" and row["task_name"] == current and row["probe_layer"] == 28)
        disabled = _mean(external_rows, lambda row, current=task_name: row["mode"] == "adapter_disabled" and row["task_name"] == current and row["probe_layer"] == 28)
        external_improvements.append(
            {
                "task_name": task_name,
                "full_accuracy": full,
                "adapter_disabled_accuracy": disabled,
                "improvement": full - disabled,
            }
        )

    local_final_full = [row for row in local_rows if row["mode"] == "full" and row["probe_layer"] == 28]
    local_flip = _mean(local_final_full, lambda _row: True, field="surface_group_flip_accuracy")
    local_accuracy = _mean(local_final_full, lambda _row: True)
    drops = {row["mode"]: row["absolute_drop"] for row in ablation_rows}
    external_wins = sum(row["improvement"] >= 0.05 for row in external_improvements)
    seed_means = [
        _mean(local_final_full, lambda row, current=seed: row["seed"] == current)
        for seed in seeds
    ]
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            for row in training_rows
        ),
        "no_engineering_failures": not failure_cases,
        "local_accuracy": local_accuracy >= 0.75,
        "local_surface_flip": local_flip >= 0.85,
        "memory_path_drop": drops.get("no_memory_path", 0.0) >= 0.15,
        "state_path_drop": drops.get("no_state_path", 0.0) >= 0.10,
        "rule_path_drop": drops.get("no_rule_path", 0.0) >= 0.10,
        "wrong_context_control": drops.get("wrong_context", 0.0) >= 0.10,
        "adapter_disabled_control": _mean(local_rows, lambda row: row["mode"] == "adapter_disabled" and row["probe_layer"] == 28) <= 0.45,
        "external_baselines_recorded": len(external_improvements) == len(external_task_names),
        "external_evidence": not external_task_names or external_wins >= min(2, len(external_task_names)),
        "seed_stability": len(seed_means) <= 1 or statistics.pstdev(seed_means) <= 0.12,
        "language_preservation": all(
            (
                _mean(
                    preservation_rows,
                    lambda row, variant=variant: row["mode"] == "adapter_disabled" and row["variant"] == variant,
                )
                - _mean(
                    preservation_rows,
                    lambda row, variant=variant: row["mode"] == "empty_context" and row["variant"] == variant,
                )
            )
            <= 0.05
            for variant in ("canonical", "synonym")
        ),
        "generation_sanity": bool(generation_sanity) and all(row["nonempty"] and not row["hook_leak"] for row in generation_sanity),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "target_layers": [16, 24],
        "probe_layers": list(PROBE_LAYERS),
        "seeds": list(seeds),
        "local_samples_per_label": local_samples_per_label,
        "local_train_groups": local_train_groups,
        "external_task_names": list(external_task_names),
        "external_training_default": "evaluation_only",
        "training_steps": training_steps,
        "max_length": max_length,
        "external_max_length": external_max_length,
        "local_accuracy": local_accuracy,
        "local_surface_group_flip_accuracy": local_flip,
        "external_improvements": external_improvements,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_real_task_expansion": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "external_truncation_count": len(truncation_cases),
        "backend_runtime_trace": backend.runtime_trace,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "generation_sanity": generation_sanity,
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "dataset_manifest.json", dataset_manifest)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "local_task_metrics.csv", local_rows)
    _write_csv(output_path / "external_task_metrics.csv", external_rows)
    _write_csv(output_path / "ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context.csv", wrong_context_rows)
    _write_csv(output_path / "language_preservation.csv", preservation_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    return summary
