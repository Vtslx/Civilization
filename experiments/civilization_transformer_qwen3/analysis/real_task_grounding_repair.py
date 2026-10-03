from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.civilization_transformer_torch.analysis.dataset import LogicSample

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, EVALUATION_MODES, _mean
from .answer_option_readout import (
    answer_options_for_samples,
    build_answer_option_vectors,
    score_answer_options,
)
from .multilayer_adapter_benchmark import _collect_multilayer_vectors
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
    split_by_surface_group,
)
from .real_task_migration import _balanced_external_split, _generic_centroid_metrics, _surface_flip_accuracy
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


def _evaluate_samples(
    *,
    backend: Qwen3Backend,
    model,
    train_samples: list[LogicSample],
    test_samples: list[LogicSample],
    mode: str,
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
        mode,
        max_length,
        batch_size,
        PROBE_LAYERS,
        fail_on_truncation=fail_on_truncation,
    )
    test_vectors, _test_traces, test_failures = _collect_multilayer_vectors(
        backend,
        model,
        test_samples,
        mode,
        max_length,
        batch_size,
        PROBE_LAYERS,
        fail_on_truncation=fail_on_truncation,
    )
    centroid_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    labels_train = [sample.label for sample in train_samples]
    labels_test = [sample.label for sample in test_samples]
    for layer in PROBE_LAYERS:
        centroid = _generic_centroid_metrics(
            train_vectors[layer],
            labels_train,
            test_vectors[layer],
            labels_test,
        )
        centroid_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": layer,
                "accuracy": centroid["accuracy"],
                "macro_accuracy": centroid["macro_accuracy"],
                "mean_distance": centroid["mean_distance"],
                "surface_group_flip_accuracy": _surface_flip_accuracy(test_samples, centroid["predictions"]),
            }
        )
        options = answer_options_for_samples(train_samples + test_samples)
        option_vectors = build_answer_option_vectors(backend, options)
        answer = score_answer_options(test_vectors[layer], test_samples, option_vectors, options)
        answer_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": layer,
                "accuracy": answer.accuracy,
                "macro_accuracy": answer.macro_accuracy,
                "mean_margin": answer.mean_margin,
                "options": "|".join(options),
            }
        )
    failures = [
        {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "train", **item}
        for item in train_failures
    ] + [
        {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "test", **item}
        for item in test_failures
    ]
    truncations = [item for item in failures if item.get("type") == "truncation"]
    hard_failures = [item for item in failures if item.get("type") != "truncation"]
    return centroid_rows, answer_rows, hard_failures, truncations


def run_qwen3_real_task_grounding_repair(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/real_task_grounding_repair",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 12,
    local_train_groups: int = 7,
    external_train_per_label: int = 16,
    external_heldout_per_label: int = 16,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "experiments/civilization_transformer_qwen3/data/external_cache",
    training_steps: int = 60,
    max_length: int = 64,
    external_max_length: int = 384,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
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

    local_train: list[LogicSample] = []
    local_splits: dict[str, tuple[list[LogicSample], list[LogicSample]]] = {}
    for task_type, samples in local_datasets.items():
        train, test = split_by_surface_group(samples, local_train_groups, seed)
        local_train.extend(train)
        local_splits[task_type] = (train, test)

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

    fixed_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    for mode in evaluation_modes:
        for task_type, (train, test) in local_splits.items():
            centroid, answer, failures, truncations = _evaluate_samples(
                backend=backend,
                model=model,
                train_samples=train,
                test_samples=test,
                mode=mode,
                max_length=max_length,
                batch_size=evaluation_batch_size,
                source_type="local_semireal",
                task_name=task_type,
                seed=seed,
                fail_on_truncation=True,
            )
            fixed_rows.extend(centroid)
            answer_rows.extend(answer)
            failure_cases.extend(failures)
            truncation_cases.extend(truncations)
        for task_name, samples in external_datasets.items():
            train, test = _balanced_external_split(samples, external_train_per_label, external_heldout_per_label)
            centroid, answer, failures, truncations = _evaluate_samples(
                backend=backend,
                model=model,
                train_samples=train,
                test_samples=test,
                mode=mode,
                max_length=external_max_length,
                batch_size=evaluation_batch_size,
                source_type="external_benchmark",
                task_name=task_name,
                seed=seed,
                fail_on_truncation=False,
            )
            fixed_rows.extend(centroid)
            answer_rows.extend(answer)
            failure_cases.extend(failures)
            truncation_cases.extend(truncations)

    ablation_rows: list[dict[str, Any]] = []
    for mode, tasks in (
        ("no_memory_path", LOCAL_MEMORY_TASKS),
        ("no_state_path", LOCAL_STATE_TASKS),
        ("no_rule_path", LOCAL_RULE_TASKS),
        ("adapter_disabled", set(local_datasets)),
        ("wrong_context", set(local_datasets)),
    ):
        full = _mean(
            fixed_rows,
            lambda row, task_set=tasks: row["source_type"] == "local_semireal"
            and row["mode"] == "full"
            and row["task_name"] in task_set
            and row["probe_layer"] == 28,
        )
        ablated = _mean(
            fixed_rows,
            lambda row, task_set=tasks, current=mode: row["source_type"] == "local_semireal"
            and row["mode"] == current
            and row["task_name"] in task_set
            and row["probe_layer"] == 28,
        )
        ablation_rows.append(
            {
                "source_type": "local_semireal",
                "mode": mode,
                "full_accuracy": full,
                "ablated_accuracy": ablated,
                "absolute_drop": full - ablated,
            }
        )

    local_fixed_full = [
        row for row in fixed_rows
        if row["source_type"] == "local_semireal" and row["mode"] == "full" and row["probe_layer"] == 28
    ]
    local_answer_full = [
        row for row in answer_rows
        if row["source_type"] == "local_semireal" and row["mode"] == "full" and row["probe_layer"] == 28
    ]
    external_full = [
        row for row in answer_rows
        if row["source_type"] == "external_benchmark" and row["mode"] == "full" and row["probe_layer"] == 28
    ]
    external_disabled = [
        row for row in answer_rows
        if row["source_type"] == "external_benchmark" and row["mode"] == "adapter_disabled" and row["probe_layer"] == 28
    ]
    disabled_by_task = {row["task_name"]: row for row in external_disabled}
    external_improvements = [
        {
            "task_name": row["task_name"],
            "full_accuracy": row["accuracy"],
            "adapter_disabled_accuracy": disabled_by_task.get(row["task_name"], {}).get("accuracy", 0.0),
            "improvement": row["accuracy"] - disabled_by_task.get(row["task_name"], {}).get("accuracy", 0.0),
        }
        for row in external_full
    ]
    drops = {row["mode"]: row["absolute_drop"] for row in ablation_rows}
    external_wins = sum(row["improvement"] >= 0.05 for row in external_improvements)
    stage_gates = {
        "qwen_frozen": training.qwen_trainable_parameter_count == 0
        and training.qwen_gradients_present == 0
        and training.fingerprint_unchanged,
        "no_engineering_failures": not failure_cases,
        "local_fixed_centroid_accuracy": _mean(local_fixed_full, lambda _row: True) >= 0.55,
        "local_answer_option_accuracy": _mean(local_answer_full, lambda _row: True) >= 0.65,
        "local_surface_flip": _mean(local_fixed_full, lambda _row: True, field="surface_group_flip_accuracy") >= 0.50,
        "path_drop_any": (
            drops.get("no_memory_path", 0.0) >= 0.10
            or drops.get("no_state_path", 0.0) >= 0.08
            or drops.get("no_rule_path", 0.0) >= 0.08
        ),
        "external_answer_option_evidence": external_wins >= min(2, len(external_improvements)),
        "wrong_context_external": (
            _mean(
                answer_rows,
                lambda row: row["source_type"] == "external_benchmark"
                and row["mode"] == "full"
                and row["probe_layer"] == 28,
            )
            - _mean(
                answer_rows,
                lambda row: row["source_type"] == "external_benchmark"
                and row["mode"] == "wrong_context"
                and row["probe_layer"] == 28,
            )
        )
        >= 0.05,
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "context_grounding_mode": "grounded_v1",
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "training_steps": training_steps,
        "max_length": max_length,
        "external_max_length": external_max_length,
        "local_fixed_centroid_accuracy": _mean(local_fixed_full, lambda _row: True),
        "local_answer_option_accuracy": _mean(local_answer_full, lambda _row: True),
        "local_surface_group_flip_accuracy": _mean(local_fixed_full, lambda _row: True, field="surface_group_flip_accuracy"),
        "external_improvements": external_improvements,
        "external_truncation_count": len(truncation_cases),
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage27b_rerun": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(
        output_path / "grounded_context_manifest.json",
        {
            "local_semireal": local_real_task_manifest(local_records),
            "external_benchmark": external_manifest,
            "context_grounding_mode": "grounded_v1",
        },
    )
    training_row = asdict(training)
    losses = training_row.pop("losses")
    _json_dump(output_path / "training_runs.json", [training_row])
    _write_csv(output_path / "loss_curves.csv", [{"seed": seed, **row} for row in losses])
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "answer_option_metrics.csv", answer_rows)
    _write_csv(output_path / "ablation_drop.csv", ablation_rows)
    _write_csv(
        output_path / "wrong_context.csv",
        [
            row for row in answer_rows
            if row["mode"] in {"wrong_context", "adapter_disabled", "empty_context"}
        ],
    )
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    return summary
