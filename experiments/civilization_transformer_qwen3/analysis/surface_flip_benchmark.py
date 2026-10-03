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

from experiments.civilization_transformer_torch.analysis.codebook import build_logic_codebook_train_test
from experiments.civilization_transformer_torch.analysis.dataset import (
    LOGIC_LABELS,
    PATH_DEPENDENCY_SCENARIOS,
    PATH_STRESS_PROFILES,
    LogicSample,
    build_path_dependency_datasets,
)

from ..backend import Qwen3Backend
from .adapter_benchmark import (
    DEFAULT_MODEL_PATH,
    EVALUATION_MODES,
    MEMORY_SCENARIOS,
    RULE_SCENARIOS,
    STATE_SCENARIOS,
    _mean,
    split_dependency_samples,
)
from .multilayer_adapter_benchmark import (
    PROBE_LAYERS,
    _collect_multilayer_vectors,
    _language_preservation_rows,
)
from .surface_group_training import (
    Qwen3SurfaceFlipTrainingResult,
    SurfaceGroupLossWeights,
    run_qwen3_surface_flip_training,
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


def _training_result_row(result: Qwen3SurfaceFlipTrainingResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("losses")
    return row


def _surface_group_metrics(
    samples: list[LogicSample],
    vectors: np.ndarray,
    predictions: list[str],
    centroids: dict[str, np.ndarray],
) -> dict[str, float | int]:
    groups: dict[str, list[tuple[LogicSample, str, np.ndarray]]] = {}
    for sample, prediction, vector in zip(samples, predictions, vectors, strict=True):
        groups.setdefault(sample.surface_group_id, []).append((sample, prediction, vector))
    complete = 0
    one_error = 0
    multi_error = 0
    member_correct = 0
    margins = []
    separations = []
    for rows in groups.values():
        errors = sum(prediction != sample.label for sample, prediction, _vector in rows)
        member_correct += len(rows) - errors
        if errors == 0 and len(rows) > 1:
            complete += 1
        elif errors == 1:
            one_error += 1
        else:
            multi_error += 1
        normalized = []
        for sample, _prediction, vector in rows:
            distances = {
                label: float(np.linalg.norm(vector - centroid))
                for label, centroid in centroids.items()
            }
            true_distance = distances[sample.label]
            nearest_wrong = min(distance for label, distance in distances.items() if label != sample.label)
            margins.append(nearest_wrong - true_distance)
            norm = np.linalg.norm(vector)
            normalized.append(vector / max(norm, 1e-12))
        for left_index, left in enumerate(normalized):
            for right in normalized[left_index + 1 :]:
                separations.append(float(1.0 - np.dot(left, right)))
    group_count = len(groups)
    member_count = len(samples)
    return {
        "surface_group_total": group_count,
        "complete_group_count": complete,
        "one_error_group_count": one_error,
        "multi_error_group_count": multi_error,
        "surface_group_flip_accuracy": complete / group_count if group_count else 0.0,
        "group_member_accuracy": member_correct / member_count if member_count else 0.0,
        "group_min_target_margin": min(margins) if margins else 0.0,
        "group_context_separation": float(np.mean(separations)) if separations else 0.0,
    }


def _load_stage25_baseline() -> dict[str, Any]:
    path = Path("experiments/civilization_transformer_qwen3/artifacts/multilayer_adapter")
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {"available": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    flip_rows = []
    flip_path = path / "surface_group_flips.csv"
    if flip_path.exists():
        with flip_path.open(encoding="utf-8") as handle:
            flip_rows = list(csv.DictReader(handle))
    final = [
        float(row["surface_group_flip_accuracy"])
        for row in flip_rows
        if row["config"] == "dual_16_24"
        and row["mode"] == "full"
        and int(row["probe_layer"]) == 28
    ]
    return {
        "available": True,
        "surface_group_flip_accuracy": float(np.mean(final)) if final else 0.0,
        "main_final_accuracy": summary.get("main_final_accuracy"),
        "main_best_middle_accuracy": summary.get("main_best_middle_accuracy"),
        "source": str(summary_path),
    }


def run_qwen3_surface_flip_repair(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/surface_flip_repair/round_01",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 30,
    train_per_label: int = 18,
    stress_profiles: tuple[str, ...] = PATH_STRESS_PROFILES,
    training_steps: int = 140,
    max_length: int = 64,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
    loss_weights: SurfaceGroupLossWeights | None = None,
    scenario_weights: dict[str, float] | None = None,
    training_mode: str = "surface_group_flip_alignment",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    checkpoint_path = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    probes = PROBE_LAYERS["dual_16_24"]
    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    generation_sanity: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []

    for run_index, seed in enumerate(seeds, start=1):
        print(f"qwen3_surface_flip_training seed={seed} run={run_index}/{len(seeds)}", flush=True)
        profile_datasets: dict[str, dict[str, list[LogicSample]]] = {}
        train_samples: list[LogicSample] = []
        test_samples: dict[str, dict[str, list[LogicSample]]] = {}
        for profile in stress_profiles:
            datasets, _ = build_path_dependency_datasets(
                samples_per_label=samples_per_label,
                max_seq_len=max_length,
                seed=seed,
                stress_profile=profile,
            )
            profile_datasets[profile] = datasets
            test_samples[profile] = {}
            for scenario, samples in datasets.items():
                train, test = split_dependency_samples(
                    samples,
                    train_groups=train_per_label,
                    seed=seed,
                )
                train_samples.extend(train)
                test_samples[profile][scenario] = test
        run_started = time.perf_counter()
        model, _heads, training = run_qwen3_surface_flip_training(
            backend=backend,
            train_samples=train_samples,
            output_dir=checkpoint_path,
            target_layers=(16, 24),
            seed=seed,
            steps=training_steps,
            max_length=max_length,
            loss_weights=loss_weights,
            scenario_weights=scenario_weights,
            training_mode=training_mode,
        )
        training_rows.append(_training_result_row(training))
        for loss in training.losses:
            loss_rows.append({"seed": seed, **loss})
        consistency_rows.append(
            {
                "seed": seed,
                "initial_cross_layer_flip_consistency_loss": (
                    training.losses[0]["cross_layer_flip_consistency_loss"]
                ),
                "final_cross_layer_flip_consistency_loss": (
                    training.losses[-1]["cross_layer_flip_consistency_loss"]
                ),
                "decreased": training.loss_decreased["cross_layer_flip_consistency_loss"],
            }
        )

        reference_train: dict[str, tuple[dict[int, np.ndarray], list[LogicSample], dict[str, slice]]] = {}
        for mode in evaluation_modes:
            for profile in stress_profiles:
                print(
                    f"qwen3_surface_flip_evaluation seed={seed} mode={mode} profile={profile}",
                    flush=True,
                )
                profile_train: list[LogicSample] = []
                train_slices: dict[str, slice] = {}
                for scenario, scenario_samples in profile_datasets[profile].items():
                    train, _ = split_dependency_samples(
                        scenario_samples,
                        train_groups=train_per_label,
                        seed=seed,
                    )
                    start = len(profile_train)
                    profile_train.extend(train)
                    train_slices[scenario] = slice(start, len(profile_train))
                current_train_vectors, current_train_traces, train_failures = _collect_multilayer_vectors(
                    backend,
                    model,
                    profile_train,
                    mode,
                    max_length,
                    evaluation_batch_size,
                    probes,
                )
                if mode == "full":
                    reference_train[profile] = (
                        current_train_vectors,
                        profile_train,
                        train_slices,
                    )
                if profile not in reference_train:
                    raise RuntimeError("full mode must run before ablations")
                train_vectors, fixed_train_samples, fixed_train_slices = reference_train[profile]
                trace_rows.extend(
                    {"seed": seed, "profile": profile, "scenario": "all", "split": "train", **row}
                    for row in current_train_traces
                )
                failure_cases.extend(
                    {"seed": seed, "profile": profile, "split": "train", **row}
                    for row in train_failures
                )
                for scenario, scenario_test in test_samples[profile].items():
                    test_vectors, test_traces, test_failures = _collect_multilayer_vectors(
                        backend,
                        model,
                        scenario_test,
                        mode,
                        max_length,
                        evaluation_batch_size,
                        probes,
                    )
                    scenario_slice = fixed_train_slices[scenario]
                    scenario_train = fixed_train_samples[scenario_slice]
                    train_labels = np.array([sample.label for sample in scenario_train])
                    test_labels = np.array([sample.label for sample in scenario_test])
                    for probe in probes:
                        codebook = build_logic_codebook_train_test(
                            train_vectors[probe][scenario_slice],
                            train_labels,
                            test_vectors[probe],
                            test_labels,
                        )
                        predictions = [
                            result.predicted_label
                            for result in codebook.nearest_neighbors
                        ]
                        row = {
                            "seed": seed,
                            "stress_profile": profile,
                            "mode": mode,
                            "scenario": scenario,
                            "probe_layer": probe,
                            "accuracy": codebook.nearest_neighbor_accuracy,
                            "macro_accuracy": codebook.macro_accuracy,
                            "ari": codebook.adjusted_rand_score,
                            "most_confused_pair": codebook.easiest_confusion_pair,
                            "hidden_norm": float(np.linalg.norm(test_vectors[probe], axis=1).mean()),
                        }
                        probe_rows.append(row)
                        if probe == 28:
                            path_rows.append(row)
                        group_rows.append(
                            {
                                **{key: row[key] for key in (
                                    "seed",
                                    "stress_profile",
                                    "mode",
                                    "scenario",
                                    "probe_layer",
                                )},
                                **_surface_group_metrics(
                                    scenario_test,
                                    test_vectors[probe],
                                    predictions,
                                    codebook.centroids,
                                ),
                            }
                        )
                    trace_rows.extend(
                        {"seed": seed, "profile": profile, "scenario": scenario, "split": "test", **row}
                        for row in test_traces
                    )
                    failure_cases.extend(
                        {"seed": seed, "profile": profile, "scenario": scenario, "split": "test", **row}
                        for row in test_failures
                    )

        preservation, generation = _language_preservation_rows(
            backend,
            model,
            "dual_16_24",
            seed,
            max_length,
            evaluation_batch_size,
        )
        language_rows.extend(preservation)
        generation_sanity.append(generation)
        resources.append(
            {
                "seed": seed,
                "seconds": time.perf_counter() - run_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": (
                    torch.mps.current_allocated_memory()
                    if backend.device.type == "mps"
                    else 0
                ),
            }
        )
        del model, _heads
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    drop_rows = []
    for mode in ("no_memory_path", "no_state_path", "no_rule_path", "adapter_disabled"):
        relevant = (
            MEMORY_SCENARIOS
            if mode == "no_memory_path"
            else STATE_SCENARIOS
            if mode == "no_state_path"
            else RULE_SCENARIOS
            if mode == "no_rule_path"
            else set(PATH_DEPENDENCY_SCENARIOS)
        )
        full = _mean(path_rows, lambda row: row["mode"] == "full" and row["scenario"] in relevant)
        ablated = _mean(path_rows, lambda row, current=mode: row["mode"] == current and row["scenario"] in relevant)
        drop_rows.append(
            {
                "mode": mode,
                "full_accuracy": full,
                "ablated_accuracy": ablated,
                "absolute_drop": full - ablated,
            }
        )
    final_full = [row for row in path_rows if row["mode"] == "full"]
    probe_full = [row for row in probe_rows if row["mode"] == "full"]
    final_groups = [row for row in group_rows if row["mode"] == "full" and row["probe_layer"] == 28]
    final_seed_means = [
        _mean(final_full, lambda row, current=seed: row["seed"] == current)
        for seed in seeds
    ]
    flip_by_scenario = {
        scenario: _mean(
            final_groups,
            lambda row, current=scenario: row["scenario"] == current,
            field="surface_group_flip_accuracy",
        )
        for scenario in PATH_DEPENDENCY_SCENARIOS
    }
    flip_by_profile = {
        profile: _mean(
            final_groups,
            lambda row, current=profile: row["stress_profile"] == current,
            field="surface_group_flip_accuracy",
        )
        for profile in stress_profiles
    }
    flip_by_seed = {
        seed: _mean(
            final_groups,
            lambda row, current=seed: row["seed"] == current,
            field="surface_group_flip_accuracy",
        )
        for seed in seeds
    }
    main_drops = {row["mode"]: row["absolute_drop"] for row in drop_rows}
    full_traces = [row for row in trace_rows if row["mode"] == "full"]
    zero_or_disabled = [
        row for row in trace_rows
        if row["mode"] in {"zero_scale", "adapter_disabled"}
    ]
    best_middle_accuracy = max(
        _mean(probe_full, lambda row, current=probe: row["probe_layer"] == current)
        for probe in probes
        if probe != 28
    )
    final_accuracy = _mean(final_full, lambda _row: True)
    flip_accuracy = _mean(
        final_groups,
        lambda _row: True,
        field="surface_group_flip_accuracy",
    )
    member_accuracy = _mean(
        final_groups,
        lambda _row: True,
        field="group_member_accuracy",
    )
    training_loss_gate = all(
        row["loss_decreased"].get("total_loss", False)
        and row["loss_decreased"].get("group_all_correct_loss", False)
        for row in training_rows
    )
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            for row in training_rows
        ),
        "adapter_parameter_limit": all(row["adapter_parameter_count"] < 4_000_000 for row in training_rows),
        "training_losses_decreased": training_loss_gate,
        "no_engineering_failures": not failure_cases,
        "zero_and_disabled_equivalent": bool(zero_or_disabled)
        and all(row["delta_norm"] == 0.0 and row["hidden_norm_ratio"] == 1.0 for row in zero_or_disabled),
        "surface_group_flip_accuracy": flip_accuracy >= 0.90,
        "surface_invariant_label_flip": flip_by_scenario["surface_invariant_label_flip"] >= 0.90,
        "rule_required_priority": flip_by_scenario["rule_required_priority"] >= 0.90,
        "profile_flip_accuracy": all(value >= 0.88 for value in flip_by_profile.values()),
        "seed_flip_accuracy": all(value >= 0.85 for value in flip_by_seed.values()),
        "group_member_accuracy": member_accuracy >= 0.97,
        "best_middle_accuracy": best_middle_accuracy >= 0.95,
        "final_layer_accuracy": final_accuracy >= 0.75,
        "memory_path_drop": main_drops.get("no_memory_path", 0.0) >= 0.20,
        "state_path_drop": main_drops.get("no_state_path", 0.0) >= 0.15,
        "rule_path_drop": main_drops.get("no_rule_path", 0.0) >= 0.15,
        "wrong_context_control": _mean(path_rows, lambda row: row["mode"] == "wrong_context") <= 0.30,
        "adapter_disabled_control": _mean(path_rows, lambda row: row["mode"] == "adapter_disabled") <= 0.45,
        "noisy_profile_accuracy": _mean(final_full, lambda row: row["stress_profile"] == "noisy_context_v1") >= 0.70,
        "conflicting_profile_accuracy": _mean(final_full, lambda row: row["stress_profile"] == "conflicting_context_v1") >= 0.70,
        "seed_stability": len(final_seed_means) > 1 and statistics.pstdev(final_seed_means) <= 0.12,
        "hidden_norm_limit": bool(full_traces) and max(row["hidden_norm_ratio"] for row in full_traces) <= 2.0,
        "attention_normalized": bool(full_traces)
        and all(row["memory_attention_ok"] and row["rule_attention_ok"] for row in full_traces),
        "language_preservation": all(
            (
                _mean(
                    language_rows,
                    lambda row, variant=variant: row["mode"] == "adapter_disabled" and row["variant"] == variant,
                )
                - _mean(
                    language_rows,
                    lambda row, variant=variant: row["mode"] == "empty_context" and row["variant"] == variant,
                )
            )
            <= 0.05
            for variant in ("canonical", "synonym")
        ),
        "generation_sanity": bool(generation_sanity)
        and all(row["nonempty"] and not row["hook_leak"] for row in generation_sanity),
    }
    baseline = _load_stage25_baseline()
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "target_layers": [16, 24],
        "probe_layers": list(probes),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "stress_profiles": list(stress_profiles),
        "training_steps": training_steps,
        "training_mode": training_mode,
        "num_training_runs": len(training_rows),
        "stage25_baseline": baseline,
        "final_layer_accuracy": final_accuracy,
        "best_middle_accuracy": best_middle_accuracy,
        "surface_group_flip_accuracy": flip_accuracy,
        "surface_group_member_accuracy": member_accuracy,
        "flip_by_scenario": flip_by_scenario,
        "flip_by_profile": flip_by_profile,
        "flip_by_seed": flip_by_seed,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_real_task_migration": all(stage_gates.values()),
        "centroid_reference": "full_context_train_only",
        "runtime_seconds": time.perf_counter() - started,
        "backend_runtime_trace": backend.runtime_trace,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "generation_sanity": generation_sanity,
    }
    scenario_rows = [
        {
            "scenario": scenario,
            "stage25_baseline": baseline.get("surface_group_flip_accuracy", 0.0),
            "repaired_flip_accuracy": accuracy,
        }
        for scenario, accuracy in flip_by_scenario.items()
    ]
    profile_rows = [
        {"stress_profile": profile, "flip_accuracy": accuracy}
        for profile, accuracy in flip_by_profile.items()
    ]
    seed_rows = [
        {"seed": seed, "flip_accuracy": accuracy, "final_accuracy": final_seed_means[index]}
        for index, (seed, accuracy) in enumerate(flip_by_seed.items())
    ]
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "surface_group_metrics.csv", group_rows)
    _write_csv(output_path / "group_failure_distribution.csv", [
        {
            "seed": row["seed"],
            "stress_profile": row["stress_profile"],
            "scenario": row["scenario"],
            "probe_layer": row["probe_layer"],
            "complete_group_count": row["complete_group_count"],
            "one_error_group_count": row["one_error_group_count"],
            "multi_error_group_count": row["multi_error_group_count"],
        }
        for row in group_rows
        if row["mode"] == "full"
    ])
    _write_csv(output_path / "scenario_flip_comparison.csv", scenario_rows)
    _write_csv(output_path / "profile_flip_comparison.csv", profile_rows)
    _write_csv(output_path / "seed_stability.csv", seed_rows)
    _write_csv(output_path / "layer_probe_metrics.csv", probe_rows)
    _write_csv(output_path / "path_metrics.csv", path_rows)
    _write_csv(output_path / "ablation_drop.csv", drop_rows)
    _write_csv(output_path / "cross_layer_flip_consistency.csv", consistency_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _write_csv(output_path / "language_preservation.csv", language_rows)
    _json_dump(output_path / "resource_usage.json", resources)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    return summary
