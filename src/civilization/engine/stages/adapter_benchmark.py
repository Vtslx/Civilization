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

from civilization.research.torch_line.analysis.codebook import build_logic_codebook_train_test
from civilization.research.torch_line.analysis.dataset import (
    LOGIC_LABELS,
    PATH_DEPENDENCY_SCENARIOS,
    PATH_STRESS_PROFILES,
    LogicSample,
    build_logic_variant_datasets,
    build_path_dependency_datasets,
)
from civilization.research.torch_line.model import CivilizationAblationConfig

from ..adapter import CivilizationAdapterConfig
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import (
    Qwen3AdapterTrainingResult,
    run_qwen3_adapter_training,
    split_dependency_samples,
)
from .hidden_states import last_non_padding_pool


from civilization.engine.model_paths import DEFAULT_MODEL_PATH
EVALUATION_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_state_path",
    "no_rule_path",
    "empty_context",
    "wrong_context",
)
MEMORY_SCENARIOS = {
    "memory_required_two_hop",
    "memory_conflict_resolution",
    "counterfactual_memory_swap",
    "surface_invariant_label_flip",
}
STATE_SCENARIOS = {
    "state_required_disambiguation",
    "memory_conflict_resolution",
    "surface_invariant_label_flip",
}
RULE_SCENARIOS = {
    "rule_required_priority",
    "memory_conflict_resolution",
    "surface_invariant_label_flip",
}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mode_arguments(mode: str) -> tuple[CivilizationAblationConfig | None, bool, bool, str]:
    if mode == "full":
        return None, True, False, "full"
    if mode == "adapter_disabled":
        return None, False, False, "full"
    if mode == "zero_scale":
        return None, True, True, "full"
    if mode == "no_memory_path":
        return CivilizationAblationConfig(use_memory_path=False), True, False, "full"
    if mode == "no_state_path":
        return CivilizationAblationConfig(use_state_path=False), True, False, "full"
    if mode == "no_rule_path":
        return CivilizationAblationConfig(use_rule_path=False), True, False, "full"
    if mode == "empty_context":
        return None, True, False, "empty"
    if mode == "wrong_context":
        return None, True, False, "wrong"
    raise ValueError(f"unknown adapter evaluation mode: {mode}")


def _collect_adapter_vectors(
    backend: Qwen3Backend,
    model,
    samples: list[LogicSample],
    mode: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    ablation, enabled, force_zero, context_mode = _mode_arguments(mode)
    context_encoder = FrozenQwenContextEncoder(backend)
    vectors: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        encoded, truncations = backend.encode(
            [qwen_text_for_sample(sample) for sample in batch],
            max_length=max_length,
        )
        if truncations:
            raise ValueError(f"core dependency text truncated in mode={mode}")
        encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
        context = context_encoder.build_context(
            batch,
            encoded["attention_mask"],
            ablation_config=ablation,
            adapter_enabled=enabled,
            force_zero_scale=force_zero,
            context_mode=context_mode,
        )
        with torch.no_grad():
            output = model(encoded, context)
            # Transformers records each decoder input before the layer call.
            # A hook on decoder N first appears in hidden_states[N + 2];
            # the final decoder is represented by the normalized final state.
            representation_layer = min(model.adapter.config.target_layer + 2, 28)
            pooled = last_non_padding_pool(
                output.hidden_states[representation_layer],
                output.attention_mask,
            )
        if not torch.isfinite(pooled).all():
            failures.append({"mode": mode, "batch_start": start, "type": "nan_inf"})
        vectors.append(pooled.float().cpu())
        trace = output.trace
        memory_attention_ok = (
            trace.memory_attention.shape[-1] == 0
            or torch.allclose(
                trace.memory_attention.sum(dim=-1),
                torch.ones_like(trace.memory_attention.sum(dim=-1)),
                atol=1e-3,
            )
        )
        rule_attention_ok = (
            trace.rule_attention.shape[-1] == 0
            or torch.allclose(
                trace.rule_attention.sum(dim=-1),
                torch.ones_like(trace.rule_attention.sum(dim=-1)),
                atol=1e-3,
            )
        )
        traces.append(
            {
                "mode": mode,
                "batch_start": start,
                "batch_size": len(batch),
                "target_layer": trace.target_layer,
                "hidden_norm_ratio": trace.hidden_norm_ratio,
                "delta_norm": trace.delta_norm,
                "memory_contribution_norm": trace.memory_contribution_norm,
                "state_contribution_norm": trace.state_contribution_norm,
                "rule_contribution_norm": trace.rule_contribution_norm,
                "residual_scale": trace.residual_scale,
                "memory_attention_ok": memory_attention_ok,
                "rule_attention_ok": rule_attention_ok,
            }
        )
        del output, pooled, context
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    return torch.cat(vectors, dim=0).numpy(), traces, failures


def _surface_flip_accuracy(samples: list[LogicSample], predictions: list[str]) -> tuple[int, int, float]:
    groups: dict[str, list[bool]] = {}
    for sample, prediction in zip(samples, predictions, strict=True):
        groups.setdefault(sample.surface_group_id, []).append(prediction == sample.label)
    total = len(groups)
    correct = sum(all(values) and len(values) > 1 for values in groups.values())
    return total, correct, correct / total if total else 0.0


def _mean(rows: list[dict[str, Any]], predicate, field: str = "accuracy") -> float:
    values = [float(row[field]) for row in rows if predicate(row)]
    return float(np.mean(values)) if values else 0.0


def _training_result_row(result: Qwen3AdapterTrainingResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("losses")
    return row


def _logic_split(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        rows = [sample for sample in samples if sample.label == label]
        train.extend(rows[:train_per_label])
        test.extend(rows[train_per_label:])
    return train, test


def _language_preservation_rows(
    backend: Qwen3Backend,
    model,
    layer: int,
    seed: int,
    max_length: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets, _ = build_logic_variant_datasets(
        samples_per_label=4,
        max_seq_len=32,
        seed=seed,
        template_bank="expanded_v1",
    )
    canonical_train, _ = _logic_split(datasets["canonical"], train_per_label=2)
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str], np.ndarray] = {}
    for mode in ("adapter_disabled", "empty_context"):
        train_vectors, _traces, _failures = _collect_adapter_vectors(
            backend,
            model,
            canonical_train,
            mode,
            max_length,
            batch_size,
        )
        train_labels = np.array([sample.label for sample in canonical_train])
        for variant, samples in datasets.items():
            _, test = _logic_split(samples, train_per_label=2)
            test_vectors, _test_traces, _test_failures = _collect_adapter_vectors(
                backend,
                model,
                test,
                mode,
                max_length,
                batch_size,
            )
            vectors[(mode, variant)] = test_vectors
            codebook = build_logic_codebook_train_test(
                train_vectors,
                train_labels,
                test_vectors,
                np.array([sample.label for sample in test]),
            )
            rows.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "mode": mode,
                    "variant": variant,
                    "accuracy": codebook.nearest_neighbor_accuracy,
                    "macro_accuracy": codebook.macro_accuracy,
                    "hidden_norm": float(np.linalg.norm(test_vectors, axis=1).mean()),
                }
            )
    for variant in datasets:
        baseline = vectors[("adapter_disabled", variant)]
        adapted = vectors[("empty_context", variant)]
        cosine = np.sum(baseline * adapted, axis=1) / (
            np.linalg.norm(baseline, axis=1) * np.linalg.norm(adapted, axis=1)
        ).clip(min=1e-8)
        rows.append(
            {
                "layer": layer,
                "seed": seed,
                "mode": "drift",
                "variant": variant,
                "accuracy": "",
                "macro_accuracy": "",
                "hidden_norm": "",
                "mean_cosine_similarity": float(cosine.mean()),
            }
        )

    prompt = "A causes B and B causes C. State the resulting relation."
    encoded, truncations = backend.encode([prompt], max_length=max_length)
    if truncations:
        raise ValueError("generation sanity prompt was truncated")
    context_encoder = FrozenQwenContextEncoder(backend)
    dummy = datasets["canonical"][0]
    context = context_encoder.build_context(
        [dummy],
        encoded["attention_mask"],
        context_mode="empty",
    )
    generated = model.generate(encoded, context, max_new_tokens=4)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    text = backend.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return rows, {
        "layer": layer,
        "seed": seed,
        "generated_tokens": int(new_tokens.shape[0]),
        "text": text,
        "nonempty": bool(text),
        "hook_leak": model.active_hook_count != 0,
    }


def run_qwen3_civilization_adapter(
    output_dir: str | Path = "artifacts/civilization/civilization_adapter",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    layers: tuple[int, ...] = (2, 16, 27),
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 20,
    train_per_label: int = 12,
    stress_profiles: tuple[str, ...] = PATH_STRESS_PROFILES,
    training_steps: int = 80,
    max_length: int = 64,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 5,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    if not 0 < train_per_label < samples_per_label:
        raise ValueError("train_per_label must leave held-out surface groups")
    output_path = Path(output_dir)
    checkpoint_path = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    generation_sanity: list[dict[str, Any]] = []

    for layer in layers:
        for seed in seeds:
            print(
                f"qwen3_adapter_training layer={layer} seed={seed} "
                f"run={len(training_rows) + 1}/{len(layers) * len(seeds)}",
                flush=True,
            )
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
                    train, test = split_dependency_samples(samples, train_groups=train_per_label, seed=seed)
                    train_samples.extend(train)
                    test_samples[profile][scenario] = test

            run_started = time.perf_counter()
            model, _heads, training = run_qwen3_adapter_training(
                backend=backend,
                train_samples=train_samples,
                adapter_config=CivilizationAdapterConfig(target_layer=layer),
                output_dir=checkpoint_path,
                seed=seed,
                steps=training_steps,
                max_length=max_length,
            )
            training_rows.append(_training_result_row(training))
            for loss in training.losses:
                loss_rows.append({"layer": layer, "seed": seed, **loss})

            reference_train_vectors: dict[str, tuple[np.ndarray, list[LogicSample], dict[str, slice]]] = {}
            for mode in evaluation_modes:
                for profile in stress_profiles:
                    print(
                        f"qwen3_adapter_evaluation layer={layer} seed={seed} "
                        f"mode={mode} profile={profile}",
                        flush=True,
                    )
                    profile_train: list[LogicSample] = []
                    train_slices: dict[str, slice] = {}
                    for scenario_samples in profile_datasets[profile].values():
                        train, _ = split_dependency_samples(
                            scenario_samples,
                            train_groups=train_per_label,
                            seed=seed,
                        )
                        start = len(profile_train)
                        profile_train.extend(train)
                        train_slices[scenario_samples[0].variant] = slice(start, len(profile_train))
                    train_vectors, train_traces, train_failures = _collect_adapter_vectors(
                        backend,
                        model,
                        profile_train,
                        mode,
                        max_length,
                        evaluation_batch_size,
                    )
                    if mode == "full":
                        reference_train_vectors[profile] = (
                            train_vectors,
                            profile_train,
                            train_slices,
                        )
                    elif profile in reference_train_vectors:
                        train_vectors, profile_train, train_slices = reference_train_vectors[profile]
                    else:
                        raise RuntimeError("full mode must be evaluated before ablation modes")
                    trace_rows.extend(
                        {"layer": layer, "seed": seed, "profile": profile, "split": "train", **row}
                        for row in train_traces
                    )
                    failure_cases.extend(
                        {"layer": layer, "seed": seed, "profile": profile, **row}
                        for row in train_failures
                    )
                    for scenario, scenario_test in test_samples[profile].items():
                        scenario_slice = train_slices[scenario]
                        scenario_train = profile_train[scenario_slice]
                        scenario_train_vectors = train_vectors[scenario_slice]
                        test_vectors, test_traces, test_failures = _collect_adapter_vectors(
                            backend,
                            model,
                            scenario_test,
                            mode,
                            max_length,
                            evaluation_batch_size,
                        )
                        test_labels = np.array([sample.label for sample in scenario_test])
                        codebook = build_logic_codebook_train_test(
                            scenario_train_vectors,
                            np.array([sample.label for sample in scenario_train]),
                            test_vectors,
                            test_labels,
                        )
                        predictions = [result.predicted_label for result in codebook.nearest_neighbors]
                        flip_total, flip_correct, flip_accuracy = _surface_flip_accuracy(
                            scenario_test,
                            predictions,
                        )
                        path_rows.append(
                            {
                                "layer": layer,
                                "seed": seed,
                                "stress_profile": profile,
                                "mode": mode,
                                "scenario": scenario,
                                "accuracy": codebook.nearest_neighbor_accuracy,
                                "macro_accuracy": codebook.macro_accuracy,
                                "ari": codebook.adjusted_rand_score,
                                "most_confused_pair": codebook.easiest_confusion_pair,
                                "hidden_norm": float(np.linalg.norm(test_vectors, axis=1).mean()),
                            }
                        )
                        flip_rows.append(
                            {
                                "layer": layer,
                                "seed": seed,
                                "stress_profile": profile,
                                "mode": mode,
                                "scenario": scenario,
                                "surface_group_total": flip_total,
                                "surface_group_correct": flip_correct,
                                "surface_group_flip_accuracy": flip_accuracy,
                            }
                        )
                        trace_rows.extend(
                            {"layer": layer, "seed": seed, "profile": profile, "scenario": scenario, "split": "test", **row}
                            for row in test_traces
                        )
                        failure_cases.extend(
                            {"layer": layer, "seed": seed, "profile": profile, "scenario": scenario, **row}
                            for row in test_failures
                        )
            resources.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "seconds": time.perf_counter() - run_started,
                    "rss": psutil.Process().memory_info().rss,
                    "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
                }
            )
            if layer == 16:
                preservation, generation = _language_preservation_rows(
                    backend,
                    model,
                    layer,
                    seed,
                    max_length,
                    evaluation_batch_size,
                )
                language_rows.extend(preservation)
                generation_sanity.append(generation)
            del model, _heads
            if backend.device.type == "mps":
                torch.mps.empty_cache()

    layer_rows = []
    for layer in layers:
        full_values = [
            row["accuracy"] for row in path_rows if row["layer"] == layer and row["mode"] == "full"
        ]
        layer_rows.append(
            {
                "layer": layer,
                "accuracy_mean": float(np.mean(full_values)),
                "accuracy_std": float(np.std(full_values)),
                "accuracy_min": float(np.min(full_values)),
                "accuracy_max": float(np.max(full_values)),
            }
        )

    drop_rows = []
    for layer in layers:
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
            full = _mean(path_rows, lambda row: row["layer"] == layer and row["mode"] == "full" and row["scenario"] in relevant)
            ablated = _mean(path_rows, lambda row: row["layer"] == layer and row["mode"] == mode and row["scenario"] in relevant)
            drop_rows.append(
                {
                    "layer": layer,
                    "mode": mode,
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )

    main_rows = [row for row in path_rows if row["layer"] == 16]
    main_full = [row for row in main_rows if row["mode"] == "full"]
    main_seed_means = [
        _mean(main_full, lambda row, current=seed: row["seed"] == current)
        for seed in seeds
    ]
    main_drops = {
        row["mode"]: row["absolute_drop"]
        for row in drop_rows
        if row["layer"] == 16
    }
    main_flip_values = [
        row["surface_group_flip_accuracy"]
        for row in flip_rows
        if row["layer"] == 16 and row["mode"] == "full"
    ]
    full_traces = [row for row in trace_rows if row["layer"] == 16 and row["mode"] == "full"]
    zero_or_disabled_traces = [
        row for row in trace_rows
        if row["layer"] == 16 and row["mode"] in {"zero_scale", "adapter_disabled"}
    ]
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            for row in training_rows
        ),
        "adapter_parameter_limit": all(row["adapter_parameter_count"] < 2_000_000 for row in training_rows),
        "no_engineering_failures": not failure_cases,
        "zero_and_disabled_equivalent": bool(zero_or_disabled_traces)
        and all(row["delta_norm"] == 0.0 and row["hidden_norm_ratio"] == 1.0 for row in zero_or_disabled_traces),
        "full_dependency_accuracy": _mean(main_full, lambda _row: True) >= 0.65,
        "surface_group_flip_accuracy": bool(main_flip_values) and float(np.mean(main_flip_values)) >= 0.65,
        "memory_path_drop": main_drops.get("no_memory_path", 0.0) >= 0.15,
        "state_path_drop": main_drops.get("no_state_path", 0.0) >= 0.10,
        "rule_path_drop": main_drops.get("no_rule_path", 0.0) >= 0.10,
        "adapter_disabled_surface_leakage": _mean(main_rows, lambda row: row["mode"] == "adapter_disabled") <= 0.45,
        "noisy_profile_accuracy": _mean(main_full, lambda row: row["stress_profile"] == "noisy_context_v1") >= 0.55,
        "conflicting_profile_accuracy": _mean(main_full, lambda row: row["stress_profile"] == "conflicting_context_v1") >= 0.55,
        "seed_stability": len(main_seed_means) > 1 and statistics.pstdev(main_seed_means) <= 0.12,
        "hidden_norm_limit": bool(full_traces) and max(row["hidden_norm_ratio"] for row in full_traces) <= 2.0,
        "attention_normalized": bool(full_traces)
        and all(row["memory_attention_ok"] and row["rule_attention_ok"] for row in full_traces),
        "language_preservation": all(
            (
                _mean(
                    language_rows,
                    lambda row, variant=variant: row["mode"] == "adapter_disabled"
                    and row["variant"] == variant,
                )
                - _mean(
                    language_rows,
                    lambda row, variant=variant: row["mode"] == "empty_context"
                    and row["variant"] == variant,
                )
            )
            <= 0.05
            for variant in ("canonical", "synonym")
        ),
        "generation_sanity": bool(generation_sanity)
        and all(row["nonempty"] and not row["hook_leak"] for row in generation_sanity),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "layers": list(layers),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "stress_profiles": list(stress_profiles),
        "training_steps": training_steps,
        "num_training_runs": len(training_rows),
        "main_layer": 16,
        "main_layer_accuracy": _mean(main_full, lambda _row: True),
        "main_layer_seed_std": statistics.pstdev(main_seed_means) if len(main_seed_means) > 1 else 0.0,
        "main_layer_surface_flip_accuracy": float(np.mean(main_flip_values)) if main_flip_values else 0.0,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage25_planning": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "backend_runtime_trace": backend.runtime_trace,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "generation_sanity": generation_sanity,
    }

    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "layer_comparison.csv", layer_rows)
    _write_csv(output_path / "path_metrics.csv", path_rows)
    _write_csv(output_path / "ablation_drop.csv", drop_rows)
    _write_csv(output_path / "surface_group_flips.csv", flip_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _write_csv(output_path / "language_preservation.csv", language_rows)
    _json_dump(output_path / "resource_usage.json", resources)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    return summary
