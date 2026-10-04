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

from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import (
    DEFAULT_MODEL_PATH,
    EVALUATION_MODES,
    MEMORY_SCENARIOS,
    RULE_SCENARIOS,
    STATE_SCENARIOS,
    _logic_split,
    _mean,
    _mode_arguments,
    _surface_flip_accuracy,
)
from .adapter_training import split_dependency_samples
from .hidden_states import last_non_padding_pool
from .multilayer_adapter_training import (
    Qwen3MultiAdapterTrainingResult,
    run_qwen3_multilayer_adapter_training,
)


MULTILAYER_CONFIGS: dict[str, tuple[int, ...]] = {
    "single_16": (16,),
    "dual_16_24": (16, 24),
    "dual_2_16": (2, 16),
    "dual_16_27": (16, 27),
}

PROBE_LAYERS: dict[str, tuple[int, ...]] = {
    "single_16": (18,),
    "dual_16_24": (18, 26, 28),
    "dual_2_16": (4, 18, 28),
    "dual_16_27": (18, 28),
}


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


def _attention_sums_are_normalized_or_empty(attention: torch.Tensor) -> bool:
    if attention.shape[-1] == 0:
        return True
    sums = attention.sum(dim=-1)
    ones = torch.ones_like(sums)
    zeros = torch.zeros_like(sums)
    return bool(torch.all(torch.isclose(sums, ones, atol=1e-3) | torch.isclose(sums, zeros, atol=1e-3)).item())


def _training_result_row(result: Qwen3MultiAdapterTrainingResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("losses")
    return row


def _collect_multilayer_vectors(
    backend: Qwen3Backend,
    model,
    samples: list[LogicSample],
    mode: str,
    max_length: int,
    batch_size: int,
    probe_layers: tuple[int, ...],
    fail_on_truncation: bool = True,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    ablation, enabled, force_zero, context_mode = _mode_arguments(mode)
    context_encoder = FrozenQwenContextEncoder(backend)
    vectors: dict[int, list[torch.Tensor]] = {layer: [] for layer in probe_layers}
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        encoded, truncations = backend.encode(
            [qwen_text_for_sample(sample) for sample in batch],
            max_length=max_length,
        )
        if truncations:
            if fail_on_truncation:
                raise ValueError(f"core dependency text truncated in mode={mode}")
            failures.extend(
                {
                    "mode": mode,
                    "batch_start": start,
                    "type": "truncation",
                    **truncation,
                }
                for truncation in truncations
            )
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
            for layer in probe_layers:
                pooled = last_non_padding_pool(output.hidden_states[layer], output.attention_mask)
                if not torch.isfinite(pooled).all():
                    failures.append({"mode": mode, "batch_start": start, "probe_layer": layer, "type": "nan_inf"})
                vectors[layer].append(pooled.float().cpu())
        for target_layer, trace in output.traces.items():
            memory_attention_ok = _attention_sums_are_normalized_or_empty(trace.memory_attention)
            rule_attention_ok = _attention_sums_are_normalized_or_empty(trace.rule_attention)
            traces.append(
                {
                    "mode": mode,
                    "batch_start": start,
                    "batch_size": len(batch),
                    "target_layer": target_layer,
                    "hidden_norm_ratio": trace.hidden_norm_ratio,
                    "delta_norm": trace.delta_norm,
                    "memory_contribution_norm": trace.memory_contribution_norm,
                    "state_contribution_norm": trace.state_contribution_norm,
                    "rule_contribution_norm": trace.rule_contribution_norm,
                    "residual_scale": trace.residual_scale,
                    "memory_delta_norm": trace.memory_delta_norm,
                    "rule_delta_norm": trace.rule_delta_norm,
                    "state_delta_norm": trace.state_delta_norm,
                    "base_delta_norm": trace.base_delta_norm,
                    "memory_residual_scale": trace.memory_residual_scale,
                    "rule_residual_scale": trace.rule_residual_scale,
                    "path_dominance_ratio": trace.path_dominance_ratio,
                    "path_specific_adapter_version": trace.path_specific_adapter_version,
                    "memory_attention_ok": memory_attention_ok,
                    "rule_attention_ok": rule_attention_ok,
                }
            )
        del output, context
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    return {layer: torch.cat(parts, dim=0).numpy() for layer, parts in vectors.items()}, traces, failures


def _language_preservation_rows(
    backend: Qwen3Backend,
    model,
    config_name: str,
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
    probe_layer = max(PROBE_LAYERS[config_name])
    canonical_train, _ = _logic_split(datasets["canonical"], train_per_label=2)
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str], np.ndarray] = {}
    for mode in ("adapter_disabled", "empty_context"):
        train_vectors, _traces, _failures = _collect_multilayer_vectors(
            backend,
            model,
            canonical_train,
            mode,
            max_length,
            batch_size,
            (probe_layer,),
        )
        train_labels = np.array([sample.label for sample in canonical_train])
        for variant, samples in datasets.items():
            _, test = _logic_split(samples, train_per_label=2)
            test_vectors, _test_traces, _test_failures = _collect_multilayer_vectors(
                backend,
                model,
                test,
                mode,
                max_length,
                batch_size,
                (probe_layer,),
            )
            vectors[(mode, variant)] = test_vectors[probe_layer]
            codebook = build_logic_codebook_train_test(
                train_vectors[probe_layer],
                train_labels,
                test_vectors[probe_layer],
                np.array([sample.label for sample in test]),
            )
            rows.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "mode": mode,
                    "variant": variant,
                    "probe_layer": probe_layer,
                    "accuracy": codebook.nearest_neighbor_accuracy,
                    "macro_accuracy": codebook.macro_accuracy,
                    "hidden_norm": float(np.linalg.norm(test_vectors[probe_layer], axis=1).mean()),
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
                "config": config_name,
                "seed": seed,
                "mode": "drift",
                "variant": variant,
                "probe_layer": probe_layer,
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
    context = context_encoder.build_context(
        [datasets["canonical"][0]],
        encoded["attention_mask"],
        context_mode="empty",
    )
    generated = model.generate(encoded, context, max_new_tokens=4)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    text = backend.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return rows, {
        "config": config_name,
        "seed": seed,
        "generated_tokens": int(new_tokens.shape[0]),
        "text": text,
        "nonempty": bool(text),
        "hook_leak": model.active_hook_count != 0,
    }


def run_qwen3_multilayer_adapter(
    output_dir: str | Path = "artifacts/civilization/multilayer_adapter",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    configs: tuple[str, ...] = tuple(MULTILAYER_CONFIGS),
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 24,
    train_per_label: int = 14,
    stress_profiles: tuple[str, ...] = PATH_STRESS_PROFILES,
    training_steps: int = 100,
    max_length: int = 64,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    checkpoint_path = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    generation_sanity: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []

    for config_name in configs:
        if config_name not in MULTILAYER_CONFIGS:
            raise ValueError(f"unknown multilayer config: {config_name}")
        target_layers = MULTILAYER_CONFIGS[config_name]
        probes = PROBE_LAYERS[config_name]
        for seed in seeds:
            print(
                f"qwen3_multilayer_training config={config_name} seed={seed} "
                f"run={len(training_rows) + 1}/{len(configs) * len(seeds)}",
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
            model, _heads, training = run_qwen3_multilayer_adapter_training(
                backend=backend,
                train_samples=train_samples,
                config_name=config_name,
                target_layers=target_layers,
                output_dir=checkpoint_path,
                seed=seed,
                steps=training_steps,
                max_length=max_length,
            )
            training_rows.append(_training_result_row(training))
            for loss in training.losses:
                loss_rows.append({"config": config_name, "seed": seed, **loss})
            consistency_rows.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "initial_cross_layer_consistency_loss": training.initial_cross_layer_consistency_loss,
                    "final_cross_layer_consistency_loss": training.final_cross_layer_consistency_loss,
                    "cross_layer_consistency_decreased": training.cross_layer_consistency_decreased,
                }
            )

            reference_train_vectors: dict[str, tuple[dict[int, np.ndarray], list[LogicSample], dict[str, slice]]] = {}
            for mode in evaluation_modes:
                for profile in stress_profiles:
                    print(
                        f"qwen3_multilayer_evaluation config={config_name} seed={seed} "
                        f"mode={mode} profile={profile}",
                        flush=True,
                    )
                    profile_train: list[LogicSample] = []
                    train_slices: dict[str, slice] = {}
                    for scenario_samples in profile_datasets[profile].values():
                        train, _ = split_dependency_samples(scenario_samples, train_groups=train_per_label, seed=seed)
                        start = len(profile_train)
                        profile_train.extend(train)
                        train_slices[scenario_samples[0].variant] = slice(start, len(profile_train))
                    train_vectors, train_traces, train_failures = _collect_multilayer_vectors(
                        backend,
                        model,
                        profile_train,
                        mode,
                        max_length,
                        evaluation_batch_size,
                        probes,
                    )
                    if mode == "full":
                        reference_train_vectors[profile] = (train_vectors, profile_train, train_slices)
                    elif profile in reference_train_vectors:
                        train_vectors, profile_train, train_slices = reference_train_vectors[profile]
                    else:
                        raise RuntimeError("full mode must be evaluated before ablation modes")
                    trace_rows.extend(
                        {"config": config_name, "seed": seed, "profile": profile, "split": "train", **row}
                        for row in train_traces
                    )
                    failure_cases.extend(
                        {"config": config_name, "seed": seed, "profile": profile, **row}
                        for row in train_failures
                    )
                    for scenario, scenario_test in test_samples[profile].items():
                        scenario_slice = train_slices[scenario]
                        scenario_train = profile_train[scenario_slice]
                        test_vectors, test_traces, test_failures = _collect_multilayer_vectors(
                            backend,
                            model,
                            scenario_test,
                            mode,
                            max_length,
                            evaluation_batch_size,
                            probes,
                        )
                        test_labels = np.array([sample.label for sample in scenario_test])
                        for probe in probes:
                            scenario_train_vectors = train_vectors[probe][scenario_slice]
                            codebook = build_logic_codebook_train_test(
                                scenario_train_vectors,
                                np.array([sample.label for sample in scenario_train]),
                                test_vectors[probe],
                                test_labels,
                            )
                            predictions = [result.predicted_label for result in codebook.nearest_neighbors]
                            flip_total, flip_correct, flip_accuracy = _surface_flip_accuracy(scenario_test, predictions)
                            row = {
                                "config": config_name,
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
                            if probe == max(probes):
                                path_rows.append(row)
                            flip_rows.append(
                                {
                                    "config": config_name,
                                    "seed": seed,
                                    "stress_profile": profile,
                                    "mode": mode,
                                    "scenario": scenario,
                                    "probe_layer": probe,
                                    "surface_group_total": flip_total,
                                    "surface_group_correct": flip_correct,
                                    "surface_group_flip_accuracy": flip_accuracy,
                                }
                            )
                        trace_rows.extend(
                            {"config": config_name, "seed": seed, "profile": profile, "scenario": scenario, "split": "test", **row}
                            for row in test_traces
                        )
                        failure_cases.extend(
                            {"config": config_name, "seed": seed, "profile": profile, "scenario": scenario, **row}
                            for row in test_failures
                        )
            if config_name in {"single_16", "dual_16_24"}:
                preservation, generation = _language_preservation_rows(
                    backend,
                    model,
                    config_name,
                    seed,
                    max_length,
                    evaluation_batch_size,
                )
                language_rows.extend(preservation)
                generation_sanity.append(generation)
            resources.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "seconds": time.perf_counter() - run_started,
                    "rss": psutil.Process().memory_info().rss,
                    "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
                }
            )
            del model, _heads
            if backend.device.type == "mps":
                torch.mps.empty_cache()

    config_rows = []
    for config_name in configs:
        full_values = [row["accuracy"] for row in path_rows if row["config"] == config_name and row["mode"] == "full"]
        config_rows.append(
            {
                "config": config_name,
                "accuracy_mean": float(np.mean(full_values)),
                "accuracy_std": float(np.std(full_values)),
                "accuracy_min": float(np.min(full_values)),
                "accuracy_max": float(np.max(full_values)),
            }
        )
    drop_rows = []
    for config_name in configs:
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
            full = _mean(path_rows, lambda row, c=config_name: row["config"] == c and row["mode"] == "full" and row["scenario"] in relevant)
            ablated = _mean(path_rows, lambda row, c=config_name, m=mode: row["config"] == c and row["mode"] == m and row["scenario"] in relevant)
            drop_rows.append(
                {
                    "config": config_name,
                    "mode": mode,
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
    main_config = "dual_16_24"
    main_final = [row for row in probe_rows if row["config"] == main_config and row["probe_layer"] == 28]
    main_final_full = [row for row in main_final if row["mode"] == "full"]
    main_probe_full = [row for row in probe_rows if row["config"] == main_config and row["mode"] == "full"]
    best_middle_accuracy = max(
        _mean(main_probe_full, lambda row, probe=probe: row["probe_layer"] == probe)
        for probe in PROBE_LAYERS[main_config]
        if probe != 28
    )
    final_seed_means = [
        _mean(main_final_full, lambda row, current=seed: row["seed"] == current)
        for seed in seeds
    ]
    main_drops = {row["mode"]: row["absolute_drop"] for row in drop_rows if row["config"] == main_config}
    main_flip = [
        row["surface_group_flip_accuracy"]
        for row in flip_rows
        if row["config"] == main_config and row["mode"] == "full" and row["probe_layer"] == 28
    ]
    full_traces = [row for row in trace_rows if row["config"] == main_config and row["mode"] == "full"]
    zero_or_disabled = [
        row for row in trace_rows
        if row["config"] == main_config and row["mode"] in {"zero_scale", "adapter_disabled"}
    ]
    single16_final = _mean(
        probe_rows,
        lambda row: row["config"] == "single_16" and row["mode"] == "full" and row["probe_layer"] == 18,
    )
    dual_final = _mean(main_final_full, lambda _row: True)
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and row["fingerprint_unchanged"]
            for row in training_rows
        ),
        "adapter_parameter_limit": all(row["adapter_parameter_count"] < 4_000_000 for row in training_rows),
        "no_engineering_failures": not failure_cases,
        "zero_and_disabled_equivalent": bool(zero_or_disabled)
        and all(row["delta_norm"] == 0.0 and row["hidden_norm_ratio"] == 1.0 for row in zero_or_disabled),
        "best_middle_accuracy": best_middle_accuracy >= 0.95,
        "final_layer_accuracy": dual_final >= 0.75,
        "surface_group_flip_accuracy": bool(main_flip) and float(np.mean(main_flip)) >= 0.90,
        "memory_path_drop": main_drops.get("no_memory_path", 0.0) >= 0.20,
        "state_path_drop": main_drops.get("no_state_path", 0.0) >= 0.15,
        "rule_path_drop": main_drops.get("no_rule_path", 0.0) >= 0.15,
        "wrong_context_control": _mean(main_final, lambda row: row["mode"] == "wrong_context") <= 0.30,
        "adapter_disabled_control": _mean(main_final, lambda row: row["mode"] == "adapter_disabled") <= 0.45,
        "noisy_profile_accuracy": _mean(main_final_full, lambda row: row["stress_profile"] == "noisy_context_v1") >= 0.70,
        "conflicting_profile_accuracy": _mean(main_final_full, lambda row: row["stress_profile"] == "conflicting_context_v1") >= 0.70,
        "seed_stability": len(final_seed_means) > 1 and statistics.pstdev(final_seed_means) <= 0.12,
        "hidden_norm_limit": bool(full_traces) and max(row["hidden_norm_ratio"] for row in full_traces) <= 2.0,
        "attention_normalized": bool(full_traces)
        and all(row["memory_attention_ok"] and row["rule_attention_ok"] for row in full_traces),
        "language_preservation": all(
            (
                _mean(language_rows, lambda row, variant=variant: row["config"] == main_config and row["mode"] == "adapter_disabled" and row["variant"] == variant)
                - _mean(language_rows, lambda row, variant=variant: row["config"] == main_config and row["mode"] == "empty_context" and row["variant"] == variant)
            )
            <= 0.05
            for variant in ("canonical", "synonym")
        ),
        "generation_sanity": bool(generation_sanity)
        and all(row["nonempty"] and not row["hook_leak"] for row in generation_sanity if row["config"] == main_config),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "configs": list(configs),
        "config_layers": {name: list(MULTILAYER_CONFIGS[name]) for name in configs},
        "probe_layers": {name: list(PROBE_LAYERS[name]) for name in configs},
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "stress_profiles": list(stress_profiles),
        "training_steps": training_steps,
        "num_training_runs": len(training_rows),
        "main_config": main_config,
        "main_final_accuracy": dual_final,
        "main_best_middle_accuracy": best_middle_accuracy,
        "main_final_seed_std": statistics.pstdev(final_seed_means) if len(final_seed_means) > 1 else 0.0,
        "single16_accuracy": single16_final,
        "dual16_24_final_improves_single16": dual_final >= single16_final,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage26_planning": all(stage_gates.values()) and dual_final >= single16_final,
        "centroid_reference": "full_context_train_only",
        "runtime_seconds": time.perf_counter() - started,
        "backend_runtime_trace": backend.runtime_trace,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "generation_sanity": generation_sanity,
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "config_comparison.csv", config_rows)
    _write_csv(output_path / "layer_probe_metrics.csv", probe_rows)
    _write_csv(output_path / "path_metrics.csv", path_rows)
    _write_csv(output_path / "ablation_drop.csv", drop_rows)
    _write_csv(output_path / "cross_layer_consistency.csv", consistency_rows)
    _write_csv(output_path / "surface_group_flips.csv", flip_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _write_csv(output_path / "language_preservation.csv", language_rows)
    _json_dump(output_path / "resource_usage.json", resources)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    return summary
