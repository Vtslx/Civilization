from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..model import CivilizationAblationConfig, CivilizationTransformerTorch, TransformerConfigTorch
from .chain_alignment import split_by_label
from .codebook import build_logic_codebook_train_test
from .dataset import LOGIC_LABELS, PATH_DEPENDENCY_SCENARIOS, LogicSample, build_path_dependency_datasets
from .path_dependency_training import (
    PathDependencyLossConfig,
    contextual_vectors,
    run_path_dependency_training,
)
from .run_hard_logic_benchmark import HARD_MODEL_SIZES


PATH_DEPENDENCY_MODES: dict[str, tuple[CivilizationAblationConfig, PathDependencyLossConfig]] = {
    "full": (CivilizationAblationConfig(), PathDependencyLossConfig()),
    "no_memory_path": (CivilizationAblationConfig(use_memory_path=False), PathDependencyLossConfig()),
    "no_state_path": (CivilizationAblationConfig(use_state_path=False), PathDependencyLossConfig()),
    "no_rule_path": (CivilizationAblationConfig(use_rule_path=False), PathDependencyLossConfig()),
    "no_memory_dependency_loss": (CivilizationAblationConfig(), PathDependencyLossConfig(use_memory_dependency_loss=False)),
    "no_state_dependency_loss": (CivilizationAblationConfig(), PathDependencyLossConfig(use_state_dependency_loss=False)),
    "no_rule_dependency_loss": (CivilizationAblationConfig(), PathDependencyLossConfig(use_rule_dependency_loss=False)),
    "no_context_flip_loss": (CivilizationAblationConfig(), PathDependencyLossConfig(use_context_flip_loss=False)),
    "structure_only": (
        CivilizationAblationConfig(
            use_memory_path=False,
            use_state_path=False,
            use_rule_path=False,
            use_chain_state_loss=False,
            use_final_state_loss=False,
            use_priority_control_loss=False,
            use_centroid_separation_loss=False,
            use_hard_negative_loss=False,
        ),
        PathDependencyLossConfig(False, False, False, False),
    ),
}


MEMORY_REQUIRED_SCENARIOS = ("memory_required_two_hop", "memory_conflict_resolution", "counterfactual_memory_swap", "surface_invariant_label_flip")
STATE_REQUIRED_SCENARIOS = ("state_required_disambiguation", "memory_conflict_resolution", "surface_invariant_label_flip")
RULE_REQUIRED_SCENARIOS = ("rule_required_priority", "memory_conflict_resolution", "surface_invariant_label_flip")


def _validate_dependency_datasets(datasets: dict[str, list[LogicSample]], samples_per_label: int, tokenizer_vocab_size: int) -> dict:
    if set(datasets) != set(PATH_DEPENDENCY_SCENARIOS):
        raise ValueError("dependency datasets must include all dependency scenarios")
    stats = {
        "scenario_count": len(datasets),
        "surface_group_label_flips": 0,
        "memory_required_text_leaks": 0,
        "missing_required_paths": 0,
        "token_shape_uniform": True,
    }
    expected_shape = len(next(iter(datasets.values()))[0].token_ids)
    for scenario, samples in datasets.items():
        for label in LOGIC_LABELS:
            if len([sample for sample in samples if sample.label == label]) < samples_per_label:
                raise ValueError(f"{scenario}/{label} does not meet samples_per_label")
        groups: dict[str, set[str]] = {}
        for sample in samples:
            if len(sample.token_ids) != expected_shape:
                stats["token_shape_uniform"] = False
            if max(sample.token_ids) >= tokenizer_vocab_size or min(sample.token_ids) < 0:
                raise ValueError("dependency token ids out of range")
            if not sample.required_paths:
                stats["missing_required_paths"] += 1
            if scenario.startswith("memory_required") and sample.label in sample.text:
                stats["memory_required_text_leaks"] += 1
            groups.setdefault(sample.surface_group_id, set()).add(sample.label)
        stats["surface_group_label_flips"] += sum(1 for labels in groups.values() if len(labels) > 1)
    return stats


def _scenario_metrics(
    model: CivilizationTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    train_per_label: int,
    device: torch.device,
    seed: int,
    size: str,
    seq_len: int,
    mode: str,
    ablation_config: CivilizationAblationConfig,
) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    failures: list[dict] = []
    trace_stats = {
        "nan_inf": 0,
        "trace_missing": 0,
        "memory_attention_bad": 0,
        "rule_trace_zero": 0,
        "surface_group_flip_total": 0,
        "surface_group_flip_correct": 0,
    }
    for scenario in PATH_DEPENDENCY_SCENARIOS:
        train_samples, test_samples = split_by_label(datasets[scenario], train_per_label)
        train_vectors, _ = contextual_vectors(model, train_samples, device, ablation_config)
        test_vectors, traces = contextual_vectors(model, test_samples, device, ablation_config)
        train_labels = np.array([sample.label for sample in train_samples])
        test_labels = np.array([sample.label for sample in test_samples])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, test_vectors, test_labels)
        finite = bool(np.isfinite(train_vectors).all() and np.isfinite(test_vectors).all())
        if not finite:
            trace_stats["nan_inf"] += 1
        flat_traces = [layer_trace for trace_group in traces for layer_trace in trace_group]
        if not flat_traces:
            trace_stats["trace_missing"] += 1
        for trace in flat_traces:
            if trace.memory_attention.shape[-1] > 0:
                sums = trace.memory_attention.sum(dim=-1)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
                    trace_stats["memory_attention_bad"] += 1
            if scenario in RULE_REQUIRED_SCENARIOS and ablation_config.use_rule_path and trace.rule_influence_norm <= 1e-8:
                trace_stats["rule_trace_zero"] += 1
        predictions_by_surface: dict[str, set[str]] = {}
        for result in codebook.nearest_neighbors:
            sample = test_samples[result.sample_index]
            predictions_by_surface.setdefault(sample.surface_group_id, set()).add(result.predicted_label)
            if not result.correct and len(failures) < 300:
                failures.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "size": size,
                        "seq_len": seq_len,
                        "scenario": scenario,
                        "true_label": result.true_label,
                        "predicted_label": result.predicted_label,
                        "distance": result.distance,
                        "surface_group_id": sample.surface_group_id,
                        "required_paths": list(sample.required_paths),
                        "text": sample.text,
                    }
                )
        if scenario == "surface_invariant_label_flip":
            for labels in predictions_by_surface.values():
                trace_stats["surface_group_flip_total"] += 1
                if len(labels) > 1:
                    trace_stats["surface_group_flip_correct"] += 1
        rows.append(
            {
                "mode": mode,
                "seed": seed,
                "size": size,
                "seq_len": seq_len,
                "scenario": scenario,
                "accuracy": codebook.nearest_neighbor_accuracy,
                "macro_accuracy": codebook.macro_accuracy,
                "adjusted_rand_score": codebook.adjusted_rand_score,
                "per_label_accuracy": codebook.per_label_accuracy,
                "hidden_norm": float(np.linalg.norm(test_vectors, axis=1).mean()),
                "trace_count": len(flat_traces),
                "finite_vectors": finite,
                "easiest_confusion_pair": codebook.easiest_confusion_pair,
            }
        )
    return rows, failures, trace_stats


def run_path_dependency_benchmark(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/path_dependency_benchmark",
    modes: tuple[str, ...] = tuple(PATH_DEPENDENCY_MODES),
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 300,
    train_per_label: int = 200,
    seq_lens: tuple[int, ...] = (32, 64),
    model_sizes: tuple[str, ...] = ("medium", "large_toy"),
    training_steps: int = 60,
    device: str | torch.device | None = "cpu",
) -> dict:
    target_device = torch.device(device) if isinstance(device, str) else device or torch.device("cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    all_rows: list[dict] = []
    failure_cases: list[dict] = []
    trace_contribution: dict[str, dict] = {}
    dataset_stats: dict[str, dict] = {}
    total_runs = len(modes) * len(seeds) * len(seq_lens) * len(model_sizes)
    completed = 0
    for mode in modes:
        if mode not in PATH_DEPENDENCY_MODES:
            raise ValueError(f"unknown path dependency mode {mode}")
        ablation_config, loss_config = PATH_DEPENDENCY_MODES[mode]
        mode_trace_stats = {"nan_inf": 0, "trace_missing": 0, "memory_attention_bad": 0, "rule_trace_zero": 0, "surface_group_flip_total": 0, "surface_group_flip_correct": 0}
        for size_name in model_sizes:
            size = HARD_MODEL_SIZES[size_name]
            for seq_len in seq_lens:
                for seed in seeds:
                    datasets, tokenizer = build_path_dependency_datasets(samples_per_label=samples_per_label, max_seq_len=seq_len, seed=seed)
                    dataset_stats[f"{mode}_{seed}_{seq_len}"] = _validate_dependency_datasets(datasets, samples_per_label, tokenizer.vocab_size)
                    config = TransformerConfigTorch(
                        vocab_size=tokenizer.vocab_size,
                        model_dim=size["model_dim"],
                        hidden_dim=size["hidden_dim"],
                        num_heads=size["num_heads"],
                        num_layers=size["num_layers"],
                        max_seq_len=seq_len,
                        seed=seed,
                    )
                    torch.manual_seed(seed)
                    model = CivilizationTransformerTorch(config).to(target_device)
                    training = run_path_dependency_training(
                        model,
                        datasets,
                        target_device,
                        seed=seed,
                        train_per_label=train_per_label,
                        steps=training_steps,
                        ablation_config=ablation_config,
                        loss_config=loss_config,
                    )
                    rows, failures, trace_stats = _scenario_metrics(model, datasets, train_per_label, target_device, seed, size_name, seq_len, mode, ablation_config)
                    all_rows.extend(rows)
                    failure_cases.extend(failures)
                    for key, value in trace_stats.items():
                        mode_trace_stats[key] += value
                    runs.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "size": size_name,
                            "seq_len": seq_len,
                            "training": {
                                "steps": training.steps,
                                "train_samples": training.train_samples,
                                "initial_total_loss": training.initial_total_loss,
                                "final_total_loss": training.final_total_loss,
                                "total_loss_decreased": training.total_loss_decreased,
                                "initial_classification_loss": training.initial_classification_loss,
                                "final_classification_loss": training.final_classification_loss,
                                "classification_loss_decreased": training.classification_loss_decreased,
                                "initial_memory_dependency_loss": training.initial_memory_dependency_loss,
                                "final_memory_dependency_loss": training.final_memory_dependency_loss,
                                "memory_dependency_loss_decreased": training.memory_dependency_loss_decreased,
                                "initial_state_dependency_loss": training.initial_state_dependency_loss,
                                "final_state_dependency_loss": training.final_state_dependency_loss,
                                "state_dependency_loss_decreased": training.state_dependency_loss_decreased,
                                "initial_rule_dependency_loss": training.initial_rule_dependency_loss,
                                "final_rule_dependency_loss": training.final_rule_dependency_loss,
                                "rule_dependency_loss_decreased": training.rule_dependency_loss_decreased,
                                "initial_context_flip_loss": training.initial_context_flip_loss,
                                "final_context_flip_loss": training.final_context_flip_loss,
                                "context_flip_loss_decreased": training.context_flip_loss_decreased,
                                "disabled_losses": training.disabled_losses,
                            },
                        }
                    )
                    completed += 1
                    print(f"path_dependency_progress {completed}/{total_runs} mode={mode} seed={seed} size={size_name} seq_len={seq_len}", flush=True)
        trace_contribution[mode] = mode_trace_stats
    comparison = _compare_modes(all_rows)
    gates = _stage_gates(all_rows, comparison, trace_contribution)
    summary = {
        "modes": list(modes),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "scenarios": list(PATH_DEPENDENCY_SCENARIOS),
        "num_training_runs": len(runs),
        "mode_average_accuracy": _mode_average_accuracy(all_rows),
        "scenario_average_accuracy": _scenario_average_accuracy(all_rows),
        "comparison": comparison,
        "trace_contribution": trace_contribution,
        "dataset_stats": dataset_stats,
        "stage_gates": gates,
        "weak_modules": _weak_modules(comparison),
        "passes_stage_gate": all(gates.values()),
        "allows_large_model_migration_planning": all(gates.values()),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "failure_cases.json").write_text(json.dumps(failure_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_path_metrics(output_path / "path_metrics.csv", all_rows)
    _write_ablation_drop(output_path / "ablation_drop.csv", comparison)
    _write_surface_group_flips(output_path / "surface_group_flips.csv", trace_contribution)
    _write_trace_contribution(output_path / "trace_contribution.csv", trace_contribution)
    return summary


def _mode_average_accuracy(rows: list[dict]) -> dict[str, float]:
    return {mode: float(np.mean([row["accuracy"] for row in rows if row["mode"] == mode])) for mode in sorted({row["mode"] for row in rows})}


def _scenario_average_accuracy(rows: list[dict]) -> dict[str, dict[str, float]]:
    modes = sorted({row["mode"] for row in rows})
    return {
        mode: {
            scenario: float(np.mean([row["accuracy"] for row in rows if row["mode"] == mode and row["scenario"] == scenario]))
            for scenario in PATH_DEPENDENCY_SCENARIOS
        }
        for mode in modes
    }


def _compare_modes(rows: list[dict]) -> dict[str, dict]:
    by_mode = _scenario_average_accuracy(rows)
    full = by_mode["full"]
    comparison: dict[str, dict] = {}
    for mode, values in by_mode.items():
        drops = {scenario: full[scenario] - values[scenario] for scenario in PATH_DEPENDENCY_SCENARIOS}
        comparison[mode] = {
            "scenario_drop": drops,
            "average_drop": float(np.mean(list(drops.values()))),
            "memory_required_drop": float(np.mean([drops[scenario] for scenario in MEMORY_REQUIRED_SCENARIOS])),
            "state_required_drop": float(np.mean([drops[scenario] for scenario in STATE_REQUIRED_SCENARIOS])),
            "rule_required_drop": float(np.mean([drops[scenario] for scenario in RULE_REQUIRED_SCENARIOS])),
        }
    return comparison


def _stage_gates(rows: list[dict], comparison: dict[str, dict], trace_contribution: dict[str, dict]) -> dict[str, bool]:
    full_rows = [row for row in rows if row["mode"] == "full"]
    full_by_scenario = _scenario_average_accuracy(rows)["full"]
    full_failures = trace_contribution["full"]
    flip_total = full_failures["surface_group_flip_total"]
    flip_accuracy = full_failures["surface_group_flip_correct"] / flip_total if flip_total else 0.0
    return {
        "full_engineering_failures_zero": bool(full_failures["nan_inf"] == 0 and full_failures["trace_missing"] == 0 and full_failures["memory_attention_bad"] == 0 and full_failures["rule_trace_zero"] == 0 and all(row["finite_vectors"] for row in full_rows)),
        "full_dependency_average_accuracy": float(np.mean(list(full_by_scenario.values()))) >= 0.80,
        "no_memory_path_drop": comparison["no_memory_path"]["memory_required_drop"] >= 0.20,
        "no_state_path_drop": comparison["no_state_path"]["state_required_drop"] >= 0.15,
        "no_rule_path_drop": comparison["no_rule_path"]["rule_required_drop"] >= 0.15,
        "structure_only_drop": comparison["structure_only"]["average_drop"] >= 0.25,
        "surface_group_flip_accuracy": flip_accuracy >= 0.80,
    }


def _weak_modules(comparison: dict[str, dict]) -> list[str]:
    weak: list[str] = []
    if comparison["no_memory_path"]["memory_required_drop"] < 0.20:
        weak.append("memory_path")
    if comparison["no_state_path"]["state_required_drop"] < 0.15:
        weak.append("state_path")
    if comparison["no_rule_path"]["rule_required_drop"] < 0.15:
        weak.append("rule_path")
    return weak


def _write_path_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "seed", "size", "seq_len", "scenario", "accuracy", "macro_accuracy", "adjusted_rand_score", "hidden_norm", "trace_count", *[f"{label}_accuracy" for label in LOGIC_LABELS]])
        for row in rows:
            writer.writerow([row["mode"], row["seed"], row["size"], row["seq_len"], row["scenario"], row["accuracy"], row["macro_accuracy"], row["adjusted_rand_score"], row["hidden_norm"], row["trace_count"], *[row["per_label_accuracy"][label] for label in LOGIC_LABELS]])


def _write_ablation_drop(path: Path, comparison: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "average_drop", "memory_required_drop", "state_required_drop", "rule_required_drop", *PATH_DEPENDENCY_SCENARIOS])
        for mode, values in comparison.items():
            writer.writerow([mode, values["average_drop"], values["memory_required_drop"], values["state_required_drop"], values["rule_required_drop"], *[values["scenario_drop"][scenario] for scenario in PATH_DEPENDENCY_SCENARIOS]])


def _write_surface_group_flips(path: Path, trace_contribution: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "surface_group_flip_total", "surface_group_flip_correct", "surface_group_flip_accuracy"])
        for mode, values in trace_contribution.items():
            total = values["surface_group_flip_total"]
            accuracy = values["surface_group_flip_correct"] / total if total else 0.0
            writer.writerow([mode, total, values["surface_group_flip_correct"], accuracy])


def _write_trace_contribution(path: Path, trace_contribution: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "nan_inf", "trace_missing", "memory_attention_bad", "rule_trace_zero"])
        for mode, values in trace_contribution.items():
            writer.writerow([mode, values["nan_inf"], values["trace_missing"], values["memory_attention_bad"], values["rule_trace_zero"]])


def main() -> None:
    summary = run_path_dependency_benchmark()
    print("mode_average_accuracy", summary["mode_average_accuracy"])
    print("stage_gates", summary["stage_gates"])
    print("weak_modules", summary["weak_modules"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
