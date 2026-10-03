from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..model import CivilizationTransformerTorch, TransformerConfigTorch
from .dataset import LOGIC_LABELS, PATH_DEPENDENCY_SCENARIOS, PATH_STRESS_PROFILES, LogicSample, build_path_dependency_datasets
from .path_dependency_training import run_path_dependency_training
from .run_path_dependency_benchmark import (
    MEMORY_REQUIRED_SCENARIOS,
    PATH_DEPENDENCY_MODES,
    RULE_REQUIRED_SCENARIOS,
    STATE_REQUIRED_SCENARIOS,
    _scenario_metrics,
)


PATH_STRESS_MODEL_SIZES = {
    "medium": {"model_dim": 32, "hidden_dim": 64, "num_layers": 3, "num_heads": 4},
    "large_toy": {"model_dim": 48, "hidden_dim": 96, "num_layers": 4, "num_heads": 4},
    "path_large": {"model_dim": 64, "hidden_dim": 128, "num_layers": 4, "num_heads": 4},
}
DEFAULT_STRESS_MODES = ("full", "no_memory_path", "no_state_path", "no_rule_path", "structure_only")


def _validate_stress_dataset(
    datasets: dict[str, list[LogicSample]],
    samples_per_label: int,
    tokenizer_vocab_size: int,
    stress_profile: str,
) -> dict:
    stats = {
        "token_shape_error": 0,
        "dataset_leakage_error": 0,
        "missing_noise_context": 0,
        "missing_conflict_context": 0,
        "surface_group_label_flips": 0,
    }
    expected_shape = len(next(iter(datasets.values()))[0].token_ids)
    for scenario, samples in datasets.items():
        for label in LOGIC_LABELS:
            if len([sample for sample in samples if sample.label == label]) < samples_per_label:
                raise ValueError(f"{stress_profile}/{scenario}/{label} does not meet samples_per_label")
        groups: dict[str, set[str]] = {}
        for sample in samples:
            if len(sample.token_ids) != expected_shape or max(sample.token_ids) >= tokenizer_vocab_size or min(sample.token_ids) < 0:
                stats["token_shape_error"] += 1
            if stress_profile == "obfuscated_v1" and any(label in f"{sample.memory_target} {sample.rule_target} {sample.state_target}" for label in LOGIC_LABELS):
                stats["dataset_leakage_error"] += 1
            if stress_profile == "noisy_context_v1" and sample.context_noise_count <= 0:
                stats["missing_noise_context"] += 1
            if stress_profile == "conflicting_context_v1" and sample.conflict_context_count <= 0:
                stats["missing_conflict_context"] += 1
            groups.setdefault(sample.surface_group_id, set()).add(sample.label)
        stats["surface_group_label_flips"] += sum(1 for labels in groups.values() if len(labels) > 1)
    return stats


def run_path_dependency_stress_test(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/path_dependency_stress_test",
    modes: tuple[str, ...] = DEFAULT_STRESS_MODES,
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606, 707),
    samples_per_label: int = 500,
    train_per_label: int = 320,
    seq_lens: tuple[int, ...] = (32, 64, 96),
    model_sizes: tuple[str, ...] = ("medium", "large_toy", "path_large"),
    stress_profiles: tuple[str, ...] = PATH_STRESS_PROFILES,
    training_steps: int = 45,
    device: str | torch.device | None = "cpu",
    resume: bool = True,
) -> dict:
    target_device = torch.device(device) if isinstance(device, str) else device or torch.device("cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(output_path) if resume else _empty_checkpoint()
    runs: list[dict] = checkpoint["runs"]
    rows: list[dict] = checkpoint["rows"]
    failure_cases: list[dict] = checkpoint["failure_cases"]
    trace_contribution: dict[str, dict] = checkpoint["trace_contribution"]
    dataset_stats: dict[str, dict] = checkpoint["dataset_stats"]
    completed_keys: set[str] = set(checkpoint["completed_keys"])
    total = len(modes) * len(seeds) * len(seq_lens) * len(model_sizes) * len(stress_profiles)
    current = len(completed_keys)
    for mode in modes:
        if mode not in PATH_DEPENDENCY_MODES:
            raise ValueError(f"unknown path dependency mode {mode}")
        ablation_config, loss_config = PATH_DEPENDENCY_MODES[mode]
        mode_trace = trace_contribution.setdefault(mode, _empty_trace())
        for stress_profile in stress_profiles:
            if stress_profile not in PATH_STRESS_PROFILES:
                raise ValueError(f"unknown stress_profile {stress_profile}")
            for size_name in model_sizes:
                if size_name not in PATH_STRESS_MODEL_SIZES:
                    raise ValueError(f"unknown model size {size_name}")
                size = PATH_STRESS_MODEL_SIZES[size_name]
                for seq_len in seq_lens:
                    for seed in seeds:
                        run_key = _run_key(mode, stress_profile, size_name, seq_len, seed)
                        if run_key in completed_keys:
                            print(
                                f"path_dependency_stress_skip {current}/{total} mode={mode} seed={seed} size={size_name} seq_len={seq_len} stress_profile={stress_profile}",
                                flush=True,
                            )
                            continue
                        datasets, tokenizer = build_path_dependency_datasets(samples_per_label=samples_per_label, max_seq_len=seq_len, seed=seed, stress_profile=stress_profile)
                        dataset_key = f"{mode}_{stress_profile}_{size_name}_{seq_len}_{seed}"
                        dataset_stats[dataset_key] = _validate_stress_dataset(datasets, samples_per_label, tokenizer.vocab_size, stress_profile)
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
                        scenario_rows, failures, trace_stats = _scenario_metrics(model, datasets, train_per_label, target_device, seed, size_name, seq_len, mode, ablation_config)
                        for row in scenario_rows:
                            row["stress_profile"] = stress_profile
                        for failure in failures:
                            failure["stress_profile"] = stress_profile
                        rows.extend(scenario_rows)
                        failure_cases.extend(failures)
                        for key, value in trace_stats.items():
                            mode_trace[key] += value
                        run_record = {
                            "mode": mode,
                            "seed": seed,
                            "size": size_name,
                            "seq_len": seq_len,
                            "stress_profile": stress_profile,
                            "training": {
                                "steps": training.steps,
                                "train_samples": training.train_samples,
                                "initial_total_loss": training.initial_total_loss,
                                "final_total_loss": training.final_total_loss,
                                "total_loss_decreased": training.total_loss_decreased,
                                "initial_classification_loss": training.initial_classification_loss,
                                "final_classification_loss": training.final_classification_loss,
                                "classification_loss_decreased": training.classification_loss_decreased,
                                "disabled_losses": training.disabled_losses,
                            },
                        }
                        runs.append(run_record)
                        completed_keys.add(run_key)
                        _append_checkpoint(
                            output_path,
                            run_key,
                            run_record,
                            scenario_rows,
                            failures,
                            trace_stats,
                            dataset_key,
                            dataset_stats[dataset_key],
                        )
                        current += 1
                        print(
                            f"path_dependency_stress_progress {current}/{total} mode={mode} seed={seed} size={size_name} seq_len={seq_len} stress_profile={stress_profile}",
                            flush=True,
                        )
        trace_contribution[mode] = mode_trace
    comparison = _compare_modes_by_profile(rows)
    gates = _stage_gates(rows, comparison, trace_contribution, dataset_stats)
    summary = {
        "modes": list(modes),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "stress_profiles": list(stress_profiles),
        "scenarios": list(PATH_DEPENDENCY_SCENARIOS),
        "num_training_runs": len(runs),
        "mode_average_accuracy": _mode_average(rows),
        "profile_average_accuracy": _profile_average(rows),
        "mode_profile_accuracy": _mode_profile_average(rows),
        "scenario_profile_accuracy": _scenario_profile_average(rows),
        "comparison": comparison,
        "seed_stability": _seed_stability(rows),
        "size_comparison": _size_comparison(rows),
        "seq_len_comparison": _seq_len_comparison(rows),
        "trace_contribution": trace_contribution,
        "dataset_stats": dataset_stats,
        "stage_gates": gates,
        "passes_stage_gate": all(gates.values()),
        "allows_large_model_migration_planning": all(gates.values()),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "failure_cases.json").write_text(json.dumps(failure_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_path_metrics(output_path / "path_metrics.csv", rows)
    _write_ablation_drop(output_path / "ablation_drop.csv", comparison)
    _write_context_profile_metrics(output_path / "context_profile_metrics.csv", summary["mode_profile_accuracy"])
    _write_seed_stability(output_path / "seed_stability.csv", summary["seed_stability"])
    _write_size_comparison(output_path / "size_comparison.csv", summary["size_comparison"])
    _write_seq_len_comparison(output_path / "seq_len_comparison.csv", summary["seq_len_comparison"])
    _write_trace_contribution(output_path / "trace_contribution.csv", trace_contribution)
    return summary


def _run_key(mode: str, stress_profile: str, size_name: str, seq_len: int, seed: int) -> str:
    return f"{mode}|{stress_profile}|{size_name}|{seq_len}|{seed}"


def _empty_trace() -> dict[str, int]:
    return {"nan_inf": 0, "trace_missing": 0, "memory_attention_bad": 0, "rule_trace_zero": 0, "surface_group_flip_total": 0, "surface_group_flip_correct": 0}


def _empty_checkpoint() -> dict:
    return {"completed_keys": set(), "runs": [], "rows": [], "failure_cases": [], "trace_contribution": {}, "dataset_stats": {}}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_checkpoint(output_path: Path) -> dict:
    completed_records = _jsonl(output_path / "checkpoint_completed.jsonl")
    completed_keys = {record["run_key"] for record in completed_records}
    runs = [record["run"] for record in _jsonl(output_path / "checkpoint_runs.jsonl") if record["run_key"] in completed_keys]
    rows: list[dict] = []
    for record in _jsonl(output_path / "checkpoint_rows.jsonl"):
        if record["run_key"] in completed_keys:
            rows.extend(record["rows"])
    failure_cases: list[dict] = []
    for record in _jsonl(output_path / "checkpoint_failures.jsonl"):
        if record["run_key"] in completed_keys:
            failure_cases.extend(record["failures"])
    trace_contribution: dict[str, dict] = {}
    for record in _jsonl(output_path / "checkpoint_trace.jsonl"):
        if record["run_key"] not in completed_keys:
            continue
        mode = record["mode"]
        trace = trace_contribution.setdefault(mode, _empty_trace())
        for key, value in record["trace_stats"].items():
            trace[key] += value
    dataset_stats = {
        record["dataset_key"]: record["dataset_stats"]
        for record in _jsonl(output_path / "checkpoint_dataset_stats.jsonl")
        if record["run_key"] in completed_keys
    }
    return {
        "completed_keys": completed_keys,
        "runs": runs,
        "rows": rows,
        "failure_cases": failure_cases,
        "trace_contribution": trace_contribution,
        "dataset_stats": dataset_stats,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_checkpoint(
    output_path: Path,
    run_key: str,
    run_record: dict,
    scenario_rows: list[dict],
    failures: list[dict],
    trace_stats: dict,
    dataset_key: str,
    dataset_stat: dict,
) -> None:
    _append_jsonl(output_path / "checkpoint_runs.jsonl", {"run_key": run_key, "run": run_record})
    _append_jsonl(output_path / "checkpoint_rows.jsonl", {"run_key": run_key, "rows": scenario_rows})
    _append_jsonl(output_path / "checkpoint_failures.jsonl", {"run_key": run_key, "failures": failures})
    _append_jsonl(output_path / "checkpoint_trace.jsonl", {"run_key": run_key, "mode": run_record["mode"], "trace_stats": trace_stats})
    _append_jsonl(output_path / "checkpoint_dataset_stats.jsonl", {"run_key": run_key, "dataset_key": dataset_key, "dataset_stats": dataset_stat})
    _append_jsonl(output_path / "checkpoint_completed.jsonl", {"run_key": run_key})


def _mode_average(rows: list[dict]) -> dict[str, float]:
    return {mode: float(np.mean([row["accuracy"] for row in rows if row["mode"] == mode])) for mode in sorted({row["mode"] for row in rows})}


def _profile_average(rows: list[dict]) -> dict[str, float]:
    return {profile: float(np.mean([row["accuracy"] for row in rows if row["mode"] == "full" and row["stress_profile"] == profile])) for profile in sorted({row["stress_profile"] for row in rows})}


def _mode_profile_average(rows: list[dict]) -> dict[str, dict[str, float]]:
    modes = sorted({row["mode"] for row in rows})
    profiles = sorted({row["stress_profile"] for row in rows})
    return {
        mode: {
            profile: float(np.mean([row["accuracy"] for row in rows if row["mode"] == mode and row["stress_profile"] == profile]))
            for profile in profiles
        }
        for mode in modes
    }


def _scenario_profile_average(rows: list[dict]) -> dict[str, dict[str, float]]:
    profiles = sorted({row["stress_profile"] for row in rows})
    return {
        profile: {
            scenario: float(np.mean([row["accuracy"] for row in rows if row["mode"] == "full" and row["stress_profile"] == profile and row["scenario"] == scenario]))
            for scenario in PATH_DEPENDENCY_SCENARIOS
        }
        for profile in profiles
    }


def _compare_modes_by_profile(rows: list[dict]) -> dict[str, dict]:
    mode_profile = _mode_profile_average(rows)
    full_profile = mode_profile["full"]
    comparison: dict[str, dict] = {}
    for mode in mode_profile:
        mode_rows = [row for row in rows if row["mode"] == mode]
        drops_by_profile = {profile: full_profile[profile] - mode_profile[mode][profile] for profile in full_profile}
        scenario_drops = {}
        for scenario in PATH_DEPENDENCY_SCENARIOS:
            full_value = float(np.mean([row["accuracy"] for row in rows if row["mode"] == "full" and row["scenario"] == scenario]))
            mode_value = float(np.mean([row["accuracy"] for row in mode_rows if row["scenario"] == scenario]))
            scenario_drops[scenario] = full_value - mode_value
        comparison[mode] = {
            "profile_drop": drops_by_profile,
            "scenario_drop": scenario_drops,
            "average_drop": float(np.mean(list(scenario_drops.values()))),
            "memory_required_drop": float(np.mean([scenario_drops[scenario] for scenario in MEMORY_REQUIRED_SCENARIOS])),
            "state_required_drop": float(np.mean([scenario_drops[scenario] for scenario in STATE_REQUIRED_SCENARIOS])),
            "rule_required_drop": float(np.mean([scenario_drops[scenario] for scenario in RULE_REQUIRED_SCENARIOS])),
        }
    return comparison


def _seed_stability(rows: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for profile in sorted({row["stress_profile"] for row in rows}):
        values = []
        for seed in sorted({row["seed"] for row in rows}):
            seed_values = [row["accuracy"] for row in rows if row["mode"] == "full" and row["stress_profile"] == profile and row["seed"] == seed]
            if seed_values:
                values.append(float(np.mean(seed_values)))
        result[profile] = {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)), "std": float(np.std(values))}
    return result


def _size_comparison(rows: list[dict]) -> dict[str, float]:
    return {
        size: float(np.mean([row["accuracy"] for row in rows if row["mode"] == "full" and row["size"] == size]))
        for size in sorted({row["size"] for row in rows})
    }


def _seq_len_comparison(rows: list[dict]) -> dict[str, float]:
    return {
        str(seq_len): float(np.mean([row["accuracy"] for row in rows if row["mode"] == "full" and row["seq_len"] == seq_len]))
        for seq_len in sorted({row["seq_len"] for row in rows})
    }


def _stage_gates(rows: list[dict], comparison: dict[str, dict], trace_contribution: dict[str, dict], dataset_stats: dict[str, dict]) -> dict[str, bool]:
    full_rows = [row for row in rows if row["mode"] == "full"]
    full_trace = trace_contribution["full"]
    mode_profile = _mode_profile_average(rows)
    full_profile = mode_profile["full"]
    size = _size_comparison(rows)
    seq = _seq_len_comparison(rows)
    dataset_errors = sum(stats["token_shape_error"] + stats["dataset_leakage_error"] + stats["missing_noise_context"] + stats["missing_conflict_context"] for stats in dataset_stats.values())
    direct = full_profile.get("direct_v1", 0.0)
    obfuscated = full_profile.get("obfuscated_v1", 0.0)
    noisy = full_profile.get("noisy_context_v1", 0.0)
    conflicting = full_profile.get("conflicting_context_v1", 0.0)
    structure_average = _mode_average(rows).get("structure_only", 1.0)
    return {
        "full_engineering_failures_zero": bool(full_trace["nan_inf"] == 0 and full_trace["trace_missing"] == 0 and full_trace["memory_attention_bad"] == 0 and full_trace["rule_trace_zero"] == 0 and dataset_errors == 0 and all(row["finite_vectors"] for row in full_rows)),
        "full_profile_accuracy": all(value >= 0.80 for value in full_profile.values()),
        "surface_group_flip_accuracy": _surface_flip_ok(trace_contribution["full"]),
        "no_memory_path_drop": comparison["no_memory_path"]["memory_required_drop"] >= 0.20,
        "no_state_path_drop": comparison["no_state_path"]["state_required_drop"] >= 0.15,
        "no_rule_path_drop": comparison["no_rule_path"]["rule_required_drop"] >= 0.15,
        "structure_only_drop": comparison["structure_only"]["average_drop"] >= 0.25,
        "structure_only_near_random": structure_average <= 0.45,
        "obfuscated_not_over_degraded": direct - obfuscated <= 0.10,
        "noisy_context_accuracy": noisy >= 0.75,
        "conflicting_context_accuracy": conflicting >= 0.75,
        "path_large_not_below_large_toy": size.get("path_large", 0.0) + 0.05 >= size.get("large_toy", 0.0),
        "seq96_not_below_seq32": seq.get("96", 0.0) + 0.08 >= seq.get("32", 0.0),
    }


def _surface_flip_ok(trace: dict) -> bool:
    total = trace["surface_group_flip_total"]
    return total > 0 and trace["surface_group_flip_correct"] / total >= 0.80


def _write_path_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "stress_profile", "seed", "size", "seq_len", "scenario", "accuracy", "macro_accuracy", "adjusted_rand_score", "hidden_norm", "trace_count", *[f"{label}_accuracy" for label in LOGIC_LABELS]])
        for row in rows:
            writer.writerow([row["mode"], row["stress_profile"], row["seed"], row["size"], row["seq_len"], row["scenario"], row["accuracy"], row["macro_accuracy"], row["adjusted_rand_score"], row["hidden_norm"], row["trace_count"], *[row["per_label_accuracy"][label] for label in LOGIC_LABELS]])


def _write_ablation_drop(path: Path, comparison: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "average_drop", "memory_required_drop", "state_required_drop", "rule_required_drop", *PATH_DEPENDENCY_SCENARIOS])
        for mode, values in comparison.items():
            writer.writerow([mode, values["average_drop"], values["memory_required_drop"], values["state_required_drop"], values["rule_required_drop"], *[values["scenario_drop"][scenario] for scenario in PATH_DEPENDENCY_SCENARIOS]])


def _write_context_profile_metrics(path: Path, mode_profile: dict[str, dict[str, float]]) -> None:
    profiles = sorted(next(iter(mode_profile.values())).keys()) if mode_profile else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", *profiles])
        for mode, values in mode_profile.items():
            writer.writerow([mode, *[values[profile] for profile in profiles]])


def _write_seed_stability(path: Path, stability: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stress_profile", "mean", "min", "max", "std"])
        for profile, values in stability.items():
            writer.writerow([profile, values["mean"], values["min"], values["max"], values["std"]])


def _write_size_comparison(path: Path, values: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["size", "full_accuracy"])
        for size, accuracy in values.items():
            writer.writerow([size, accuracy])


def _write_seq_len_comparison(path: Path, values: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_len", "full_accuracy"])
        for seq_len, accuracy in values.items():
            writer.writerow([seq_len, accuracy])


def _write_trace_contribution(path: Path, trace_contribution: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "nan_inf", "trace_missing", "memory_attention_bad", "rule_trace_zero", "surface_group_flip_total", "surface_group_flip_correct"])
        for mode, values in trace_contribution.items():
            writer.writerow([mode, values["nan_inf"], values["trace_missing"], values["memory_attention_bad"], values["rule_trace_zero"], values["surface_group_flip_total"], values["surface_group_flip_correct"]])


def main() -> None:
    summary = run_path_dependency_stress_test()
    print("profile_average_accuracy", summary["profile_average_accuracy"])
    print("mode_average_accuracy", summary["mode_average_accuracy"])
    print("stage_gates", summary["stage_gates"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
