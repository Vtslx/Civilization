from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch
from torch import nn
import torch.nn.functional as F

from ..device import resolve_device
from ..model import CivilizationAblationConfig, CivilizationTransformerTorch, TransformerConfigTorch
from .chain_alignment import run_chain_state_alignment_training
from .codebook import build_logic_codebook_train_test
from .dataset import HARD_LOGIC_SCENARIOS, LOGIC_LABELS, LogicSample, build_hard_logic_datasets
from .hidden_states import samples_to_tensor
from .run_civilization_stress_test import _context_checks, _memory_profiles, _rule_profiles, _state_profiles, run_civilization_stress_test


HARD_MODEL_SIZES = {
    "medium": {"model_dim": 32, "hidden_dim": 64, "num_layers": 3, "num_heads": 4},
    "large_toy": {"model_dim": 48, "hidden_dim": 96, "num_layers": 4, "num_heads": 4},
}
PRIMARY_SCENARIO_GATES = {
    "canonical_hard": 0.85,
    "synonym_hard": 0.85,
    "masked_keywords_hard": 0.85,
}
HARD_DIAGNOSTIC_GATES = {
    "two_hop_logic": 0.60,
    "three_hop_logic": 0.60,
    "counterfactual_pair": 0.60,
    "mixed_logic_priority": 0.60,
}


def _split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        if len(label_samples) <= train_per_label:
            raise ValueError("train_per_label must leave held-out hard samples for every label")
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


def _select_hard_train_samples(datasets: dict[str, list[LogicSample]], train_per_label: int, train_scenarios: tuple[str, ...]) -> list[LogicSample]:
    samples: list[LogicSample] = []
    for scenario in train_scenarios:
        train, _ = _split_by_label(datasets[scenario], train_per_label)
        samples.extend(train)
    return samples


def _labels_to_tensor(samples: list[LogicSample], device: torch.device) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(LOGIC_LABELS)}
    return torch.tensor([label_to_id[sample.label] for sample in samples], dtype=torch.long, device=device)


def _centroid_separation_loss(vectors: torch.Tensor, labels: torch.Tensor, margin: float = 4.0) -> torch.Tensor:
    centers: list[torch.Tensor] = []
    within_terms: list[torch.Tensor] = []
    for label_id in range(len(LOGIC_LABELS)):
        label_vectors = vectors[labels == label_id]
        if label_vectors.numel() == 0:
            raise ValueError(f"missing hard vectors for label id {label_id}")
        center = label_vectors.mean(dim=0)
        centers.append(center)
        within_terms.append(torch.mean(torch.sum((label_vectors - center) ** 2, dim=1)))
    centers_tensor = torch.stack(centers)
    distances = torch.cdist(centers_tensor, centers_tensor, p=2)
    mask = ~torch.eye(len(LOGIC_LABELS), dtype=torch.bool, device=vectors.device)
    return torch.stack(within_terms).mean() + F.relu(margin - distances[mask]).pow(2).mean()


def _hard_negative_loss(vectors: torch.Tensor, labels: torch.Tensor, margin: float = 6.0) -> torch.Tensor:
    distances = torch.cdist(vectors, vectors, p=2)
    different = labels.unsqueeze(0) != labels.unsqueeze(1)
    if not different.any():
        return torch.zeros((), dtype=vectors.dtype, device=vectors.device)
    return F.relu(margin - distances[different]).pow(2).mean()


def _train_hard_alignment(
    model: CivilizationTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    device: torch.device,
    seed: int,
    train_per_label: int,
    steps: int,
    forward_kwargs: dict,
    train_scenarios: tuple[str, ...],
    batch_size: int = 512,
) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    train_samples = _select_hard_train_samples(datasets, train_per_label, train_scenarios)
    input_ids = samples_to_tensor(train_samples, device)
    label_ids = _labels_to_tensor(train_samples, device)
    label_index_groups = [torch.where(label_ids == label_id)[0] for label_id in range(len(LOGIC_LABELS))]
    classifier = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=0.01, weight_decay=0.0)
    losses: list[dict[str, float]] = []
    for step in range(steps + 1):
        per_label_batch = max(1, batch_size // len(LOGIC_LABELS))
        balanced_indices: list[torch.Tensor] = []
        for label_id, group in enumerate(label_index_groups):
            start = (step * per_label_batch + label_id * 7) % len(group)
            local = torch.arange(start, start + per_label_batch, device=device) % len(group)
            balanced_indices.append(group[local])
        indices = torch.cat(balanced_indices)
        batch_input_ids = input_ids[indices]
        batch_label_ids = label_ids[indices]
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_input_ids, **forward_kwargs)
        pooled = output.hidden_states[-1].mean(dim=1)
        next_token = F.cross_entropy(output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]), batch_input_ids[:, 1:].reshape(-1))
        classification = F.cross_entropy(classifier(pooled), batch_label_ids)
        contrastive = _centroid_separation_loss(pooled, batch_label_ids)
        hard_negative = _hard_negative_loss(pooled, batch_label_ids)
        total = 0.12 * next_token + classification + 0.35 * contrastive + 0.08 * hard_negative
        losses.append(
            {
                "step": float(step),
                "next_token_loss": float(next_token.detach().cpu().item()),
                "classification_loss": float(classification.detach().cpu().item()),
                "contrastive_loss": float(contrastive.detach().cpu().item()),
                "hard_negative_loss": float(hard_negative.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            }
        )
        if step < steps:
            total.backward()
            optimizer.step()
    return {
        "seed": seed,
        "train_scenarios": list(train_scenarios),
        "train_samples": len(train_samples),
        "batch_shape": list(input_ids.shape),
        "train_batch_size": min(batch_size, len(train_samples)),
        "steps": steps,
        "initial_total_loss": losses[0]["total_loss"],
        "final_total_loss": losses[-1]["total_loss"],
        "total_loss_decreased": losses[-1]["total_loss"] < losses[0]["total_loss"],
        "initial_classification_loss": losses[0]["classification_loss"],
        "final_classification_loss": losses[-1]["classification_loss"],
        "classification_loss_decreased": losses[-1]["classification_loss"] < losses[0]["classification_loss"],
        "losses": losses,
    }


@torch.no_grad()
def _vectors(model: CivilizationTransformerTorch, samples: list[LogicSample], device: torch.device, context: dict) -> tuple[np.ndarray, object]:
    output = model(samples_to_tensor(samples, device), **context)
    pooled = output.hidden_states[-1].mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("hard benchmark hidden states contain NaN or Inf")
    return pooled.detach().cpu().numpy(), output


def _scenario_metrics(
    model: CivilizationTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    device: torch.device,
    train_per_label: int,
    context: dict,
    seed: int,
    size: str,
    seq_len: int,
    ablation_config: CivilizationAblationConfig | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if ablation_config is not None:
        context = {**context, "ablation_config": ablation_config}
    train_samples, _ = _split_by_label(datasets["canonical_hard"], train_per_label)
    train_vectors, _ = _vectors(model, train_samples, device, context)
    train_labels = np.array([sample.label for sample in train_samples])
    rows: list[dict] = []
    failures: list[dict] = []
    confusion_total = {label: {candidate: 0 for candidate in LOGIC_LABELS} for label in LOGIC_LABELS}
    for scenario in HARD_LOGIC_SCENARIOS:
        _, test_samples = _split_by_label(datasets[scenario], train_per_label)
        test_vectors, output = _vectors(model, test_samples, device, context)
        test_labels = np.array([sample.label for sample in test_samples])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, test_vectors, test_labels)
        finite = bool(np.isfinite(test_vectors).all())
        hidden_norm = float(np.linalg.norm(test_vectors, axis=1).mean())
        for true_label in LOGIC_LABELS:
            for predicted in LOGIC_LABELS:
                confusion_total[true_label][predicted] += codebook.confusion_matrix[true_label][predicted]
        for result in codebook.nearest_neighbors:
            if not result.correct and len(failures) < 200:
                sample = test_samples[result.sample_index]
                failures.append(
                    {
                        "seed": seed,
                        "size": size,
                        "seq_len": seq_len,
                        "scenario": scenario,
                        "sample_index": result.sample_index,
                        "true_label": result.true_label,
                        "predicted_label": result.predicted_label,
                        "distance": result.distance,
                        "text": sample.text,
                        "difficulty_level": sample.difficulty_level,
                        "logic_depth": sample.logic_depth,
                        "distractor_count": sample.distractor_count,
                    }
                )
        rows.append(
            {
                "seed": seed,
                "size": size,
                "seq_len": seq_len,
                "scenario": scenario,
                "accuracy": codebook.nearest_neighbor_accuracy,
                "macro_accuracy": codebook.macro_accuracy,
                "adjusted_rand_score": codebook.adjusted_rand_score,
                "per_label_accuracy": codebook.per_label_accuracy,
                "easiest_confusion_pair": codebook.easiest_confusion_pair,
                "hidden_norm": hidden_norm,
                "finite_vectors": finite,
                "trace_count": len(output.civilization_traces),
                "test_samples": len(test_samples),
            }
        )
    return rows, failures, confusion_total


def _validate_hard_datasets(datasets: dict[str, list[LogicSample]], samples_per_label: int, tokenizer_vocab_size: int, train_per_label: int) -> dict:
    masked_tokens = {"because", "therefore", "not", "never", "always", "critical", "priority", "if", "then"}
    stats = {"scenario_count": len(datasets), "masked_keyword_violations": 0, "adversarial_keyword_rows": 0, "counterfactual_split_safe": True}
    if set(datasets) != set(HARD_LOGIC_SCENARIOS):
        raise ValueError("hard datasets must include all hard scenarios")
    for scenario, samples in datasets.items():
        for label in LOGIC_LABELS:
            label_samples = [sample for sample in samples if sample.label == label]
            if len(label_samples) < samples_per_label:
                raise ValueError(f"{scenario}/{label} does not meet samples_per_label")
        for sample in samples:
            if len(sample.token_ids) != len(next(iter(datasets.values()))[0].token_ids):
                raise ValueError("hard token shapes are not uniform")
            if max(sample.token_ids) >= tokenizer_vocab_size or min(sample.token_ids) < 0:
                raise ValueError("hard token ids out of range")
            tokens = set(sample.text.split())
            if scenario == "masked_keywords_hard" and tokens.intersection(masked_tokens):
                stats["masked_keyword_violations"] += 1
            if scenario == "adversarial_keywords" and {"because", "not", "always", "critical", "if", "then"}.issubset(tokens):
                stats["adversarial_keyword_rows"] += 1
    for index in range(samples_per_label):
        in_train = index < train_per_label
        for label in LOGIC_LABELS:
            pair_samples = [sample for sample in datasets["counterfactual_pair"] if sample.label == label and sample.pair_id == f"counterfactual_pair_{index:04d}"]
            if pair_samples and ((index < train_per_label) != in_train):
                stats["counterfactual_split_safe"] = False
    return stats


def run_hard_logic_benchmark(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/hard_logic_benchmark",
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606, 707, 808, 909, 1001, 1112),
    samples_per_label: int = 300,
    train_per_label: int = 200,
    seq_lens: tuple[int, ...] = (32, 48, 64),
    model_sizes: tuple[str, ...] = ("medium", "large_toy"),
    training_steps: int = 70,
    device: str | torch.device | None = "cpu",
    run_stage16_regression: bool = True,
    training_mode: str = "hard_alignment_baseline",
    ablation_config: CivilizationAblationConfig | None = None,
) -> dict:
    if training_mode not in {"hard_alignment_baseline", "chain_state_alignment"}:
        raise ValueError("training_mode must be hard_alignment_baseline or chain_state_alignment")
    target_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage16 = None
    if run_stage16_regression:
        stage16 = run_civilization_stress_test(
            output_dir=output_path / "stage16_regression",
            seeds=(202,),
            samples_per_label=30,
            train_per_label=20,
            seq_lens=(18,),
            model_sizes=("small", "medium"),
            training_steps=35,
            device=target_device,
            run_regressions=False,
        )

    runs: list[dict] = []
    scenario_rows: list[dict] = []
    failure_cases: list[dict] = []
    trace_failures: list[dict] = []
    confusion_total = {label: {candidate: 0 for candidate in LOGIC_LABELS} for label in LOGIC_LABELS}
    dataset_stats: dict[str, dict] = {}
    train_scenarios = ("canonical_hard", "synonym_hard", "order_shuffled", "long_context", "distractor_facts", "masked_keywords_hard")
    total_runs = len(model_sizes) * len(seq_lens) * len(seeds)
    completed_runs = 0
    for size_name in model_sizes:
        size = HARD_MODEL_SIZES[size_name]
        for seq_len in seq_lens:
            for seed in seeds:
                datasets, tokenizer = build_hard_logic_datasets(samples_per_label=samples_per_label, max_seq_len=seq_len, seed=seed, include_chain_supervision=training_mode == "chain_state_alignment")
                dataset_stats[f"{seed}_{seq_len}"] = _validate_hard_datasets(datasets, samples_per_label, tokenizer.vocab_size, train_per_label)
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
                memories = _memory_profiles(config.model_dim, target_device)
                states = _state_profiles()
                rules = _rule_profiles(config.model_dim, target_device)
                train_context = {
                    "memory_vectors": memories["balanced"],
                    "state": states["balanced"],
                    "rule_vectors": rules["soft"]["vectors"],
                }
                if training_mode == "chain_state_alignment":
                    training_result = run_chain_state_alignment_training(
                        model,
                        datasets,
                        target_device,
                        seed=seed,
                        train_per_label=train_per_label,
                        steps=training_steps,
                        forward_kwargs=train_context,
                        train_scenarios=HARD_LOGIC_SCENARIOS,
                        ablation_config=ablation_config,
                    )
                    training = {
                        "seed": training_result.seed,
                        "train_scenarios": list(training_result.train_scenarios),
                        "train_samples": training_result.train_samples,
                        "train_batch_size": training_result.train_batch_size,
                        "steps": training_result.steps,
                        "initial_total_loss": training_result.initial_total_loss,
                        "final_total_loss": training_result.final_total_loss,
                        "total_loss_decreased": training_result.total_loss_decreased,
                        "initial_classification_loss": training_result.initial_classification_loss,
                        "final_classification_loss": training_result.final_classification_loss,
                        "classification_loss_decreased": training_result.classification_loss_decreased,
                        "initial_chain_state_loss": training_result.initial_chain_state_loss,
                        "final_chain_state_loss": training_result.final_chain_state_loss,
                        "chain_state_loss_decreased": training_result.chain_state_loss_decreased,
                        "initial_priority_control_loss": training_result.initial_priority_control_loss,
                        "final_priority_control_loss": training_result.final_priority_control_loss,
                        "priority_control_loss_decreased": training_result.priority_control_loss_decreased,
                        "chain_step_accuracy": training_result.chain_step_accuracy,
                        "final_target_accuracy": training_result.final_target_accuracy,
                        "priority_control_accuracy": training_result.priority_control_accuracy,
                        "disabled_losses": training_result.disabled_losses,
                    }
                else:
                    training = _train_hard_alignment(model, datasets, target_device, seed, train_per_label, training_steps, train_context, train_scenarios)
                rows, failures, confusion = _scenario_metrics(model, datasets, target_device, train_per_label, train_context, seed, size_name, seq_len, ablation_config=ablation_config)
                scenario_rows.extend(rows)
                failure_cases.extend(failures)
                for true_label in LOGIC_LABELS:
                    for predicted in LOGIC_LABELS:
                        confusion_total[true_label][predicted] += confusion[true_label][predicted]
                _, trace_samples = _split_by_label(datasets["canonical_hard"], train_per_label)
                for memory_name in ("empty", "noisy", "conflicting", "balanced"):
                    for state_name in ("strict", "creative", "defensive", "balanced"):
                        for rule_name in ("empty", "soft", "hard_block", "conflict"):
                            context = {"memory_vectors": memories[memory_name], "state": states[state_name], "rule_vectors": rules[rule_name]["vectors"]}
                            checks = _context_checks(model, trace_samples, target_device, context | {"ablation_config": ablation_config} if ablation_config is not None else context, rules[rule_name]["result"])
                            if (
                                not checks["finite_logits"]
                                or not checks["finite_hidden"]
                                or not checks["memory_attention_ok"]
                                or checks["trace_count"] <= 0
                                or (checks["state_output_delta"] <= 1e-6 and (ablation_config is None or ablation_config.use_state_path))
                                or (rule_name != "empty" and checks["rule_trace_delta"] <= 1e-6 and (ablation_config is None or ablation_config.use_rule_path))
                                or (rule_name == "hard_block" and not checks["hard_block_recorded"])
                            ):
                                trace_failures.append(
                                    {
                                        "seed": seed,
                                        "size": size_name,
                                        "seq_len": seq_len,
                                        "memory": memory_name,
                                        "state": state_name,
                                        "rules": rule_name,
                                        "checks": checks,
                                    }
                                )
                runs.append(
                    {
                        "seed": seed,
                        "size": size_name,
                        "seq_len": seq_len,
                        "training": {key: value for key, value in training.items() if key != "losses"},
                    }
                )
                completed_runs += 1
                print(f"hard_benchmark_progress {completed_runs}/{total_runs} seed={seed} size={size_name} seq_len={seq_len}", flush=True)

    summary = _summarize(runs, scenario_rows, trace_failures, dataset_stats, stage16, seeds, samples_per_label, train_per_label, seq_lens, model_sizes, training_mode)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "failure_cases.json").write_text(json.dumps(failure_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "trace_failures.json").write_text(json.dumps(trace_failures, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_scenario_metrics(output_path / "scenario_metrics.csv", scenario_rows)
    _write_confusion(output_path / "confusion_by_scenario.csv", confusion_total)
    return summary


def _summarize(
    runs: list[dict],
    scenario_rows: list[dict],
    trace_failures: list[dict],
    dataset_stats: dict[str, dict],
    stage16: dict | None,
    seeds: tuple[int, ...],
    samples_per_label: int,
    train_per_label: int,
    seq_lens: tuple[int, ...],
    model_sizes: tuple[str, ...],
    training_mode: str,
) -> dict:
    by_scenario = {
        scenario: float(np.mean([row["accuracy"] for row in scenario_rows if row["scenario"] == scenario]))
        for scenario in HARD_LOGIC_SCENARIOS
    }
    by_size = {
        size: {
            scenario: float(np.mean([row["accuracy"] for row in scenario_rows if row["size"] == size and row["scenario"] == scenario]))
            for scenario in HARD_LOGIC_SCENARIOS
        }
        for size in model_sizes
    }
    by_label: dict[str, float] = {}
    for label in LOGIC_LABELS:
        values = [row["per_label_accuracy"][label] for row in scenario_rows]
        by_label[label] = float(np.mean(values))
    failures = {
        "nan_inf": sum(1 for row in scenario_rows if not row["finite_vectors"]),
        "trace_missing": sum(1 for row in scenario_rows if row["trace_count"] <= 0),
        "trace_failures": len(trace_failures),
        "masked_keyword_violations": sum(stats["masked_keyword_violations"] for stats in dataset_stats.values()),
        "adversarial_keyword_missing": sum(1 for stats in dataset_stats.values() if stats["adversarial_keyword_rows"] == 0),
        "counterfactual_split_unsafe": sum(1 for stats in dataset_stats.values() if not stats["counterfactual_split_safe"]),
    }
    primary_gates = {scenario: by_scenario[scenario] >= threshold for scenario, threshold in PRIMARY_SCENARIO_GATES.items()}
    hard_gates = {scenario: by_scenario[scenario] >= threshold for scenario, threshold in HARD_DIAGNOSTIC_GATES.items()}
    losses_ok = all(run["training"]["total_loss_decreased"] and run["training"]["classification_loss_decreased"] for run in runs)
    medium_vs_large = None
    if "medium" in by_size and "large_toy" in by_size:
        medium_avg = float(np.mean(list(by_size["medium"].values())))
        large_avg = float(np.mean(list(by_size["large_toy"].values())))
        medium_vs_large = {"medium_average": medium_avg, "large_toy_average": large_avg, "large_toy_lower_than_medium": large_avg + 0.05 < medium_avg}
    stage16_ok = True if stage16 is None else bool(stage16["passes_stage_gate"])
    allows_migration = bool(stage16_ok and losses_ok and all(value == 0 for value in failures.values()) and all(primary_gates.values()) and all(hard_gates.values()))
    return {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "test_per_label": samples_per_label - train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "training_mode": training_mode,
        "scenarios": list(HARD_LOGIC_SCENARIOS),
        "num_training_runs": len(runs),
        "num_scenario_rows": len(scenario_rows),
        "average_accuracy_by_scenario": by_scenario,
        "average_accuracy_by_size": by_size,
        "average_accuracy_by_label": by_label,
        "worst_scenario": min(by_scenario, key=by_scenario.get),
        "worst_label": min(by_label, key=by_label.get),
        "primary_gates": primary_gates,
        "hard_diagnostic_gates": hard_gates,
        "losses_decreased": losses_ok,
        "chain_metrics": _chain_metrics(runs),
        "failures": failures,
        "medium_vs_large": medium_vs_large,
        "stage16_regression": None if stage16 is None else {"passes_stage_gate": stage16["passes_stage_gate"]},
        "allows_large_model_migration_planning": allows_migration,
        "passes_stage_gate": allows_migration,
    }


def _chain_metrics(runs: list[dict]) -> dict:
    values = [run["training"] for run in runs if "chain_step_accuracy" in run["training"]]
    if not values:
        return {}
    return {
        "chain_step_accuracy": float(np.mean([value["chain_step_accuracy"] for value in values])),
        "final_target_accuracy": float(np.mean([value["final_target_accuracy"] for value in values])),
        "priority_control_accuracy": float(np.mean([value["priority_control_accuracy"] for value in values])),
        "chain_state_loss_decreased": all(value["chain_state_loss_decreased"] for value in values),
        "priority_control_loss_decreased": all(value["priority_control_loss_decreased"] for value in values),
    }


def _write_scenario_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "size", "seq_len", "scenario", "accuracy", "macro_accuracy", "adjusted_rand_score", "hidden_norm", "trace_count", "easiest_confusion_pair", *[f"{label}_accuracy" for label in LOGIC_LABELS]])
        for row in rows:
            writer.writerow([
                row["seed"],
                row["size"],
                row["seq_len"],
                row["scenario"],
                row["accuracy"],
                row["macro_accuracy"],
                row["adjusted_rand_score"],
                row["hidden_norm"],
                row["trace_count"],
                row["easiest_confusion_pair"],
                *[row["per_label_accuracy"][label] for label in LOGIC_LABELS],
            ])


def _write_confusion(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *LOGIC_LABELS])
        for label in LOGIC_LABELS:
            writer.writerow([label, *[matrix[label][candidate] for candidate in LOGIC_LABELS]])


def main() -> None:
    summary = run_hard_logic_benchmark()
    print("average_accuracy_by_scenario", summary["average_accuracy_by_scenario"])
    print("worst_scenario", summary["worst_scenario"])
    print("worst_label", summary["worst_label"])
    print("losses_decreased", summary["losses_decreased"])
    print("failures", summary["failures"])
    print("num_training_runs", summary["num_training_runs"])
    print("num_scenario_rows", summary["num_scenario_rows"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
