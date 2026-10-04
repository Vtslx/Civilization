from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..device import resolve_device
from ..memory import MemoryEncoderTorch, MemoryItem
from ..model import CivilizationTransformerTorch, TransformerConfigTorch
from ..rules import RuleEngineTorch, RuleItem
from ..state import StateConfig
from .alignment import run_alignment_training
from .codebook import build_logic_codebook_train_test, run_codebook_generalization_matrix
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, LogicSample, build_logic_variant_datasets
from .hidden_states import samples_to_tensor
from .run_civilization_transformer_experiment import run_civilization_transformer_experiment
from .run_fusion_injection import run_fusion_injection_analysis
from .run_hidden_injection import run_hidden_injection_analysis


MODEL_SIZES = {
    "small": {"model_dim": 24, "hidden_dim": 48, "num_layers": 2, "num_heads": 4},
    "medium": {"model_dim": 32, "hidden_dim": 64, "num_layers": 3, "num_heads": 4},
}


def _split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


def _memory_profiles(model_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoder = MemoryEncoderTorch(model_dim, device=device)
    profiles = {
        "empty": [],
        "balanced": [
            MemoryItem("m-cause", "causal memory", "cause therefore leads effect", "causality", 1.0, 1.0),
            MemoryItem("m-neg", "negation memory", "not never deny reject", "negation", 0.9, 1.0),
            MemoryItem("m-conflict", "conflict memory", "always never contradict", "conflict", 0.9, 1.0),
            MemoryItem("m-priority", "priority memory", "critical priority urgent override", "priority", 1.0, 1.0),
            MemoryItem("m-condition", "condition memory", "if then threshold when", "condition", 0.9, 1.0),
        ],
        "noisy": [
            MemoryItem("m-noise-a", "unrelated memory", "alpha beta gamma delta", "noise", 0.5, 0.7),
            MemoryItem("m-noise-b", "mixed memory", "cause noise priority random", "mixed", 0.4, 0.5),
        ],
        "conflicting": [
            MemoryItem("m-conflict-a", "conflicting memory", "always stable never stable contradiction", "conflict", 1.0, 1.0),
            MemoryItem("m-conflict-b", "opposed memory", "critical condition not cause", "mixed", 0.9, 0.9),
        ],
    }
    return {name: encoder.encode(items) for name, items in profiles.items()}


def _state_profiles() -> dict[str, StateConfig]:
    return {
        "strict": StateConfig(rigor=1.0, creativity=0.0, defensiveness=0.9),
        "creative": StateConfig(rigor=0.1, creativity=1.0, defensiveness=0.1),
        "defensive": StateConfig(rigor=0.7, creativity=0.1, defensiveness=1.0),
        "balanced": StateConfig(rigor=0.55, creativity=0.45, defensiveness=0.55),
    }


def _rule_profiles(model_dim: int, device: torch.device) -> dict[str, dict]:
    base_rules = [
        RuleItem("r-hard", "hard", "unsafe", "block", 1.0, "stage16"),
        RuleItem("r-soft", "soft", "evidence", "prefer evidence", 0.7, "stage16"),
        RuleItem("r-conflict", "conflict", "always|never", "record conflict", 1.0, "stage16"),
    ]
    texts = {
        "empty": "plain",
        "soft": "evidence present",
        "hard_block": "unsafe request",
        "conflict": "always and never",
    }
    results: dict[str, dict] = {}
    for name, text in texts.items():
        rules = [] if name == "empty" else base_rules
        engine = RuleEngineTorch(rules, model_dim=model_dim, device=device)
        results[name] = {
            "vectors": engine.encode_vectors(),
            "result": engine.evaluate(text),
            "text": text,
        }
    return results


@torch.no_grad()
def _pooled(model: CivilizationTransformerTorch, samples: list[LogicSample], device: torch.device, context: dict) -> tuple[np.ndarray, object]:
    output = model(samples_to_tensor(samples, device), **context)
    pooled = output.hidden_states[-1].mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled hidden states contain NaN or Inf")
    return pooled.detach().cpu().numpy(), output


def _accuracy(model: CivilizationTransformerTorch, datasets: dict[str, list[LogicSample]], device: torch.device, train_per_label: int, context: dict) -> dict[str, float]:
    train_samples, _ = _split_by_label(datasets["canonical"], train_per_label)
    train_vectors, _ = _pooled(model, train_samples, device, context)
    train_labels = np.array([sample.label for sample in train_samples])
    result: dict[str, float] = {}
    for variant in LOGIC_VARIANTS:
        _, test_samples = _split_by_label(datasets[variant], train_per_label)
        test_vectors, _ = _pooled(model, test_samples, device, context)
        test_labels = np.array([sample.label for sample in test_samples])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, test_vectors, test_labels)
        result[variant] = codebook.nearest_neighbor_accuracy
    return result


def _context_checks(model: CivilizationTransformerTorch, samples: list[LogicSample], device: torch.device, context: dict, rule_result) -> dict:
    sample_tensor = samples_to_tensor(samples[:8], device)
    output = model(sample_tensor, **context)
    traces = output.civilization_traces
    memory_count = context["memory_vectors"].shape[0]
    ablation = context.get("ablation_config")
    memory_attention_ok = True
    if memory_count > 0:
        if ablation is not None and not ablation.use_memory_path:
            memory_attention_ok = all(trace.memory_attention.shape[-1] == 0 for trace in traces)
        else:
            memory_attention_ok = all(
                trace.memory_attention.shape[-1] == memory_count
                and torch.allclose(trace.memory_attention.sum(dim=-1), torch.ones_like(trace.memory_attention.sum(dim=-1)), atol=1e-5)
                for trace in traces
            )
    creative_context = {**context, "state": StateConfig(rigor=0.1, creativity=0.9, defensiveness=0.1)}
    creative = model(sample_tensor, **creative_context)
    empty_rules = {**context, "rule_vectors": torch.zeros((0, model.config.model_dim), device=device)}
    empty_rule_output = model(sample_tensor, **empty_rules)
    return {
        "trace_count": len(traces),
        "memory_attention_ok": bool(memory_attention_ok),
        "state_output_delta": float(torch.linalg.vector_norm(output.logits - creative.logits).detach().cpu().item()),
        "rule_trace_delta": abs(traces[-1].rule_influence_norm - empty_rule_output.civilization_traces[-1].rule_influence_norm),
        "finite_logits": bool(torch.isfinite(output.logits).all().item()),
        "finite_hidden": bool(all(torch.isfinite(hidden).all().item() for hidden in output.hidden_states)),
        "hard_block_recorded": bool(rule_result.violations),
        "conflict_recorded": bool(rule_result.conflicts),
        "soft_recorded": bool(rule_result.soft_matches),
    }


def run_civilization_stress_test(
    output_dir: str | Path = "artifacts/torch-line/civilization_stress_test",
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606, 707, 808, 909),
    samples_per_label: int = 120,
    train_per_label: int = 80,
    seq_lens: tuple[int, ...] = (18, 24),
    model_sizes: tuple[str, ...] = ("small", "medium"),
    training_steps: int = 50,
    device: str | torch.device | None = "cpu",
    run_regressions: bool = True,
) -> dict:
    target_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage11 = run_codebook_generalization_matrix(
        output_dir=output_path / "stage11_regression",
        seeds=(202, 303, 404),
        samples_per_label=60,
        train_per_label=40,
        device=target_device,
        template_bank="expanded_v1",
        model_training_mode="cross_template_alignment",
        max_seq_len=18,
    )
    stage13 = None
    stage14 = None
    stage15 = None
    if run_regressions:
        stage13 = run_hidden_injection_analysis(output_dir=output_path / "stage13_regression", seeds=(202,), samples_per_label=30, train_per_label=20, max_seq_len=18, device=target_device)
        stage14 = run_fusion_injection_analysis(output_dir=output_path / "stage14_regression", seeds=(202,), samples_per_label=30, train_per_label=20, max_seq_len=18, device=target_device, run_regressions=False)
        stage15 = run_civilization_transformer_experiment(output_dir=output_path / "stage15_regression", seeds=(202,), samples_per_label=30, train_per_label=20, max_seq_len=18, device=target_device, run_regressions=False)

    runs: list[dict] = []
    context_rows: list[dict] = []
    for size_name in model_sizes:
        size = MODEL_SIZES[size_name]
        for seq_len in seq_lens:
            for seed in seeds:
                datasets, tokenizer = build_logic_variant_datasets(samples_per_label=samples_per_label, max_seq_len=seq_len, seed=seed, template_bank="expanded_v1")
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
                training = run_alignment_training(
                    model,
                    datasets,
                    target_device,
                    seed=seed,
                    train_per_label=train_per_label,
                    steps=training_steps,
                    forward_kwargs=train_context,
                )
                accuracies: list[dict] = []
                _, trace_samples = _split_by_label(datasets["canonical"], train_per_label)
                for memory_name, memory_vectors in memories.items():
                    for state_name, state in states.items():
                        for rule_name, rule_info in rules.items():
                            context = {"memory_vectors": memory_vectors, "state": state, "rule_vectors": rule_info["vectors"]}
                            accuracy = _accuracy(model, datasets, target_device, train_per_label, context)
                            checks = _context_checks(model, trace_samples, target_device, context, rule_info["result"])
                            row = {
                                "seed": seed,
                                "size": size_name,
                                "seq_len": seq_len,
                                "memory": memory_name,
                                "state": state_name,
                                "rules": rule_name,
                                "accuracy": accuracy,
                                "checks": checks,
                            }
                            context_rows.append(row)
                            accuracies.append(row)
                runs.append(
                    {
                        "seed": seed,
                        "size": size_name,
                        "seq_len": seq_len,
                        "training": {
                            "initial_total_loss": training.initial_total_loss,
                            "final_total_loss": training.final_total_loss,
                            "total_loss_decreased": training.total_loss_decreased,
                            "initial_classification_loss": training.initial_classification_loss,
                            "final_classification_loss": training.final_classification_loss,
                            "classification_loss_decreased": training.classification_loss_decreased,
                        },
                        "contexts": accuracies,
                    }
                )

    summary = _summarize(runs, context_rows, stage11, stage13, stage14, stage15, seeds, samples_per_label, train_per_label, seq_lens, model_sizes)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_context_csv(output_path / "context_metrics.csv", context_rows)
    return summary


def _summarize(
    runs: list[dict],
    context_rows: list[dict],
    stage11: dict,
    stage13: dict | None,
    stage14: dict | None,
    stage15: dict | None,
    seeds: tuple[int, ...],
    samples_per_label: int,
    train_per_label: int,
    seq_lens: tuple[int, ...],
    model_sizes: tuple[str, ...],
) -> dict:
    averages: dict[str, dict[str, float]] = {}
    for size_name in model_sizes:
        rows = [row for row in context_rows if row["size"] == size_name]
        averages[size_name] = {
            variant: float(np.mean([row["accuracy"][variant] for row in rows]))
            for variant in LOGIC_VARIANTS
        }
    failures = {
        "nan_inf": sum(1 for row in context_rows if not row["checks"]["finite_logits"] or not row["checks"]["finite_hidden"]),
        "trace_missing": sum(1 for row in context_rows if row["checks"]["trace_count"] <= 0),
        "memory_attention_bad": sum(1 for row in context_rows if not row["checks"]["memory_attention_ok"]),
        "state_no_effect": sum(1 for row in context_rows if row["checks"]["state_output_delta"] <= 1e-6),
        "rule_no_effect": sum(1 for row in context_rows if row["rules"] != "empty" and row["checks"]["rule_trace_delta"] <= 1e-6),
        "hard_block_missing": sum(1 for row in context_rows if row["rules"] == "hard_block" and not row["checks"]["hard_block_recorded"]),
    }
    gates_by_size = {
        size_name: {
            "canonical": values["canonical"] >= 0.90,
            "synonym": values["synonym"] >= 0.85,
            "perturbed": values["perturbed"] >= 0.85,
            "masked_keywords": values["masked_keywords"] >= 0.85,
        }
        for size_name, values in averages.items()
    }
    losses_ok = all(run["training"]["total_loss_decreased"] and run["training"]["classification_loss_decreased"] for run in runs)
    stage13_ok = True if stage13 is None else stage13["passes_stage_gate"]
    stage14_ok = True if stage14 is None else stage14["passes_stage_gate"]
    stage15_ok = True if stage15 is None else stage15["passes_stage_gate"]
    passes = bool(
        stage11["allows_hidden_state_injection_planning"]
        and stage13_ok
        and stage14_ok
        and stage15_ok
        and losses_ok
        and all(all(gates.values()) for gates in gates_by_size.values())
        and all(value == 0 for value in failures.values())
    )
    return {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "num_training_runs": len(runs),
        "num_context_rows": len(context_rows),
        "average_accuracy_by_size": averages,
        "gates_by_size": gates_by_size,
        "losses_decreased": losses_ok,
        "failures": failures,
        "stage11_regression": {
            "allows_hidden_state_injection_planning": stage11["allows_hidden_state_injection_planning"],
            "trained_mean_average_accuracy_by_variant": stage11["trained_mean_average_accuracy_by_variant"],
            "masked_keyword_drop": stage11["masked_keyword_drop"],
        },
        "stage13_regression": None if stage13 is None else {"passes_stage_gate": stage13["passes_stage_gate"]},
        "stage14_regression": None if stage14 is None else {"passes_stage_gate": stage14["passes_stage_gate"]},
        "stage15_regression": None if stage15 is None else {"passes_stage_gate": stage15["passes_stage_gate"]},
        "passes_stage_gate": passes,
    }


def _write_context_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "size", "seq_len", "memory", "state", "rules", *LOGIC_VARIANTS, "finite_logits", "finite_hidden", "memory_attention_ok", "state_output_delta", "rule_trace_delta"])
        for row in rows:
            writer.writerow([
                row["seed"],
                row["size"],
                row["seq_len"],
                row["memory"],
                row["state"],
                row["rules"],
                *[row["accuracy"][variant] for variant in LOGIC_VARIANTS],
                row["checks"]["finite_logits"],
                row["checks"]["finite_hidden"],
                row["checks"]["memory_attention_ok"],
                row["checks"]["state_output_delta"],
                row["checks"]["rule_trace_delta"],
            ])


def main() -> None:
    summary = run_civilization_stress_test()
    print("average_accuracy_by_size", summary["average_accuracy_by_size"])
    print("gates_by_size", summary["gates_by_size"])
    print("losses_decreased", summary["losses_decreased"])
    print("failures", summary["failures"])
    print("num_training_runs", summary["num_training_runs"])
    print("num_context_rows", summary["num_context_rows"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
