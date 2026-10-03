from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..device import resolve_device
from ..memory import MemoryEncoderTorch, MemoryItem
from ..model import CivilizationTransformerTorch, MiniTransformerTorch, TransformerConfigTorch
from ..rules import RuleEngineTorch, RuleItem
from ..state import StateConfig
from .alignment import run_alignment_training
from .codebook import build_logic_codebook_train_test, run_codebook_generalization_matrix
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, LogicSample, build_logic_variant_datasets
from .hidden_states import samples_to_tensor
from .run_fusion_injection import run_fusion_injection_analysis
from .run_hidden_injection import run_hidden_injection_analysis


def _split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


def _civilization_context(model_dim: int, device: torch.device) -> dict:
    memories = [
        MemoryItem("m-cause", "causal memory", "cause therefore leads effect", "causality", 1.0, 1.0),
        MemoryItem("m-neg", "negation memory", "not never deny reject", "negation", 0.9, 1.0),
        MemoryItem("m-conflict", "conflict memory", "always never contradict", "conflict", 0.9, 1.0),
        MemoryItem("m-priority", "priority memory", "critical priority urgent override", "priority", 1.0, 1.0),
        MemoryItem("m-condition", "condition memory", "if then threshold when", "condition", 0.9, 1.0),
    ]
    memory_vectors = MemoryEncoderTorch(model_dim, device=device).encode(memories)
    rules = RuleEngineTorch(
        [
            RuleItem("r-hard", "hard", "unsafe", "block", 1.0, "stage15"),
            RuleItem("r-soft", "soft", "evidence", "prefer", 0.7, "stage15"),
            RuleItem("r-conflict", "conflict", "always|never", "record", 1.0, "stage15"),
        ],
        model_dim=model_dim,
        device=device,
    ).encode_vectors()
    return {
        "memory_vectors": memory_vectors,
        "state": StateConfig(rigor=0.75, creativity=0.25, defensiveness=0.75),
        "rule_vectors": rules,
    }


@torch.no_grad()
def _pooled_vectors(model, samples: list[LogicSample], device: torch.device, forward_kwargs: dict | None = None) -> tuple[np.ndarray, object]:
    output = model(samples_to_tensor(samples, device), **(forward_kwargs or {}))
    pooled = output.hidden_states[-1].mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled hidden states contain NaN or Inf")
    return pooled.detach().cpu().numpy(), output


def _accuracy_for_variants(model, datasets: dict[str, list[LogicSample]], device: torch.device, train_per_label: int, forward_kwargs: dict | None = None) -> tuple[dict[str, float], dict[str, object]]:
    canonical_train, _ = _split_by_label(datasets["canonical"], train_per_label)
    train_vectors, _ = _pooled_vectors(model, canonical_train, device, forward_kwargs=forward_kwargs)
    train_labels = np.array([sample.label for sample in canonical_train])
    details: dict[str, object] = {"train_shape": list(train_vectors.shape)}
    accuracies: dict[str, float] = {}
    for variant in LOGIC_VARIANTS:
        _, test_samples = _split_by_label(datasets[variant], train_per_label)
        test_vectors, output = _pooled_vectors(model, test_samples, device, forward_kwargs=forward_kwargs)
        test_labels = np.array([sample.label for sample in test_samples])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, test_vectors, test_labels)
        accuracies[variant] = codebook.nearest_neighbor_accuracy
        details[variant] = {
            "test_shape": list(test_vectors.shape),
            "macro_accuracy": codebook.macro_accuracy,
            "adjusted_rand_score": codebook.adjusted_rand_score,
            "easiest_confusion_pair": codebook.easiest_confusion_pair,
        }
    return accuracies, details


def _trace_checks(model: CivilizationTransformerTorch, datasets: dict[str, list[LogicSample]], device: torch.device, context: dict, train_per_label: int) -> dict:
    _, samples = _split_by_label(datasets["canonical"], train_per_label)
    output = model(samples_to_tensor(samples[:8], device), **context)
    traces = output.civilization_traces
    memory_attention_ok = all(
        trace.memory_attention.shape[-1] == context["memory_vectors"].shape[0]
        and torch.allclose(trace.memory_attention.sum(dim=-1), torch.ones_like(trace.memory_attention.sum(dim=-1)), atol=1e-5)
        for trace in traces
    )
    empty_output = model(
        samples_to_tensor(samples[:8], device),
        memory_vectors=torch.zeros((0, model.config.model_dim), device=device),
        state=context["state"],
        rule_vectors=torch.zeros((0, model.config.model_dim), device=device),
    )
    creative_context = {**context, "state": StateConfig(rigor=0.1, creativity=0.9, defensiveness=0.1)}
    creative_output = model(samples_to_tensor(samples[:8], device), **creative_context)
    empty_rule_context = {**context, "rule_vectors": torch.zeros((0, model.config.model_dim), device=device)}
    empty_rule_output = model(samples_to_tensor(samples[:8], device), **empty_rule_context)
    return {
        "trace_count": len(traces),
        "memory_attention_ok": bool(memory_attention_ok),
        "state_output_delta": float(torch.linalg.vector_norm(output.logits - creative_output.logits).detach().cpu().item()),
        "rule_trace_delta": abs(traces[-1].rule_influence_norm - empty_rule_output.civilization_traces[-1].rule_influence_norm),
        "empty_memory_rule_forward_ok": bool(torch.isfinite(empty_output.logits).all().item()),
        "finite_logits": bool(torch.isfinite(output.logits).all().item()),
        "finite_hidden": bool(all(torch.isfinite(hidden).all().item() for hidden in output.hidden_states)),
    }


def run_civilization_transformer_experiment(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/civilization_transformer",
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606),
    samples_per_label: int = 80,
    train_per_label: int = 50,
    max_seq_len: int = 18,
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
        max_seq_len=max_seq_len,
    )
    stage13 = None
    stage14 = None
    if run_regressions:
        stage13 = run_hidden_injection_analysis(output_dir=output_path / "stage13_regression", seeds=(202,), samples_per_label=30, train_per_label=20, max_seq_len=max_seq_len, device=target_device)
        stage14 = run_fusion_injection_analysis(output_dir=output_path / "stage14_regression", seeds=(202,), samples_per_label=30, train_per_label=20, max_seq_len=max_seq_len, device=target_device, run_regressions=False)

    runs: list[dict] = []
    for seed in seeds:
        datasets, tokenizer = build_logic_variant_datasets(samples_per_label=samples_per_label, max_seq_len=max_seq_len, seed=seed, template_bank="expanded_v1")
        config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=max_seq_len, seed=seed)

        torch.manual_seed(seed)
        mini = MiniTransformerTorch(config).to(target_device)
        mini_training = run_alignment_training(mini, datasets, target_device, seed=seed, train_per_label=train_per_label, steps=80)
        mini_accuracy, mini_details = _accuracy_for_variants(mini, datasets, target_device, train_per_label)

        torch.manual_seed(seed)
        civilization = CivilizationTransformerTorch(config).to(target_device)
        context = _civilization_context(config.model_dim, target_device)
        civ_training = run_alignment_training(
            civilization,
            datasets,
            target_device,
            seed=seed,
            train_per_label=train_per_label,
            steps=80,
            forward_kwargs=context,
        )
        civ_accuracy, civ_details = _accuracy_for_variants(civilization, datasets, target_device, train_per_label, forward_kwargs=context)
        trace_checks = _trace_checks(civilization, datasets, target_device, context, train_per_label)
        runs.append(
            {
                "seed": seed,
                "mini_training": {
                    "initial_total_loss": mini_training.initial_total_loss,
                    "final_total_loss": mini_training.final_total_loss,
                    "total_loss_decreased": mini_training.total_loss_decreased,
                    "initial_classification_loss": mini_training.initial_classification_loss,
                    "final_classification_loss": mini_training.final_classification_loss,
                    "classification_loss_decreased": mini_training.classification_loss_decreased,
                },
                "civilization_training": {
                    "initial_total_loss": civ_training.initial_total_loss,
                    "final_total_loss": civ_training.final_total_loss,
                    "total_loss_decreased": civ_training.total_loss_decreased,
                    "initial_classification_loss": civ_training.initial_classification_loss,
                    "final_classification_loss": civ_training.final_classification_loss,
                    "classification_loss_decreased": civ_training.classification_loss_decreased,
                },
                "mini_accuracy": mini_accuracy,
                "civilization_accuracy": civ_accuracy,
                "accuracy_delta_vs_mini": {variant: civ_accuracy[variant] - mini_accuracy[variant] for variant in LOGIC_VARIANTS},
                "mini_details": mini_details,
                "civilization_details": civ_details,
                "trace_checks": trace_checks,
            }
        )

    summary = _summarize(runs, stage11, stage13, stage14, seeds, samples_per_label, train_per_label)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_path / "variant_accuracy.csv", runs)
    return summary


def _summarize(runs: list[dict], stage11: dict, stage13: dict | None, stage14: dict | None, seeds: tuple[int, ...], samples_per_label: int, train_per_label: int) -> dict:
    averages = {
        variant: float(np.mean([run["civilization_accuracy"][variant] for run in runs]))
        for variant in LOGIC_VARIANTS
    }
    mini_averages = {
        variant: float(np.mean([run["mini_accuracy"][variant] for run in runs]))
        for variant in LOGIC_VARIANTS
    }
    trace_ok = all(
        run["trace_checks"]["trace_count"] > 0
        and run["trace_checks"]["memory_attention_ok"]
        and run["trace_checks"]["state_output_delta"] > 1e-6
        and run["trace_checks"]["rule_trace_delta"] > 1e-6
        and run["trace_checks"]["empty_memory_rule_forward_ok"]
        and run["trace_checks"]["finite_logits"]
        and run["trace_checks"]["finite_hidden"]
        for run in runs
    )
    losses_ok = all(run["civilization_training"]["total_loss_decreased"] and run["civilization_training"]["classification_loss_decreased"] for run in runs)
    gates = {
        "canonical": averages["canonical"] >= 0.90,
        "synonym": averages["synonym"] >= 0.80,
        "perturbed": averages["perturbed"] >= 0.80,
        "masked_keywords": averages["masked_keywords"] >= 0.80,
    }
    stage13_ok = True if stage13 is None else stage13["passes_stage_gate"]
    stage14_ok = True if stage14 is None else stage14["passes_stage_gate"]
    return {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "civilization_average_accuracy_by_variant": averages,
        "mini_average_accuracy_by_variant": mini_averages,
        "accuracy_delta_vs_mini": {variant: averages[variant] - mini_averages[variant] for variant in LOGIC_VARIANTS},
        "trace_checks_passed": trace_ok,
        "losses_decreased": losses_ok,
        "variant_gates": gates,
        "stage11_regression": {
            "allows_hidden_state_injection_planning": stage11["allows_hidden_state_injection_planning"],
            "trained_mean_average_accuracy_by_variant": stage11["trained_mean_average_accuracy_by_variant"],
            "masked_keyword_drop": stage11["masked_keyword_drop"],
        },
        "stage13_regression": None if stage13 is None else {
            "passes_stage_gate": stage13["passes_stage_gate"],
            "target_hit_rate": stage13["target_hit_rate"],
            "zero_equivalence_rate": stage13["zero_equivalence_rate"],
        },
        "stage14_regression": None if stage14 is None else {
            "passes_stage_gate": stage14["passes_stage_gate"],
            "alpha_zero_equivalence_rate": stage14["alpha_zero_equivalence_rate"],
            "hard_rule_block_rate": stage14["hard_rule_block_rate"],
        },
        "passes_stage_gate": bool(stage11["allows_hidden_state_injection_planning"] and stage13_ok and stage14_ok and trace_ok and losses_ok and all(gates.values())),
        "runs": runs,
    }


def _write_csv(path: Path, runs: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", "variant", "accuracy"])
        for run in runs:
            for variant in LOGIC_VARIANTS:
                writer.writerow([run["seed"], "mini", variant, run["mini_accuracy"][variant]])
                writer.writerow([run["seed"], "civilization", variant, run["civilization_accuracy"][variant]])


def main() -> None:
    summary = run_civilization_transformer_experiment()
    print("civilization_average_accuracy_by_variant", summary["civilization_average_accuracy_by_variant"])
    print("mini_average_accuracy_by_variant", summary["mini_average_accuracy_by_variant"])
    print("accuracy_delta_vs_mini", summary["accuracy_delta_vs_mini"])
    print("trace_checks_passed", summary["trace_checks_passed"])
    print("losses_decreased", summary["losses_decreased"])
    print("variant_gates", summary["variant_gates"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
