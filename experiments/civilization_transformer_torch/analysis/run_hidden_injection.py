from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..device import resolve_device
from ..model import MiniTransformerTorch, TransformerConfigTorch
from .alignment import run_alignment_training
from .codebook import build_logic_codebook_train_test, run_codebook_generalization_matrix
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, LogicSample, build_logic_variant_datasets
from .hidden_states import samples_to_tensor
from .injection import LogicCodeInjector, nearest_centroid_label


def _split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


@torch.no_grad()
def _pooled_vectors(model: MiniTransformerTorch, samples: list[LogicSample], device: torch.device) -> np.ndarray:
    output = model(samples_to_tensor(samples, device))
    pooled = output.hidden_states[-1].mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled hidden states contain NaN or Inf")
    return pooled.detach().cpu().numpy()


def _wrong_label(label: str) -> str:
    index = LOGIC_LABELS.index(label)
    return LOGIC_LABELS[(index + 1) % len(LOGIC_LABELS)]


def _evaluate_sample(
    model: MiniTransformerTorch,
    sample: LogicSample,
    centroids: dict[str, np.ndarray],
    device: torch.device,
    injection_layer: int,
    alpha: float,
    strong_alpha: float,
) -> list[dict]:
    input_ids = samples_to_tensor([sample], device)
    baseline = model(input_ids)
    baseline_vector = baseline.hidden_states[-1].mean(dim=1).detach().cpu().numpy()[0]
    baseline_label, baseline_distance = nearest_centroid_label(baseline_vector, centroids)
    rows: list[dict] = []

    scenarios = [
        ("correct", sample.label, alpha, "residual_norm"),
        ("wrong", _wrong_label(sample.label), alpha, "residual_norm"),
        ("zero", sample.label, 0.0, "residual_norm"),
        ("strong", sample.label, strong_alpha, "additive"),
    ]
    for scenario, target_label, scenario_alpha, strategy in scenarios:
        injector = LogicCodeInjector(
            centroids=centroids,
            target_label=target_label,
            model_dim=model.config.model_dim,
            layer_index=injection_layer,
            alpha=scenario_alpha,
            strategy=strategy,
            position="all",
        )
        injected = model(input_ids, hidden_injection_hook=injector)
        injected_vector = injected.hidden_states[-1].mean(dim=1).detach().cpu().numpy()[0]
        injected_label, injected_distance = nearest_centroid_label(injected_vector, centroids)
        target_distance_before = float(np.linalg.norm(baseline_vector - centroids[target_label]))
        target_distance_after = float(np.linalg.norm(injected_vector - centroids[target_label]))
        logits_delta_norm = float(torch.linalg.vector_norm(injected.logits - baseline.logits).detach().cpu().item())
        hidden_delta_norm = float(np.linalg.norm(injected_vector - baseline_vector))
        trace = injected.injection_traces[0] if injected.injection_traces else {}
        rows.append(
            {
                "variant": sample.variant,
                "true_label": sample.label,
                "template_id": sample.template_id,
                "scenario": scenario,
                "target_label": target_label,
                "alpha": scenario_alpha,
                "strategy": strategy,
                "baseline_nearest_label": baseline_label,
                "baseline_distance": baseline_distance,
                "injected_nearest_label": injected_label,
                "injected_distance": injected_distance,
                "target_distance_before": target_distance_before,
                "target_distance_after": target_distance_after,
                "target_distance_reduced": target_distance_after <= target_distance_before,
                "target_hit": injected_label == target_label,
                "baseline_hit": baseline_label == target_label,
                "drifted_to_wrong_label": scenario == "wrong" and injected_label == target_label,
                "logits_delta_norm": logits_delta_norm,
                "hidden_delta_norm": hidden_delta_norm,
                "zero_equivalent": scenario != "zero" or (logits_delta_norm == 0.0 and hidden_delta_norm == 0.0),
                "hidden_norm_ratio": trace.get("hidden_norm_ratio"),
                "warning": trace.get("warning"),
                "trace": trace,
            }
        )
    return rows


def run_hidden_injection_analysis(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/hidden_injection",
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 60,
    train_per_label: int = 40,
    max_seq_len: int = 18,
    alpha: float = 0.25,
    strong_alpha: float = 6.0,
    device: str | torch.device | None = "cpu",
) -> dict:
    target_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage11 = run_codebook_generalization_matrix(
        output_dir=output_path / "stage11_regression",
        seeds=seeds,
        samples_per_label=samples_per_label,
        train_per_label=train_per_label,
        device=target_device,
        template_bank="expanded_v1",
        model_training_mode="cross_template_alignment",
        max_seq_len=max_seq_len,
    )
    rows: list[dict] = []
    training_summaries: list[dict] = []
    for seed in seeds:
        datasets, tokenizer = build_logic_variant_datasets(
            samples_per_label=samples_per_label,
            max_seq_len=max_seq_len,
            seed=seed,
            template_bank="expanded_v1",
        )
        config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=max_seq_len, seed=seed)
        torch.manual_seed(seed)
        model = MiniTransformerTorch(config).to(target_device)
        training = run_alignment_training(model, datasets, target_device, seed=seed, train_per_label=train_per_label)
        training_summaries.append(
            {
                "seed": seed,
                "initial_total_loss": training.initial_total_loss,
                "final_total_loss": training.final_total_loss,
                "total_loss_decreased": training.total_loss_decreased,
                "initial_classification_loss": training.initial_classification_loss,
                "final_classification_loss": training.final_classification_loss,
                "classification_loss_decreased": training.classification_loss_decreased,
            }
        )
        canonical_train, _ = _split_by_label(datasets["canonical"], train_per_label)
        train_vectors = _pooled_vectors(model, canonical_train, target_device)
        train_labels = np.array([sample.label for sample in canonical_train])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, train_vectors, train_labels)
        centroids = codebook.centroids
        injection_layer = config.num_layers - 1
        for variant in LOGIC_VARIANTS:
            _, test_samples = _split_by_label(datasets[variant], train_per_label)
            for sample_index, sample in enumerate(test_samples):
                for row in _evaluate_sample(model, sample, centroids, target_device, injection_layer, alpha, strong_alpha):
                    row.update({"seed": seed, "sample_index": sample_index, "injection_layer": injection_layer})
                    rows.append(row)

    summary = _summarize(rows, stage11, training_summaries, seeds, samples_per_label, train_per_label, alpha, strong_alpha)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "injection_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_rows_csv(output_path / "injection_results.csv", rows)
    return summary


def _summarize(
    rows: list[dict],
    stage11: dict,
    training_summaries: list[dict],
    seeds: tuple[int, ...],
    samples_per_label: int,
    train_per_label: int,
    alpha: float,
    strong_alpha: float,
) -> dict:
    by_scenario = {scenario: [row for row in rows if row["scenario"] == scenario] for scenario in ("correct", "wrong", "zero", "strong")}
    correct = by_scenario["correct"]
    wrong = by_scenario["wrong"]
    zero = by_scenario["zero"]
    strong = by_scenario["strong"]
    target_hit_rate = sum(row["target_hit"] for row in correct) / len(correct)
    baseline_hit_rate = sum(row["baseline_hit"] for row in correct) / len(correct)
    wrong_drift_rate = sum(row["drifted_to_wrong_label"] for row in wrong) / len(wrong)
    zero_equivalence_rate = sum(row["zero_equivalent"] for row in zero) / len(zero)
    warning_count = sum(1 for row in strong if row["warning"] is not None)
    norm_exceeded_count = sum(1 for row in correct if row["hidden_norm_ratio"] is not None and row["hidden_norm_ratio"] > 2.0)
    target_distance_reduction_rate = sum(row["target_distance_reduced"] for row in correct) / len(correct)
    summary = {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "alpha": alpha,
        "strong_alpha": strong_alpha,
        "num_rows": len(rows),
        "training": training_summaries,
        "stage11_regression": {
            "allows_hidden_state_injection_planning": stage11["allows_hidden_state_injection_planning"],
            "trained_mean_average_accuracy_by_variant": stage11["trained_mean_average_accuracy_by_variant"],
            "masked_keyword_drop": stage11["masked_keyword_drop"],
        },
        "target_hit_rate": target_hit_rate,
        "baseline_hit_rate_for_target": baseline_hit_rate,
        "target_hit_not_worse_than_baseline": target_hit_rate >= baseline_hit_rate,
        "target_distance_reduction_rate": target_distance_reduction_rate,
        "wrong_label_drift_rate": wrong_drift_rate,
        "zero_equivalence_rate": zero_equivalence_rate,
        "strong_warning_count": warning_count,
        "correct_injection_norm_exceeded_count": norm_exceeded_count,
        "passes_stage_gate": bool(
            stage11["allows_hidden_state_injection_planning"]
            and zero_equivalence_rate == 1.0
            and target_hit_rate > 0.80
            and target_hit_rate >= baseline_hit_rate
            and norm_exceeded_count == 0
            and warning_count > 0
        ),
    }
    return summary


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [key for key in rows[0].keys() if key != "trace"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    summary = run_hidden_injection_analysis()
    print("target_hit_rate", summary["target_hit_rate"])
    print("baseline_hit_rate_for_target", summary["baseline_hit_rate_for_target"])
    print("target_hit_not_worse_than_baseline", summary["target_hit_not_worse_than_baseline"])
    print("target_distance_reduction_rate", summary["target_distance_reduction_rate"])
    print("wrong_label_drift_rate", summary["wrong_label_drift_rate"])
    print("zero_equivalence_rate", summary["zero_equivalence_rate"])
    print("strong_warning_count", summary["strong_warning_count"])
    print("correct_injection_norm_exceeded_count", summary["correct_injection_norm_exceeded_count"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
