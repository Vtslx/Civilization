from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import torch

from ..device import resolve_device
from ..model import MiniTransformerTorch, TransformerConfigTorch
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, build_logic_dataset, build_logic_variant_datasets
from .hidden_states import collect_hidden_state_representations
from .pipeline import _train_model_for_analysis


@dataclass(frozen=True)
class NearestNeighborResult:
    sample_index: int
    true_label: str
    predicted_label: str
    distance: float
    correct: bool


@dataclass(frozen=True)
class LogicCodebook:
    centroids: dict[str, np.ndarray]
    sample_counts: dict[str, int]
    intra_label_distance: dict[str, float]
    centroid_distances: dict[str, float]
    nearest_neighbors: list[NearestNeighborResult]
    nearest_neighbor_accuracy: float
    per_label_accuracy: dict[str, float]
    macro_accuracy: float
    adjusted_rand_score: float
    confusion_matrix: dict[str, dict[str, int]]
    easiest_confusion_pair: str


@dataclass(frozen=True)
class CodebookRunResult:
    output_dir: Path
    metrics_path: Path
    classification_report_path: Path
    centroid_distances_path: Path
    confusion_matrix_path: Path
    metrics: dict


@dataclass(frozen=True)
class GeneralizationRunResult:
    output_dir: Path
    metrics_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    variant_comparison_path: Path
    metrics: dict


def build_logic_codebook(vectors: np.ndarray, labels: np.ndarray) -> LogicCodebook:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_samples, model_dim]")
    if len(vectors) != len(labels):
        raise ValueError("vectors and labels must have the same length")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or Inf")

    centroids: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int] = {}
    intra_label_distance: dict[str, float] = {}
    for label in LOGIC_LABELS:
        label_vectors = vectors[labels == label]
        if len(label_vectors) == 0:
            raise ValueError(f"missing vectors for label {label}")
        centroid = label_vectors.mean(axis=0)
        centroids[label] = centroid
        sample_counts[label] = int(len(label_vectors))
        intra_label_distance[label] = float(np.linalg.norm(label_vectors - centroid[None, :], axis=1).mean())

    centroid_distances: dict[str, float] = {}
    for left_index, left in enumerate(LOGIC_LABELS):
        for right in LOGIC_LABELS[left_index + 1 :]:
            centroid_distances[f"{left}__{right}"] = float(np.linalg.norm(centroids[left] - centroids[right]))

    nearest_neighbors: list[NearestNeighborResult] = []
    predictions: list[str] = []
    confusion_matrix = {label: {candidate: 0 for candidate in LOGIC_LABELS} for label in LOGIC_LABELS}
    for index, vector in enumerate(vectors):
        distances = {label: float(np.linalg.norm(vector - centroid)) for label, centroid in centroids.items()}
        predicted = min(distances, key=distances.get)
        true_label = str(labels[index])
        predictions.append(predicted)
        confusion_matrix[true_label][predicted] += 1
        nearest_neighbors.append(
            NearestNeighborResult(
                sample_index=index,
                true_label=true_label,
                predicted_label=predicted,
                distance=distances[predicted],
                correct=predicted == true_label,
            )
        )

    per_label_accuracy: dict[str, float] = {}
    for label in LOGIC_LABELS:
        label_results = [result for result in nearest_neighbors if result.true_label == label]
        per_label_accuracy[label] = sum(result.correct for result in label_results) / len(label_results)
    nearest_neighbor_accuracy = sum(result.correct for result in nearest_neighbors) / len(nearest_neighbors)
    macro_accuracy = float(np.mean(list(per_label_accuracy.values())))

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(labels)
    y_pred = encoder.transform(np.array(predictions))
    ari = float(adjusted_rand_score(y_true, y_pred))
    easiest_confusion_pair = _most_confused_pair(confusion_matrix)

    return LogicCodebook(
        centroids=centroids,
        sample_counts=sample_counts,
        intra_label_distance=intra_label_distance,
        centroid_distances=centroid_distances,
        nearest_neighbors=nearest_neighbors,
        nearest_neighbor_accuracy=nearest_neighbor_accuracy,
        per_label_accuracy=per_label_accuracy,
        macro_accuracy=macro_accuracy,
        adjusted_rand_score=ari,
        confusion_matrix=confusion_matrix,
        easiest_confusion_pair=easiest_confusion_pair,
    )


def build_logic_codebook_train_test(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
) -> LogicCodebook:
    if train_vectors.ndim != 2 or test_vectors.ndim != 2:
        raise ValueError("train_vectors and test_vectors must have shape [num_samples, model_dim]")
    if train_vectors.shape[1] != test_vectors.shape[1]:
        raise ValueError("train and test vectors must use the same model_dim")
    if len(train_vectors) != len(train_labels):
        raise ValueError("train vectors and labels must have the same length")
    if len(test_vectors) != len(test_labels):
        raise ValueError("test vectors and labels must have the same length")
    if not np.isfinite(train_vectors).all() or not np.isfinite(test_vectors).all():
        raise ValueError("vectors contain NaN or Inf")

    centroids: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int] = {}
    intra_label_distance: dict[str, float] = {}
    for label in LOGIC_LABELS:
        label_vectors = train_vectors[train_labels == label]
        if len(label_vectors) == 0:
            raise ValueError(f"missing train vectors for label {label}")
        centroid = label_vectors.mean(axis=0)
        centroids[label] = centroid
        sample_counts[label] = int(len(label_vectors))
        intra_label_distance[label] = float(np.linalg.norm(label_vectors - centroid[None, :], axis=1).mean())

    centroid_distances: dict[str, float] = {}
    for left_index, left in enumerate(LOGIC_LABELS):
        for right in LOGIC_LABELS[left_index + 1 :]:
            centroid_distances[f"{left}__{right}"] = float(np.linalg.norm(centroids[left] - centroids[right]))

    nearest_neighbors: list[NearestNeighborResult] = []
    predictions: list[str] = []
    confusion_matrix = {label: {candidate: 0 for candidate in LOGIC_LABELS} for label in LOGIC_LABELS}
    for index, vector in enumerate(test_vectors):
        distances = {label: float(np.linalg.norm(vector - centroid)) for label, centroid in centroids.items()}
        predicted = min(distances, key=distances.get)
        true_label = str(test_labels[index])
        predictions.append(predicted)
        confusion_matrix[true_label][predicted] += 1
        nearest_neighbors.append(
            NearestNeighborResult(
                sample_index=index,
                true_label=true_label,
                predicted_label=predicted,
                distance=distances[predicted],
                correct=predicted == true_label,
            )
        )

    per_label_accuracy: dict[str, float] = {}
    for label in LOGIC_LABELS:
        label_results = [result for result in nearest_neighbors if result.true_label == label]
        if not label_results:
            raise ValueError(f"missing test vectors for label {label}")
        per_label_accuracy[label] = sum(result.correct for result in label_results) / len(label_results)
    nearest_neighbor_accuracy = sum(result.correct for result in nearest_neighbors) / len(nearest_neighbors)
    macro_accuracy = float(np.mean(list(per_label_accuracy.values())))

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(test_labels)
    y_pred = encoder.transform(np.array(predictions))
    ari = float(adjusted_rand_score(y_true, y_pred))
    easiest_confusion_pair = _most_confused_pair(confusion_matrix)

    return LogicCodebook(
        centroids=centroids,
        sample_counts=sample_counts,
        intra_label_distance=intra_label_distance,
        centroid_distances=centroid_distances,
        nearest_neighbors=nearest_neighbors,
        nearest_neighbor_accuracy=nearest_neighbor_accuracy,
        per_label_accuracy=per_label_accuracy,
        macro_accuracy=macro_accuracy,
        adjusted_rand_score=ari,
        confusion_matrix=confusion_matrix,
        easiest_confusion_pair=easiest_confusion_pair,
    )


def _most_confused_pair(confusion_matrix: dict[str, dict[str, int]]) -> str:
    best_pair = ""
    best_count = -1
    for true_label in LOGIC_LABELS:
        for predicted_label in LOGIC_LABELS:
            if true_label == predicted_label:
                continue
            count = confusion_matrix[true_label][predicted_label]
            if count > best_count:
                best_pair = f"{true_label}->{predicted_label}"
                best_count = count
    return best_pair


def codebook_to_metrics(codebook: LogicCodebook, seed: int, model_state: str, pooling: str, representation_shape: tuple[int, int]) -> dict:
    return {
        "seed": seed,
        "model_state": model_state,
        "pooling": pooling,
        "representation_shape": list(representation_shape),
        "sample_counts": codebook.sample_counts,
        "nearest_neighbor_accuracy": codebook.nearest_neighbor_accuracy,
        "per_label_accuracy": codebook.per_label_accuracy,
        "macro_accuracy": codebook.macro_accuracy,
        "adjusted_rand_score": codebook.adjusted_rand_score,
        "intra_label_distance": codebook.intra_label_distance,
        "centroid_distances": codebook.centroid_distances,
        "easiest_confusion_pair": codebook.easiest_confusion_pair,
    }


def _write_classification_report(path: Path, codebook: LogicCodebook) -> None:
    rows = [
        {
            "sample_index": result.sample_index,
            "true_label": result.true_label,
            "predicted_label": result.predicted_label,
            "distance": result.distance,
            "correct": result.correct,
        }
        for result in codebook.nearest_neighbors
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_generalization_classification_report(path: Path, codebook: LogicCodebook, variant: str) -> None:
    rows = [
        {
            "sample_index": result.sample_index,
            "true_label": result.true_label,
            "predicted_label": result.predicted_label,
            "distance": result.distance,
            "correct": result.correct,
            "variant": variant,
        }
        for result in codebook.nearest_neighbors
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_centroid_distances(path: Path, codebook: LogicCodebook) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_pair", "distance"])
        for key, value in sorted(codebook.centroid_distances.items()):
            writer.writerow([key, value])


def _write_confusion_matrix(path: Path, codebook: LogicCodebook) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *LOGIC_LABELS])
        for label in LOGIC_LABELS:
            writer.writerow([label, *[codebook.confusion_matrix[label][candidate] for candidate in LOGIC_LABELS]])


def _representations_for_state(
    seed: int,
    samples_per_label: int,
    model_state: str,
    device: torch.device,
) -> tuple[dict[str, list[torch.Tensor]], list[str], dict]:
    samples, tokenizer = build_logic_dataset(samples_per_label=samples_per_label, max_seq_len=12, seed=seed)
    labels = [sample.label for sample in samples]
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=12, seed=seed)
    torch.manual_seed(seed)
    model = MiniTransformerTorch(config).to(device)
    losses: list[float] | None = None
    if model_state == "trained":
        losses = _train_model_for_analysis(model, device, seed)
    elif model_state != "untrained":
        raise ValueError("model_state must be trained or untrained")
    reps = collect_hidden_state_representations(model, samples, device)
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "num_samples": len(samples),
        "samples_per_label": samples_per_label,
        "model_config": {
            "vocab_size": config.vocab_size,
            "model_dim": config.model_dim,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "max_seq_len": config.max_seq_len,
        },
        "training": None if losses is None else {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
        },
    }
    return reps, labels, metadata


def _split_samples_by_label(samples: list, train_per_label: int | None = None) -> tuple[list, list]:
    train: list = []
    test: list = []
    grouped = {label: [sample for sample in samples if sample.label == label] for label in LOGIC_LABELS}
    min_count = min(len(values) for values in grouped.values())
    split = train_per_label if train_per_label is not None else max(1, min_count * 2 // 3)
    if split <= 0 or split >= min_count:
        raise ValueError("train_per_label must leave at least one test sample per label")
    for label in LOGIC_LABELS:
        train.extend(grouped[label][:split])
        test.extend(grouped[label][split:])
    return train, test


def _build_variant_model_representations(
    seed: int,
    samples_per_label: int,
    model_state: str,
    device: torch.device,
    template_bank: str = "basic",
    max_seq_len: int = 14,
    model_training_mode: str = "baseline_next_token",
    train_per_label: int = 20,
) -> tuple[dict[str, dict[str, list[torch.Tensor]]], dict[str, list[str]], dict]:
    datasets, tokenizer = build_logic_variant_datasets(
        samples_per_label=samples_per_label,
        max_seq_len=max_seq_len,
        seed=seed,
        template_bank=template_bank,
    )
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=max_seq_len, seed=seed)
    torch.manual_seed(seed)
    model = MiniTransformerTorch(config).to(device)
    losses: list[float] | None = None
    alignment: dict | None = None
    if model_state == "trained":
        if model_training_mode == "baseline_next_token":
            losses = _train_model_for_analysis(model, device, seed)
        elif model_training_mode == "cross_template_alignment":
            from .alignment import run_alignment_training

            result = run_alignment_training(
                model=model,
                datasets=datasets,
                device=device,
                seed=seed,
                train_per_label=train_per_label,
            )
            alignment = {
                "seed": result.seed,
                "device": result.device,
                "steps": result.steps,
                "batch_shape": list(result.batch_shape),
                "train_samples": result.train_samples,
                "train_per_label": result.train_per_label,
                "initial_total_loss": result.initial_total_loss,
                "final_total_loss": result.final_total_loss,
                "total_loss_decreased": result.total_loss_decreased,
                "initial_classification_loss": result.initial_classification_loss,
                "final_classification_loss": result.final_classification_loss,
                "classification_loss_decreased": result.classification_loss_decreased,
                "losses": result.losses,
            }
        else:
            raise ValueError("model_training_mode must be baseline_next_token or cross_template_alignment")
    elif model_state != "untrained":
        raise ValueError("model_state must be trained or untrained")

    reps_by_variant: dict[str, dict[str, list[torch.Tensor]]] = {}
    labels_by_variant: dict[str, list[str]] = {}
    for variant, samples in datasets.items():
        labels_by_variant[variant] = [sample.label for sample in samples]
        reps_by_variant[variant] = collect_hidden_state_representations(model, samples, device)

    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "samples_per_label": samples_per_label,
        "template_bank": template_bank,
        "model_training_mode": model_training_mode,
        "variants": list(datasets.keys()),
        "model_config": {
            "vocab_size": config.vocab_size,
            "model_dim": config.model_dim,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "max_seq_len": config.max_seq_len,
        },
        "training": None if losses is None else {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
        },
        "alignment_training": alignment,
    }
    return reps_by_variant, labels_by_variant, metadata


def _variant_split_vectors(
    reps_by_variant: dict[str, dict[str, list[torch.Tensor]]],
    labels_by_variant: dict[str, list[str]],
    samples_per_label: int,
    pooling: str,
    variant: str,
    train_per_label: int,
    exclude_variant_train_from_test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pooling not in {"mean", "last"}:
        raise ValueError("pooling must be mean or last")
    train_mask: list[bool] = []
    test_mask: list[bool] = []
    for label in labels_by_variant["canonical"]:
        label_position = sum(1 for prior in labels_by_variant["canonical"][: len(train_mask)] if prior == label)
        train_mask.append(label_position < train_per_label)
    for label in labels_by_variant[variant]:
        label_position = sum(1 for prior in labels_by_variant[variant][: len(test_mask)] if prior == label)
        if variant == "canonical" or exclude_variant_train_from_test:
            test_mask.append(label_position >= train_per_label)
        else:
            test_mask.append(True)

    train_vectors = reps_by_variant["canonical"][pooling][-1].numpy()[np.array(train_mask, dtype=bool)]
    train_labels = np.array(labels_by_variant["canonical"])[np.array(train_mask, dtype=bool)]
    test_vectors = reps_by_variant[variant][pooling][-1].numpy()[np.array(test_mask, dtype=bool)]
    test_labels = np.array(labels_by_variant[variant])[np.array(test_mask, dtype=bool)]
    return train_vectors, train_labels, test_vectors, test_labels


def _write_variant_comparison(path: Path, variant_rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "seed",
            "model_state",
            "pooling",
            "variant",
            "train_accuracy",
            "test_accuracy",
            "macro_accuracy",
            "adjusted_rand_score",
            "variant_drop",
            "easiest_confusion_pair",
        ])
        for row in variant_rows:
            writer.writerow([
                row["seed"],
                row["model_state"],
                row["pooling"],
                row["variant"],
                row["train_accuracy"],
                row["test_accuracy"],
                row["macro_accuracy"],
                row["adjusted_rand_score"],
                row["variant_drop"],
                row["easiest_confusion_pair"],
            ])


def run_single_codebook_analysis(
    output_dir: str | Path,
    seed: int = 202,
    samples_per_label: int = 40,
    model_state: str = "trained",
    pooling: str = "mean",
    device: str | torch.device | None = "cpu",
) -> CodebookRunResult:
    target_device = resolve_device(device)
    reps, labels, metadata = _representations_for_state(seed, samples_per_label, model_state, target_device)
    if pooling not in {"mean", "last"}:
        raise ValueError("pooling must be mean or last")
    vectors = reps[pooling][-1].numpy()
    labels_array = np.array(labels)
    codebook = build_logic_codebook(vectors, labels_array)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "codebook_metrics.json"
    classification_report_path = output_path / "classification_report.json"
    centroid_distances_path = output_path / "centroid_distances.csv"
    confusion_matrix_path = output_path / "confusion_matrix.csv"

    metrics = {
        **metadata,
        **codebook_to_metrics(codebook, seed, model_state, pooling, tuple(vectors.shape)),
        "device": str(target_device),
        "random_baseline": 1.0 / len(LOGIC_LABELS),
        "passes_random_baseline": codebook.nearest_neighbor_accuracy > (1.0 / len(LOGIC_LABELS)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_classification_report(classification_report_path, codebook)
    _write_centroid_distances(centroid_distances_path, codebook)
    _write_confusion_matrix(confusion_matrix_path, codebook)

    return CodebookRunResult(
        output_dir=output_path,
        metrics_path=metrics_path,
        classification_report_path=classification_report_path,
        centroid_distances_path=centroid_distances_path,
        confusion_matrix_path=confusion_matrix_path,
        metrics=metrics,
    )


def run_codebook_matrix(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/codebook",
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 40,
    device: str | torch.device | None = "cpu",
) -> dict:
    output_path = Path(output_dir)
    runs: list[dict] = []
    for seed in seeds:
        for model_state in ("untrained", "trained"):
            for pooling in ("mean", "last"):
                run_dir = output_path / f"seed_{seed}" / model_state / pooling
                result = run_single_codebook_analysis(
                    output_dir=run_dir,
                    seed=seed,
                    samples_per_label=samples_per_label,
                    model_state=model_state,
                    pooling=pooling,
                    device=device,
                )
                runs.append({
                    "seed": seed,
                    "model_state": model_state,
                    "pooling": pooling,
                    "metrics_path": str(result.metrics_path),
                    "nearest_neighbor_accuracy": result.metrics["nearest_neighbor_accuracy"],
                    "macro_accuracy": result.metrics["macro_accuracy"],
                    "adjusted_rand_score": result.metrics["adjusted_rand_score"],
                    "easiest_confusion_pair": result.metrics["easiest_confusion_pair"],
                })

    trained_mean = [run["nearest_neighbor_accuracy"] for run in runs if run["model_state"] == "trained" and run["pooling"] == "mean"]
    summary = {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "num_runs": len(runs),
        "runs": runs,
        "trained_mean_average_accuracy": float(np.mean(trained_mean)),
        "trained_mean_min_accuracy": float(np.min(trained_mean)),
        "passes_stage_gate": bool(np.min(trained_mean) > 0.20 and np.mean(trained_mean) > 0.60),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _generalization_metrics(
    train_codebook: LogicCodebook,
    test_codebook: LogicCodebook,
    seed: int,
    model_state: str,
    pooling: str,
    variant: str,
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    canonical_test_accuracy: float,
    metadata: dict,
    device: torch.device,
) -> dict:
    variant_drop = canonical_test_accuracy - test_codebook.nearest_neighbor_accuracy
    metrics = {
        **metadata,
        "seed": seed,
        "device": str(device),
        "model_state": model_state,
        "pooling": pooling,
        "train_variant": "canonical",
        "test_variant": variant,
        "train_representation_shape": list(train_shape),
        "test_representation_shape": list(test_shape),
        "train_sample_counts": train_codebook.sample_counts,
        "test_sample_counts": {label: sum(1 for result in test_codebook.nearest_neighbors if result.true_label == label) for label in LOGIC_LABELS},
        "train_accuracy": train_codebook.nearest_neighbor_accuracy,
        "test_accuracy": test_codebook.nearest_neighbor_accuracy,
        "nearest_neighbor_accuracy": test_codebook.nearest_neighbor_accuracy,
        "per_label_accuracy": test_codebook.per_label_accuracy,
        "macro_accuracy": test_codebook.macro_accuracy,
        "adjusted_rand_score": test_codebook.adjusted_rand_score,
        "intra_label_distance": test_codebook.intra_label_distance,
        "centroid_distances": test_codebook.centroid_distances,
        "easiest_confusion_pair": test_codebook.easiest_confusion_pair,
        "canonical_test_accuracy": canonical_test_accuracy,
        "variant_drop": variant_drop,
        "masked_keyword_drop": variant_drop if variant == "masked_keywords" else None,
        "random_baseline": 1.0 / len(LOGIC_LABELS),
        "passes_random_baseline": test_codebook.nearest_neighbor_accuracy > (1.0 / len(LOGIC_LABELS)),
        "finite_metrics": bool(
            np.isfinite(test_codebook.nearest_neighbor_accuracy)
            and np.isfinite(test_codebook.macro_accuracy)
            and np.isfinite(test_codebook.adjusted_rand_score)
            and all(np.isfinite(value) for value in test_codebook.centroid_distances.values())
            and all(np.isfinite(value) for value in test_codebook.intra_label_distance.values())
        ),
    }
    return metrics


def run_codebook_generalization_matrix(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/codebook_generalization",
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 30,
    train_per_label: int = 20,
    device: str | torch.device | None = "cpu",
    template_bank: str = "basic",
    model_training_mode: str = "baseline_next_token",
    max_seq_len: int = 14,
) -> dict:
    if samples_per_label < 30:
        raise ValueError("samples_per_label must be at least 30 for stage 11")
    if train_per_label <= 0 or train_per_label >= samples_per_label:
        raise ValueError("train_per_label must leave test samples")

    target_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    aggregate_reports: list[dict] = []
    variant_rows: list[dict] = []

    for seed in seeds:
        for model_state in ("untrained", "trained"):
            reps_by_variant, labels_by_variant, metadata = _build_variant_model_representations(
                seed=seed,
                samples_per_label=samples_per_label,
                model_state=model_state,
                device=target_device,
                template_bank=template_bank,
                max_seq_len=max_seq_len,
                model_training_mode=model_training_mode,
                train_per_label=train_per_label,
            )
            for pooling in ("mean", "last"):
                canonical_train_vectors, canonical_train_labels, canonical_test_vectors, canonical_test_labels = _variant_split_vectors(
                    reps_by_variant=reps_by_variant,
                    labels_by_variant=labels_by_variant,
                    samples_per_label=samples_per_label,
                    pooling=pooling,
                    variant="canonical",
                    train_per_label=train_per_label,
                    exclude_variant_train_from_test=model_training_mode == "cross_template_alignment",
                )
                train_codebook = build_logic_codebook_train_test(
                    canonical_train_vectors,
                    canonical_train_labels,
                    canonical_train_vectors,
                    canonical_train_labels,
                )
                canonical_codebook = build_logic_codebook_train_test(
                    canonical_train_vectors,
                    canonical_train_labels,
                    canonical_test_vectors,
                    canonical_test_labels,
                )
                canonical_test_accuracy = canonical_codebook.nearest_neighbor_accuracy

                for variant in LOGIC_VARIANTS:
                    train_vectors, train_labels, test_vectors, test_labels = _variant_split_vectors(
                        reps_by_variant=reps_by_variant,
                        labels_by_variant=labels_by_variant,
                        samples_per_label=samples_per_label,
                        pooling=pooling,
                        variant=variant,
                        train_per_label=train_per_label,
                        exclude_variant_train_from_test=model_training_mode == "cross_template_alignment",
                    )
                    if not np.array_equal(train_vectors, canonical_train_vectors) or not np.array_equal(train_labels, canonical_train_labels):
                        raise ValueError("non-canonical data entered centroid train split")
                    test_codebook = build_logic_codebook_train_test(train_vectors, train_labels, test_vectors, test_labels)

                    run_dir = output_path / f"seed_{seed}" / model_state / pooling / variant
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = run_dir / "generalization_metrics.json"
                    classification_report_path = run_dir / "classification_report.json"
                    confusion_matrix_path = run_dir / "confusion_matrix.csv"
                    variant_comparison_path = run_dir / "variant_comparison.csv"
                    metrics = _generalization_metrics(
                        train_codebook=train_codebook,
                        test_codebook=test_codebook,
                        seed=seed,
                        model_state=model_state,
                        pooling=pooling,
                        variant=variant,
                        train_shape=tuple(train_vectors.shape),
                        test_shape=tuple(test_vectors.shape),
                        canonical_test_accuracy=canonical_test_accuracy,
                        metadata=metadata,
                        device=target_device,
                    )
                    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                    _write_generalization_classification_report(classification_report_path, test_codebook, variant)
                    _write_confusion_matrix(confusion_matrix_path, test_codebook)

                    row = {
                        "seed": seed,
                        "model_state": model_state,
                        "pooling": pooling,
                        "variant": variant,
                        "train_accuracy": metrics["train_accuracy"],
                        "test_accuracy": metrics["test_accuracy"],
                        "macro_accuracy": metrics["macro_accuracy"],
                        "adjusted_rand_score": metrics["adjusted_rand_score"],
                        "variant_drop": metrics["variant_drop"],
                        "easiest_confusion_pair": metrics["easiest_confusion_pair"],
                    }
                    _write_variant_comparison(variant_comparison_path, [row])
                    variant_rows.append(row)
                    aggregate_reports.extend(
                        {
                            "seed": seed,
                            "model_state": model_state,
                            "pooling": pooling,
                            "variant": variant,
                            "sample_index": result.sample_index,
                            "true_label": result.true_label,
                            "predicted_label": result.predicted_label,
                            "distance": result.distance,
                            "correct": result.correct,
                        }
                        for result in test_codebook.nearest_neighbors
                    )
                    runs.append({
                        **row,
                        "metrics_path": str(metrics_path),
                        "classification_report_path": str(classification_report_path),
                        "confusion_matrix_path": str(confusion_matrix_path),
                    })

    trained_mean_runs = [run for run in runs if run["model_state"] == "trained" and run["pooling"] == "mean"]
    by_variant: dict[str, list[float]] = {
        variant: [run["test_accuracy"] for run in trained_mean_runs if run["variant"] == variant]
        for variant in LOGIC_VARIANTS
    }
    averages = {variant: float(np.mean(values)) for variant, values in by_variant.items()}
    masked_keyword_drop = averages["canonical"] - averages["masked_keywords"]
    summary = {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "template_bank": template_bank,
        "model_training_mode": model_training_mode,
        "max_seq_len": max_seq_len,
        "test_per_label_canonical": samples_per_label - train_per_label,
        "test_per_label_variants": samples_per_label - train_per_label if model_training_mode == "cross_template_alignment" else samples_per_label,
        "variants": list(LOGIC_VARIANTS),
        "num_runs": len(runs),
        "runs": runs,
        "trained_mean_average_accuracy_by_variant": averages,
        "trained_mean_canonical_average_accuracy": averages["canonical"],
        "trained_mean_synonym_average_accuracy": averages["synonym"],
        "trained_mean_perturbed_average_accuracy": averages["perturbed"],
        "trained_mean_masked_keywords_average_accuracy": averages["masked_keywords"],
        "masked_keyword_drop": masked_keyword_drop,
        "passes_canonical_gate": averages["canonical"] > 0.60,
        "passes_synonym_gate": averages["synonym"] >= 0.50,
        "passes_perturbed_gate": averages["perturbed"] >= 0.50,
        "template_leakage_indicated": masked_keyword_drop > 0.20,
        "allows_hidden_state_injection_planning": bool(averages["canonical"] > 0.60 and averages["synonym"] >= 0.50 and averages["perturbed"] >= 0.50 and masked_keyword_drop <= 0.20),
        "worst_variant": min(averages, key=averages.get),
    }

    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "generalization_metrics.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "classification_report.json").write_text(json.dumps(aggregate_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_variant_comparison(output_path / "variant_comparison.csv", variant_rows)
    _write_confusion_matrix_from_rows(output_path / "confusion_matrix.csv", aggregate_reports)
    return summary


def _write_confusion_matrix_from_rows(path: Path, rows: list[dict]) -> None:
    matrix = {label: {candidate: 0 for candidate in LOGIC_LABELS} for label in LOGIC_LABELS}
    for row in rows:
        matrix[row["true_label"]][row["predicted_label"]] += 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *LOGIC_LABELS])
        for label in LOGIC_LABELS:
            writer.writerow([label, *[matrix[label][candidate] for candidate in LOGIC_LABELS]])
