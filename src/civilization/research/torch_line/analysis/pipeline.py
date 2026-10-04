from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import torch
import umap

from ..device import resolve_device
from ..model import MiniTransformerTorch, TransformerConfigTorch
from ..training import build_next_token_batch
from ..model.transformer_torch import next_token_loss
from .dataset import LOGIC_LABELS, build_logic_dataset
from .hidden_states import collect_hidden_state_representations


@dataclass(frozen=True)
class AnalysisResult:
    metrics_path: Path
    csv_path: Path
    pca_plot_path: Path
    umap_plot_path: Path
    metrics: dict


def _train_model_for_analysis(model: MiniTransformerTorch, device: torch.device, seed: int) -> list[float]:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    input_ids, targets = build_next_token_batch(batch_size=8, seq_len=8, vocab_size=model.config.vocab_size, device=device)
    losses: list[float] = []
    for step in range(26):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids)
        loss = next_token_loss(output.logits, targets)
        losses.append(float(loss.detach().cpu().item()))
        if step < 25:
            loss.backward()
            optimizer.step()
    return losses


def _center_distances(points: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    centers = {}
    for label in LOGIC_LABELS:
        centers[label] = points[labels == label].mean(axis=0)
    distances: dict[str, float] = {}
    for left_index, left in enumerate(LOGIC_LABELS):
        for right in LOGIC_LABELS[left_index + 1 :]:
            distances[f"{left}__{right}"] = float(np.linalg.norm(centers[left] - centers[right]))
    return distances


def _plot(points: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label in LOGIC_LABELS:
        label_points = points[labels == label]
        plt.scatter(label_points[:, 0], label_points[:, 1], label=label, s=24)
    plt.title(title)
    plt.xlabel("component_1")
    plt.ylabel("component_2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_csv(path: Path, labels: np.ndarray, pca_points: np.ndarray, umap_points: np.ndarray, clusters: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "pca_x", "pca_y", "umap_x", "umap_y", "cluster"])
        for index, label in enumerate(labels):
            writer.writerow([index, label, pca_points[index, 0], pca_points[index, 1], umap_points[index, 0], umap_points[index, 1], int(clusters[index])])


def run_hidden_state_analysis(
    output_dir: str | Path = "artifacts/torch-line/hidden_state_analysis",
    seed: int = 202,
    samples_per_label: int = 20,
    device: str | torch.device | None = "cpu",
) -> AnalysisResult:
    target_device = resolve_device(device)
    samples, tokenizer = build_logic_dataset(samples_per_label=samples_per_label, max_seq_len=12, seed=seed)
    labels = np.array([sample.label for sample in samples])
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=12, seed=seed)

    torch.manual_seed(seed)
    untrained = MiniTransformerTorch(config).to(target_device)
    untrained_reps = collect_hidden_state_representations(untrained, samples, target_device)

    torch.manual_seed(seed)
    trained = MiniTransformerTorch(config).to(target_device)
    losses = _train_model_for_analysis(trained, target_device, seed)
    trained_reps = collect_hidden_state_representations(trained, samples, target_device)

    vectors = trained_reps["mean"][-1].numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("pooled vectors contain NaN or Inf")

    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(labels)
    pca_points = PCA(n_components=2, random_state=seed).fit_transform(vectors)
    umap_points = umap.UMAP(n_components=2, random_state=seed, n_neighbors=10, min_dist=0.05, transform_seed=seed).fit_transform(vectors)
    clusters = KMeans(n_clusters=len(LOGIC_LABELS), random_state=seed, n_init=10).fit_predict(vectors)
    ari = float(adjusted_rand_score(y_true, clusters))

    output_path = Path(output_dir)
    metrics_path = output_path / "metrics.json"
    csv_path = output_path / "points.csv"
    pca_plot_path = output_path / "pca.png"
    umap_plot_path = output_path / "umap.png"

    metrics = {
        "seed": seed,
        "device": str(target_device),
        "samples_per_label": samples_per_label,
        "num_samples": len(samples),
        "labels": list(LOGIC_LABELS),
        "vocab_size": tokenizer.vocab_size,
        "model_config": {
            "vocab_size": config.vocab_size,
            "model_dim": config.model_dim,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "max_seq_len": config.max_seq_len,
        },
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
        },
        "representations": {
            "untrained_layers": len(untrained_reps["mean"]),
            "trained_layers": len(trained_reps["mean"]),
            "mean_shape": list(trained_reps["mean"][-1].shape),
            "last_shape": list(trained_reps["last"][-1].shape),
        },
        "analysis": {
            "pca_shape": list(pca_points.shape),
            "umap_shape": list(umap_points.shape),
            "kmeans_clusters": [int(value) for value in clusters.tolist()],
            "adjusted_rand_score": ari,
            "center_distances": _center_distances(vectors, labels),
        },
    }

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, labels, pca_points, umap_points, clusters)
    _plot(pca_points, labels, "Logic hidden states PCA", pca_plot_path)
    _plot(umap_points, labels, "Logic hidden states UMAP", umap_plot_path)

    return AnalysisResult(metrics_path=metrics_path, csv_path=csv_path, pca_plot_path=pca_plot_path, umap_plot_path=umap_plot_path, metrics=metrics)
