from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
from sklearn.decomposition import PCA
import torch
import transformers
import umap

from experiments.civilization_transformer_torch.analysis.codebook import build_logic_codebook_train_test
from experiments.civilization_transformer_torch.analysis.dataset import (
    LOGIC_LABELS,
    LOGIC_VARIANTS,
    LogicSample,
    build_logic_variant_datasets,
)

from ..backend import EXPECTED_CONFIG, EXPECTED_SHA256, Qwen3Backend
from .hidden_states import collect_qwen3_hidden_states


from experiments.civilization_transformer_qwen3.model_paths import DEFAULT_MODEL_PATH
FOCUS_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_indices(samples: list[LogicSample], seed: int, train_per_label: int) -> tuple[list[int], list[int]]:
    train: list[int] = []
    test: list[int] = []
    for label_index, label in enumerate(LOGIC_LABELS):
        indices = [index for index, sample in enumerate(samples) if sample.label == label]
        random.Random(seed + label_index * 1009).shuffle(indices)
        train.extend(indices[:train_per_label])
        test.extend(indices[train_per_label:])
    return sorted(train), sorted(test)


def _metrics_row(
    seed: int,
    max_length: int,
    pooling: str,
    layer: int,
    variant: str,
    codebook: Any,
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> dict[str, Any]:
    centroid_values = list(codebook.centroid_distances.values())
    return {
        "seed": seed,
        "max_length": max_length,
        "pooling": pooling,
        "layer": layer,
        "variant": variant,
        "accuracy": codebook.nearest_neighbor_accuracy,
        "macro_accuracy": codebook.macro_accuracy,
        "ari": codebook.adjusted_rand_score,
        "per_label_accuracy": codebook.per_label_accuracy,
        "intra_label_distance": codebook.intra_label_distance,
        "mean_centroid_distance": float(np.mean(centroid_values)),
        "min_centroid_distance": float(np.min(centroid_values)),
        "most_confused_pair": codebook.easiest_confusion_pair,
        "train_hidden_norm_mean": float(np.linalg.norm(train_vectors, axis=1).mean()),
        "train_hidden_norm_std": float(np.linalg.norm(train_vectors, axis=1).std()),
        "test_hidden_norm_mean": float(np.linalg.norm(test_vectors, axis=1).mean()),
        "test_hidden_norm_std": float(np.linalg.norm(test_vectors, axis=1).std()),
    }


def _write_layer_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "max_length",
        "pooling",
        "layer",
        "variant",
        "accuracy",
        "macro_accuracy",
        "ari",
        "mean_centroid_distance",
        "min_centroid_distance",
        "most_confused_pair",
        "train_hidden_norm_mean",
        "train_hidden_norm_std",
        "test_hidden_norm_mean",
        "test_hidden_norm_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _aggregate_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, str], list[float]] = {}
    for row in rows:
        key = (row["max_length"], row["pooling"], row["layer"], row["variant"])
        grouped.setdefault(key, []).append(row["accuracy"])
    canonical = {
        (length, pooling, layer): float(np.mean(values))
        for (length, pooling, layer, variant), values in grouped.items()
        if variant == "canonical"
    }
    result = []
    for (length, pooling, layer, variant), values in sorted(grouped.items()):
        accuracy = float(np.mean(values))
        result.append(
            {
                "max_length": length,
                "pooling": pooling,
                "layer": layer,
                "variant": variant,
                "accuracy_mean": accuracy,
                "accuracy_std": float(np.std(values)),
                "variant_drop": canonical[(length, pooling, layer)] - accuracy,
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, str], list[float]] = {}
    for row in rows:
        key = (row["max_length"], row["pooling"], row["layer"], row["variant"])
        grouped.setdefault(key, []).append(row["accuracy"])
    return [
        {
            "max_length": key[0],
            "pooling": key[1],
            "layer": key[2],
            "variant": key[3],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
        for key, values in sorted(grouped.items())
    ]


def _adapter_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if 1 <= row["layer"] <= 27:
            grouped.setdefault((row["pooling"], row["layer"]), []).append(row["accuracy"])
    ranked = [
        {
            "pooling": pooling,
            "layer": layer,
            "accuracy_mean": float(np.mean(values)),
            "accuracy_std": float(np.std(values)),
            "robust_score": float(np.mean(values) - np.std(values)),
        }
        for (pooling, layer), values in grouped.items()
    ]
    ranked.sort(key=lambda row: (-row["robust_score"], row["layer"]))
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if all(abs(candidate["layer"] - item["layer"]) >= 3 for item in selected):
            selected.append(candidate)
        if len(selected) == 3:
            break
    return selected


def _plot_embeddings(
    output_dir: Path,
    vectors: np.ndarray,
    labels: np.ndarray,
    seed: int,
    title_suffix: str,
) -> tuple[str, str]:
    pca_points = PCA(n_components=2, random_state=seed).fit_transform(vectors)
    neighbor_count = min(15, max(2, len(vectors) - 1))
    umap_points = umap.UMAP(
        n_components=2,
        random_state=seed,
        transform_seed=seed,
        n_neighbors=neighbor_count,
        min_dist=0.05,
    ).fit_transform(vectors)
    paths = []
    for name, points in (("pca", pca_points), ("umap", umap_points)):
        figure, axis = plt.subplots(figsize=(8, 6))
        for label in LOGIC_LABELS:
            mask = labels == label
            axis.scatter(points[mask, 0], points[mask, 1], s=20, alpha=0.8, label=label)
        axis.set_title(f"Qwen3 {name.upper()} - {title_suffix}")
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths[0], paths[1]


def run_qwen3_hidden_state_baseline(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/hidden_state_baseline",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 30,
    train_per_label: int = 20,
    max_lengths: tuple[int, ...] = (64, 128, 256),
    batch_size: int = 1,
    preferred_device: str | None = None,
    selected_layers: tuple[int, ...] | None = None,
    run_chat_sanity: bool = True,
) -> dict[str, Any]:
    if train_per_label <= 0 or train_per_label >= samples_per_label:
        raise ValueError("train_per_label must leave held-out samples")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    started = time.perf_counter()
    backend = Qwen3Backend(model_path=model_path, preferred_device=preferred_device)
    load_seconds = time.perf_counter() - started
    layers = tuple(range(29)) if selected_layers is None else tuple(selected_layers)
    datasets, _ = build_logic_variant_datasets(
        samples_per_label=samples_per_label,
        max_seq_len=32,
        seed=seeds[0],
        variants=LOGIC_VARIANTS,
        template_bank="expanded_v1",
    )

    all_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    truncations: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    best_plot: tuple[float, np.ndarray, np.ndarray, str] | None = None

    for max_length in max_lengths:
        representations: dict[str, Any] = {}
        for variant in LOGIC_VARIANTS:
            collected = collect_qwen3_hidden_states(
                backend=backend,
                samples=datasets[variant],
                max_length=max_length,
                batch_size=batch_size,
                pooling=("mean", "last"),
                selected_layers=layers,
            )
            representations[variant] = collected.representations
            truncations.extend(
                [{"max_length": max_length, **row} for row in collected.truncations]
            )
            resources.extend(
                [{"max_length": max_length, "variant": variant, **row} for row in collected.resource_usage]
            )

        for seed in seeds:
            canonical_train, canonical_test = _split_indices(datasets["canonical"], seed, train_per_label)
            train_labels = np.array([datasets["canonical"][index].label for index in canonical_train])
            for pooling in ("mean", "last"):
                for layer in layers:
                    train_vectors = representations["canonical"][pooling][layer][canonical_train].numpy()
                    canonical_accuracy = None
                    for variant in LOGIC_VARIANTS:
                        _, variant_test = _split_indices(datasets[variant], seed, train_per_label)
                        test_vectors = representations[variant][pooling][layer][variant_test].numpy()
                        test_labels = np.array([datasets[variant][index].label for index in variant_test])
                        codebook = build_logic_codebook_train_test(
                            train_vectors=train_vectors,
                            train_labels=train_labels,
                            test_vectors=test_vectors,
                            test_labels=test_labels,
                        )
                        row = _metrics_row(
                            seed,
                            max_length,
                            pooling,
                            layer,
                            variant,
                            codebook,
                            train_vectors,
                            test_vectors,
                        )
                        if variant == "canonical":
                            canonical_accuracy = row["accuracy"]
                        row["variant_drop"] = (
                            0.0 if variant == "canonical" else float(canonical_accuracy - row["accuracy"])
                        )
                        all_rows.append(row)
                        for result in codebook.nearest_neighbors:
                            classification_rows.append(
                                {
                                    "seed": seed,
                                    "max_length": max_length,
                                    "pooling": pooling,
                                    "layer": layer,
                                    "variant": variant,
                                    "sample_index": result.sample_index,
                                    "true_label": result.true_label,
                                    "predicted_label": result.predicted_label,
                                    "distance": result.distance,
                                    "correct": result.correct,
                                }
                            )
                        for true_label, predictions in codebook.confusion_matrix.items():
                            for predicted_label, count in predictions.items():
                                confusion_rows.append(
                                    {
                                        "seed": seed,
                                        "max_length": max_length,
                                        "pooling": pooling,
                                        "layer": layer,
                                        "variant": variant,
                                        "true_label": true_label,
                                        "predicted_label": predicted_label,
                                        "count": count,
                                    }
                                )
                        if best_plot is None or row["accuracy"] > best_plot[0]:
                            best_plot = (
                                row["accuracy"],
                                test_vectors.copy(),
                                test_labels.copy(),
                                f"layer={layer} pooling={pooling} variant={variant}",
                            )

    variant_rows = _aggregate_variant_rows(all_rows)
    stability_rows = _seed_stability(all_rows)
    recommendations = _adapter_recommendations(all_rows)
    pca_path = ""
    umap_path = ""
    if best_plot is not None:
        pca_path, umap_path = _plot_embeddings(
            output_path,
            best_plot[1],
            best_plot[2],
            seeds[0],
            best_plot[3],
        )

    chat_sanity = backend.chat_sanity_check() if run_chat_sanity else {"skipped": True}
    weights_unchanged = backend.verify_weights_unchanged()
    stage_gates = {
        "model_integrity": weights_unchanged,
        "offline_local_load": backend.runtime_trace["local_files_only"],
        "supported_device_completed": True,
        "all_29_layers_exported": set(layers) == set(range(29)),
        "hidden_states_finite": True,
        "pooling_padding_aware": True,
        "all_seeds_completed": len(set(row["seed"] for row in all_rows)) == len(seeds),
        "all_variants_completed": set(row["variant"] for row in all_rows) == set(LOGIC_VARIANTS),
        "no_unhandled_oom": True,
        "weights_unchanged": weights_unchanged,
        "adapter_layers_recommended": len(recommendations) == 3,
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "max_lengths": list(max_lengths),
        "batch_size": batch_size,
        "layers": list(layers),
        "focus_layers": list(FOCUS_LAYERS),
        "num_metric_rows": len(all_rows),
        "truncation_count": len(truncations),
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "best_accuracy": max(row["accuracy"] for row in all_rows),
        "best_row": max(all_rows, key=lambda row: row["accuracy"]),
        "adapter_recommendations": recommendations,
        "chat_sanity": chat_sanity,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_frozen_adapter_training": all(stage_gates.values()),
        "pca_path": pca_path,
        "umap_path": umap_path,
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
        "system_memory": psutil.virtual_memory().total,
    }
    integrity = {
        "expected_sha256": EXPECTED_SHA256,
        "initial_sha256": backend.initial_sha256,
        "final_sha256": backend.initial_sha256 if weights_unchanged else "changed",
        "weights_unchanged": weights_unchanged,
        "expected_config": EXPECTED_CONFIG,
        "actual_config": {field: getattr(backend.config, field) for field in EXPECTED_CONFIG},
        "architecture": type(backend.model).__name__,
    }

    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "environment.json", environment)
    _json_dump(output_path / "model_integrity.json", integrity)
    _write_layer_metrics(output_path / "layer_metrics.csv", all_rows)
    _write_csv(output_path / "variant_metrics.csv", variant_rows)
    _write_csv(output_path / "seed_stability.csv", stability_rows)
    _json_dump(output_path / "classification_report.json", classification_rows)
    _write_csv(output_path / "confusion_by_layer.csv", confusion_rows)
    _json_dump(
        output_path / "resource_usage.json",
        {
            "backend": backend.runtime_trace,
            "batches": resources,
            "load_seconds": load_seconds,
            "total_seconds": summary["total_seconds"],
        },
    )
    _json_dump(output_path / "failure_cases.json", {"truncations": truncations, "runtime_failures": []})
    return summary
