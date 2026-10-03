import math
from pathlib import Path

import numpy as np

from experiments.civilization_transformer_torch.analysis import LOGIC_LABELS, build_logic_dataset
from experiments.civilization_transformer_torch.analysis.codebook import (
    build_logic_codebook,
    run_codebook_matrix,
    run_single_codebook_analysis,
)


def _separable_vectors(samples_per_label: int = 40) -> tuple[np.ndarray, np.ndarray]:
    samples, _ = build_logic_dataset(samples_per_label=samples_per_label, max_seq_len=12, seed=1)
    vectors = []
    labels = []
    for sample in samples:
        base = np.zeros(len(LOGIC_LABELS), dtype=float)
        base[LOGIC_LABELS.index(sample.label)] = 1.0
        vectors.append(base)
        labels.append(sample.label)
    return np.stack(vectors), np.array(labels)


def test_logic_codebook_generates_centroids_distances_and_nn_results() -> None:
    vectors, labels = _separable_vectors(samples_per_label=40)
    codebook = build_logic_codebook(vectors, labels)

    assert set(codebook.centroids) == set(LOGIC_LABELS)
    assert set(codebook.sample_counts) == set(LOGIC_LABELS)
    for label in LOGIC_LABELS:
        assert codebook.centroids[label].shape == (len(LOGIC_LABELS),)
        assert codebook.sample_counts[label] == 40
        assert math.isfinite(codebook.intra_label_distance[label])
        assert codebook.per_label_accuracy[label] == 1.0
    assert all(math.isfinite(value) for value in codebook.centroid_distances.values())
    assert len(codebook.nearest_neighbors) == 200
    assert codebook.nearest_neighbor_accuracy == 1.0
    assert codebook.macro_accuracy == 1.0


def test_single_codebook_analysis_writes_required_artifacts(tmp_path: Path) -> None:
    result = run_single_codebook_analysis(
        output_dir=tmp_path,
        seed=202,
        samples_per_label=40,
        model_state="trained",
        pooling="mean",
        device="cpu",
    )

    assert result.metrics_path.exists()
    assert result.classification_report_path.exists()
    assert result.centroid_distances_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.metrics["samples_per_label"] == 40
    assert result.metrics["num_samples"] == 200
    assert result.metrics["representation_shape"] == [200, 24]
    assert result.metrics["nearest_neighbor_accuracy"] > result.metrics["random_baseline"]
    assert result.metrics["passes_random_baseline"]
    assert set(result.metrics["per_label_accuracy"]) == set(LOGIC_LABELS)


def test_codebook_analysis_is_stable_for_same_seed(tmp_path: Path) -> None:
    first = run_single_codebook_analysis(
        output_dir=tmp_path / "first",
        seed=303,
        samples_per_label=40,
        model_state="trained",
        pooling="mean",
        device="cpu",
    )
    second = run_single_codebook_analysis(
        output_dir=tmp_path / "second",
        seed=303,
        samples_per_label=40,
        model_state="trained",
        pooling="mean",
        device="cpu",
    )

    assert first.metrics["nearest_neighbor_accuracy"] == second.metrics["nearest_neighbor_accuracy"]
    assert first.metrics["macro_accuracy"] == second.metrics["macro_accuracy"]
    assert first.metrics["per_label_accuracy"] == second.metrics["per_label_accuracy"]
    assert first.metrics["adjusted_rand_score"] == second.metrics["adjusted_rand_score"]


def test_codebook_matrix_runs_three_seeds_trained_untrained_and_pooling_modes(tmp_path: Path) -> None:
    summary = run_codebook_matrix(output_dir=tmp_path, seeds=(202, 303, 404), samples_per_label=40, device="cpu")

    assert summary["seeds"] == [202, 303, 404]
    assert summary["samples_per_label"] == 40
    assert summary["num_runs"] == 12
    assert summary["trained_mean_min_accuracy"] > 0.20
    assert summary["trained_mean_average_accuracy"] > 0.60
    assert summary["passes_stage_gate"]
    trained_mean_runs = [run for run in summary["runs"] if run["model_state"] == "trained" and run["pooling"] == "mean"]
    assert len(trained_mean_runs) == 3
    for run in summary["runs"]:
        assert Path(run["metrics_path"]).exists()
        assert -1.0 <= run["adjusted_rand_score"] <= 1.0
        assert 0.0 <= run["nearest_neighbor_accuracy"] <= 1.0
        assert run["easiest_confusion_pair"]
