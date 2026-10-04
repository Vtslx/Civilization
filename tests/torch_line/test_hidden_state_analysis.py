from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from civilization.research.torch_line.analysis import (
    LOGIC_LABELS,
    build_logic_dataset,
    collect_hidden_state_representations,
    run_hidden_state_analysis,
)
from civilization.research.torch_line.model import MiniTransformerTorch, TransformerConfigTorch


def test_logic_dataset_has_five_labels_minimum_counts_and_is_deterministic() -> None:
    first_samples, first_tokenizer = build_logic_dataset(samples_per_label=20, max_seq_len=12, seed=42)
    second_samples, second_tokenizer = build_logic_dataset(samples_per_label=20, max_seq_len=12, seed=42)

    assert first_samples == second_samples
    assert first_tokenizer.vocab == second_tokenizer.vocab
    assert len(first_samples) == 100
    assert set(sample.label for sample in first_samples) == set(LOGIC_LABELS)

    for label in LOGIC_LABELS:
        label_samples = [sample for sample in first_samples if sample.label == label]
        assert len(label_samples) == 20
        assert all(len(sample.token_ids) == 12 for sample in label_samples)
        assert all(0 <= token_id < first_tokenizer.vocab_size for sample in label_samples for token_id in sample.token_ids)


def test_hidden_state_collection_exports_all_layers_and_pooling_shapes() -> None:
    samples, tokenizer = build_logic_dataset(samples_per_label=4, max_seq_len=12, seed=7)
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=16, hidden_dim=32, num_heads=4, num_layers=2, max_seq_len=12, seed=7)
    model = MiniTransformerTorch(config)

    representations = collect_hidden_state_representations(model, samples, device="cpu")

    assert set(representations) == {"mean", "last"}
    assert len(representations["mean"]) == config.num_layers + 1
    assert len(representations["last"]) == config.num_layers + 1
    assert representations["mean"][-1].shape == (20, config.model_dim)
    assert representations["last"][-1].shape == (20, config.model_dim)
    assert torch.isfinite(representations["mean"][-1]).all()
    assert torch.isfinite(representations["last"][-1]).all()


def test_pca_umap_kmeans_shapes_and_ari_are_recordable() -> None:
    samples, tokenizer = build_logic_dataset(samples_per_label=5, max_seq_len=12, seed=9)
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=16, hidden_dim=32, num_heads=4, num_layers=2, max_seq_len=12, seed=9)
    model = MiniTransformerTorch(config)
    representations = collect_hidden_state_representations(model, samples, device="cpu")
    vectors = representations["mean"][-1].numpy()
    labels = np.array([sample.label for sample in samples])
    y_true = LabelEncoder().fit_transform(labels)

    pca_points = PCA(n_components=2, random_state=9).fit_transform(vectors)
    umap_points = umap.UMAP(n_components=2, random_state=9, n_neighbors=5, min_dist=0.05, transform_seed=9).fit_transform(vectors)
    clusters = KMeans(n_clusters=5, random_state=9, n_init=10).fit_predict(vectors)
    ari = adjusted_rand_score(y_true, clusters)

    assert pca_points.shape == (25, 2)
    assert umap_points.shape == (25, 2)
    assert clusters.shape == (25,)
    assert -1.0 <= ari <= 1.0


def test_analysis_pipeline_generates_metrics_csv_and_plots(tmp_path: Path) -> None:
    result = run_hidden_state_analysis(output_dir=tmp_path, seed=12, samples_per_label=4, device="cpu")

    assert result.metrics_path.exists()
    assert result.csv_path.exists()
    assert result.pca_plot_path.exists()
    assert result.umap_plot_path.exists()
    assert result.metrics["num_samples"] == 20
    assert result.metrics["representations"]["trained_layers"] == 3
    assert result.metrics["representations"]["mean_shape"] == [20, 24]
    assert result.metrics["analysis"]["pca_shape"] == [20, 2]
    assert result.metrics["analysis"]["umap_shape"] == [20, 2]
    assert -1.0 <= result.metrics["analysis"]["adjusted_rand_score"] <= 1.0
    assert result.metrics["training"]["loss_decreased"]
