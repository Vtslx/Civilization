import math
from pathlib import Path

import numpy as np

from experiments.civilization_transformer_torch.analysis import (
    LOGIC_LABELS,
    LOGIC_VARIANTS,
    build_logic_variant_datasets,
    leakage_tokens_for_label,
)
from experiments.civilization_transformer_torch.analysis.codebook import (
    build_logic_codebook_train_test,
    run_codebook_generalization_matrix,
)


def test_variant_datasets_have_required_counts_are_deterministic_and_mask_keywords() -> None:
    first, tokenizer = build_logic_variant_datasets(samples_per_label=30, max_seq_len=14, seed=202)
    second, second_tokenizer = build_logic_variant_datasets(samples_per_label=30, max_seq_len=14, seed=202)

    assert first == second
    assert tokenizer.vocab == second_tokenizer.vocab
    assert set(first) == set(LOGIC_VARIANTS)
    for variant, samples in first.items():
        assert len(samples) == len(LOGIC_LABELS) * 30
        for label in LOGIC_LABELS:
            label_samples = [sample for sample in samples if sample.label == label]
            assert len(label_samples) == 30
            for sample in label_samples:
                assert sample.variant == variant
                assert len(sample.token_ids) == 14
                assert max(sample.token_ids) < tokenizer.vocab_size
                assert min(sample.token_ids) >= 0
                if variant == "masked_keywords":
                    tokens = set(tokenizer.tokenize(sample.text))
                    assert tokens.isdisjoint(leakage_tokens_for_label(label))


def test_train_test_codebook_uses_train_counts_and_classifies_test_only() -> None:
    train_vectors = []
    train_labels = []
    test_vectors = []
    test_labels = []
    for label_index, label in enumerate(LOGIC_LABELS):
        center = np.zeros(len(LOGIC_LABELS), dtype=float)
        center[label_index] = 1.0
        for _ in range(20):
            train_vectors.append(center)
            train_labels.append(label)
        for _ in range(10):
            test_vectors.append(center)
            test_labels.append(label)

    codebook = build_logic_codebook_train_test(
        train_vectors=np.stack(train_vectors),
        train_labels=np.array(train_labels),
        test_vectors=np.stack(test_vectors),
        test_labels=np.array(test_labels),
    )

    assert all(codebook.sample_counts[label] == 20 for label in LOGIC_LABELS)
    assert len(codebook.nearest_neighbors) == len(LOGIC_LABELS) * 10
    assert codebook.nearest_neighbor_accuracy == 1.0
    assert codebook.macro_accuracy == 1.0
    assert all(math.isfinite(value) for value in codebook.intra_label_distance.values())
    assert all(math.isfinite(value) for value in codebook.centroid_distances.values())


def test_codebook_generalization_matrix_outputs_all_variants_and_artifacts(tmp_path: Path) -> None:
    summary = run_codebook_generalization_matrix(
        output_dir=tmp_path,
        seeds=(202, 303, 404),
        samples_per_label=30,
        train_per_label=20,
        device="cpu",
    )

    assert summary["seeds"] == [202, 303, 404]
    assert summary["samples_per_label"] == 30
    assert summary["train_per_label"] == 20
    assert summary["num_runs"] == 48
    assert set(summary["trained_mean_average_accuracy_by_variant"]) == set(LOGIC_VARIANTS)
    assert summary["trained_mean_canonical_average_accuracy"] > 0.60
    assert isinstance(summary["template_leakage_indicated"], bool)
    assert isinstance(summary["allows_hidden_state_injection_planning"], bool)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "generalization_metrics.json").exists()
    assert (tmp_path / "classification_report.json").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "variant_comparison.csv").exists()

    required = {(seed, state, pooling, variant) for seed in (202, 303, 404) for state in ("untrained", "trained") for pooling in ("mean", "last") for variant in LOGIC_VARIANTS}
    actual = {(run["seed"], run["model_state"], run["pooling"], run["variant"]) for run in summary["runs"]}
    assert actual == required
    for run in summary["runs"]:
        assert Path(run["metrics_path"]).exists()
        assert Path(run["classification_report_path"]).exists()
        assert Path(run["confusion_matrix_path"]).exists()
        assert 0.0 <= run["test_accuracy"] <= 1.0
        assert 0.0 <= run["macro_accuracy"] <= 1.0
        assert -1.0 <= run["adjusted_rand_score"] <= 1.0
        assert math.isfinite(run["variant_drop"])
        assert run["easiest_confusion_pair"]


def test_codebook_generalization_is_stable_for_same_seed(tmp_path: Path) -> None:
    first = run_codebook_generalization_matrix(
        output_dir=tmp_path / "first",
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        device="cpu",
    )
    second = run_codebook_generalization_matrix(
        output_dir=tmp_path / "second",
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        device="cpu",
    )

    first_metrics = [(run["model_state"], run["pooling"], run["variant"], run["test_accuracy"], run["macro_accuracy"], run["adjusted_rand_score"]) for run in first["runs"]]
    second_metrics = [(run["model_state"], run["pooling"], run["variant"], run["test_accuracy"], run["macro_accuracy"], run["adjusted_rand_score"]) for run in second["runs"]]
    assert first_metrics == second_metrics
