import math
from pathlib import Path

import torch

from experiments.civilization_transformer_torch.analysis import (
    LOGIC_LABELS,
    LOGIC_VARIANTS,
    build_logic_variant_datasets,
    leakage_tokens_for_label,
    run_alignment_training,
    select_alignment_train_samples,
)
from experiments.civilization_transformer_torch.analysis.hidden_states import collect_hidden_state_representations
from experiments.civilization_transformer_torch.analysis.run_alignment_until_stage11_pass import run_alignment_until_stage11_pass
from experiments.civilization_transformer_torch.model import MiniTransformerTorch, TransformerConfigTorch


def test_expanded_template_bank_has_required_volume_metadata_and_masking() -> None:
    datasets, tokenizer = build_logic_variant_datasets(
        samples_per_label=60,
        max_seq_len=18,
        seed=202,
        template_bank="expanded_v1",
    )

    assert set(datasets) == set(LOGIC_VARIANTS)
    for variant, samples in datasets.items():
        assert len(samples) == len(LOGIC_LABELS) * 60
        for label in LOGIC_LABELS:
            label_samples = [sample for sample in samples if sample.label == label]
            assert len(label_samples) == 60
            assert len({sample.template_id for sample in label_samples}) >= 4
            for sample in label_samples:
                assert sample.variant == variant
                assert len(sample.token_ids) == 18
                assert max(sample.token_ids) < tokenizer.vocab_size
                if variant == "masked_keywords":
                    assert set(tokenizer.tokenize(sample.text)).isdisjoint(leakage_tokens_for_label(label))


def test_alignment_training_uses_train_split_and_decreases_losses() -> None:
    seed = 202
    datasets, tokenizer = build_logic_variant_datasets(
        samples_per_label=12,
        max_seq_len=18,
        seed=seed,
        template_bank="expanded_v1",
    )
    train_samples = select_alignment_train_samples(datasets, train_per_label=8)
    assert len(train_samples) == len(LOGIC_LABELS) * len(LOGIC_VARIANTS) * 8

    torch.manual_seed(seed)
    config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=16, hidden_dim=32, num_heads=4, num_layers=2, max_seq_len=18, seed=seed)
    model = MiniTransformerTorch(config)
    result = run_alignment_training(
        model=model,
        datasets=datasets,
        device="cpu",
        seed=seed,
        train_per_label=8,
        steps=25,
        learning_rate=0.02,
    )

    assert result.total_loss_decreased
    assert result.classification_loss_decreased
    assert result.train_samples == len(train_samples)
    assert all(math.isfinite(row["total_loss"]) for row in result.losses)
    reps = collect_hidden_state_representations(model, datasets["canonical"][:10], "cpu")
    assert torch.isfinite(reps["mean"][-1]).all()
    assert torch.isfinite(reps["last"][-1]).all()


def test_alignment_until_stage11_passes_for_single_seed(tmp_path: Path) -> None:
    result = run_alignment_until_stage11_pass(
        output_dir=tmp_path,
        seeds=(202,),
        initial_samples_per_label=60,
        initial_train_per_label=40,
        max_iterations=1,
        device="cpu",
    )

    summary = result["final_summary"]
    assert result["passed"]
    assert len(result["iterations"]) == 1
    assert summary["model_training_mode"] == "cross_template_alignment"
    assert summary["template_bank"] == "expanded_v1"
    assert summary["allows_hidden_state_injection_planning"]
    assert summary["trained_mean_canonical_average_accuracy"] > 0.60
    assert summary["trained_mean_synonym_average_accuracy"] >= 0.50
    assert summary["trained_mean_perturbed_average_accuracy"] >= 0.50
    assert summary["masked_keyword_drop"] <= 0.20
    assert (tmp_path / "iteration_history.json").exists()
