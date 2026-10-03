from pathlib import Path

import math

from experiments.civilization_transformer_torch.analysis import HARD_LOGIC_SCENARIOS, LOGIC_LABELS, build_hard_logic_datasets
from experiments.civilization_transformer_torch.analysis.run_hard_logic_benchmark import run_hard_logic_benchmark


def test_hard_logic_datasets_are_deterministic_complete_and_leakage_checked() -> None:
    first, tokenizer = build_hard_logic_datasets(samples_per_label=40, max_seq_len=32, seed=202)
    second, second_tokenizer = build_hard_logic_datasets(samples_per_label=40, max_seq_len=32, seed=202)

    assert first == second
    assert tokenizer.vocab == second_tokenizer.vocab
    assert set(first) == set(HARD_LOGIC_SCENARIOS)
    for scenario, samples in first.items():
        assert len(samples) == len(LOGIC_LABELS) * 40
        for label in LOGIC_LABELS:
            label_samples = [sample for sample in samples if sample.label == label]
            assert len(label_samples) == 40
            for sample in label_samples:
                assert sample.variant == scenario
                assert len(sample.token_ids) == 32
                assert max(sample.token_ids) < tokenizer.vocab_size
                assert min(sample.token_ids) >= 0
                assert sample.difficulty_level >= 2
                assert sample.logic_depth >= 1
                assert sample.leakage_family
                if scenario == "masked_keywords_hard":
                    tokens = set(tokenizer.tokenize(sample.text))
                    assert tokens.isdisjoint({"because", "therefore", "not", "never", "always", "critical", "priority", "if", "then"})

    adversarial = first["adversarial_keywords"]
    assert all({"because", "not", "always", "critical", "if", "then"}.issubset(set(tokenizer.tokenize(sample.text))) for sample in adversarial)
    counterfactual = first["counterfactual_pair"]
    assert all(sample.pair_id.startswith("counterfactual_pair_") for sample in counterfactual)


def test_hard_logic_benchmark_smoke_outputs_metrics_and_artifacts(tmp_path: Path) -> None:
    summary = run_hard_logic_benchmark(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=40,
        train_per_label=25,
        seq_lens=(32,),
        model_sizes=("medium",),
        training_steps=25,
        device="cpu",
        run_stage16_regression=False,
    )

    assert summary["num_training_runs"] == 1
    assert summary["num_scenario_rows"] == len(HARD_LOGIC_SCENARIOS)
    assert set(summary["average_accuracy_by_scenario"]) == set(HARD_LOGIC_SCENARIOS)
    assert set(summary["average_accuracy_by_label"]) == set(LOGIC_LABELS)
    assert summary["losses_decreased"]
    assert summary["failures"]["nan_inf"] == 0
    assert summary["failures"]["trace_missing"] == 0
    assert summary["failures"]["masked_keyword_violations"] == 0
    assert isinstance(summary["allows_large_model_migration_planning"], bool)
    for value in summary["average_accuracy_by_scenario"].values():
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "runs.json").exists()
    assert (tmp_path / "scenario_metrics.csv").exists()
    assert (tmp_path / "failure_cases.json").exists()
    assert (tmp_path / "confusion_by_scenario.csv").exists()
    assert (tmp_path / "trace_failures.json").exists()


def test_hard_logic_benchmark_medium_two_seed_matrix_runs(tmp_path: Path) -> None:
    summary = run_hard_logic_benchmark(
        output_dir=tmp_path,
        seeds=(202, 303),
        samples_per_label=120,
        train_per_label=80,
        seq_lens=(48,),
        model_sizes=("medium",),
        training_steps=30,
        device="cpu",
        run_stage16_regression=False,
    )

    assert summary["seeds"] == [202, 303]
    assert summary["samples_per_label"] == 120
    assert summary["train_per_label"] == 80
    assert summary["test_per_label"] == 40
    assert summary["num_training_runs"] == 2
    assert summary["num_scenario_rows"] == 2 * len(HARD_LOGIC_SCENARIOS)
    assert summary["losses_decreased"]
    assert summary["failures"]["nan_inf"] == 0
    assert summary["failures"]["trace_missing"] == 0
    assert summary["failures"]["counterfactual_split_unsafe"] == 0
