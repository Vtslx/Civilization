from pathlib import Path

from experiments.civilization_transformer_torch.analysis.dataset import (
    LOGIC_LABELS,
    PATH_STRESS_PROFILES,
    build_path_dependency_datasets,
)
from experiments.civilization_transformer_torch.analysis.run_path_dependency_stress_test import run_path_dependency_stress_test


def test_path_dependency_stress_profiles_generate_deterministic_contexts() -> None:
    for profile in PATH_STRESS_PROFILES:
        left, left_tokenizer = build_path_dependency_datasets(samples_per_label=8, max_seq_len=32, seed=202, stress_profile=profile)
        right, right_tokenizer = build_path_dependency_datasets(samples_per_label=8, max_seq_len=32, seed=202, stress_profile=profile)

        assert left == right
        assert left_tokenizer.vocab == right_tokenizer.vocab
        for samples in left.values():
            assert {sample.label for sample in samples} == set(LOGIC_LABELS)
            assert all(sample.stress_profile == profile for sample in samples)
            assert all(len(sample.token_ids) == 32 for sample in samples)
            assert all(max(sample.token_ids) < left_tokenizer.vocab_size for sample in samples)


def test_obfuscated_noisy_and_conflicting_profiles_mark_context_pressure() -> None:
    obfuscated, _ = build_path_dependency_datasets(samples_per_label=6, max_seq_len=32, stress_profile="obfuscated_v1")
    noisy, _ = build_path_dependency_datasets(samples_per_label=6, max_seq_len=32, stress_profile="noisy_context_v1")
    conflicting, _ = build_path_dependency_datasets(samples_per_label=6, max_seq_len=32, stress_profile="conflicting_context_v1")

    for samples in obfuscated.values():
        for sample in samples:
            targets = f"{sample.memory_target} {sample.rule_target} {sample.state_target}"
            assert all(label not in targets for label in LOGIC_LABELS)
    assert all(sample.context_noise_count > 0 for samples in noisy.values() for sample in samples)
    assert all(sample.conflict_context_count > 0 for samples in conflicting.values() for sample in samples)


def test_path_dependency_stress_smoke_outputs_profile_metrics(tmp_path: Path) -> None:
    summary = run_path_dependency_stress_test(
        output_dir=tmp_path,
        modes=("full", "no_memory_path", "no_state_path", "no_rule_path", "structure_only"),
        seeds=(202,),
        samples_per_label=20,
        train_per_label=12,
        seq_lens=(32,),
        model_sizes=("medium",),
        stress_profiles=("direct_v1", "obfuscated_v1"),
        training_steps=12,
        device="cpu",
    )

    assert set(summary["stress_profiles"]) == {"direct_v1", "obfuscated_v1"}
    assert "direct_v1" in summary["profile_average_accuracy"]
    assert "obfuscated_v1" in summary["profile_average_accuracy"]
    assert "path_large_not_below_large_toy" in summary["stage_gates"]
    assert isinstance(summary["passes_stage_gate"], bool)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "path_metrics.csv").exists()
    assert (tmp_path / "ablation_drop.csv").exists()
    assert (tmp_path / "context_profile_metrics.csv").exists()
    assert (tmp_path / "seed_stability.csv").exists()
    assert (tmp_path / "size_comparison.csv").exists()
    assert (tmp_path / "seq_len_comparison.csv").exists()
    assert (tmp_path / "failure_cases.json").exists()
    assert (tmp_path / "trace_contribution.csv").exists()
