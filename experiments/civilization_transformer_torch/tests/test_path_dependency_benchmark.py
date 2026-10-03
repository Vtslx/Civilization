from pathlib import Path

from experiments.civilization_transformer_torch.analysis.dataset import (
    LOGIC_LABELS,
    PATH_DEPENDENCY_SCENARIOS,
    build_path_dependency_datasets,
)
from experiments.civilization_transformer_torch.analysis.run_path_dependency_benchmark import run_path_dependency_benchmark


def test_path_dependency_dataset_requires_context_and_surface_flips() -> None:
    datasets, tokenizer = build_path_dependency_datasets(samples_per_label=12, max_seq_len=32, seed=202)

    assert set(datasets) == set(PATH_DEPENDENCY_SCENARIOS)
    assert tokenizer.vocab_size > 2
    for scenario, samples in datasets.items():
        assert {sample.label for sample in samples} == set(LOGIC_LABELS)
        assert all(sample.required_paths for sample in samples)
        assert all(max(sample.token_ids) < tokenizer.vocab_size for sample in samples)
        assert all(len(sample.token_ids) == 32 for sample in samples)
        groups: dict[str, set[str]] = {}
        for sample in samples:
            groups.setdefault(sample.surface_group_id, set()).add(sample.label)
        assert any(len(labels) > 1 for labels in groups.values())
        if scenario == "memory_required_two_hop":
            assert all(sample.label not in sample.text for sample in samples)
            assert all("memory" in sample.required_paths for sample in samples)
        if scenario == "rule_required_priority":
            assert all(sample.rule_target for sample in samples)
        if scenario == "state_required_disambiguation":
            assert all(sample.state_target for sample in samples)


def test_path_dependency_dataset_is_deterministic() -> None:
    left, left_tokenizer = build_path_dependency_datasets(samples_per_label=8, max_seq_len=32, seed=303)
    right, right_tokenizer = build_path_dependency_datasets(samples_per_label=8, max_seq_len=32, seed=303)

    assert left_tokenizer.vocab == right_tokenizer.vocab
    assert left == right


def test_path_dependency_benchmark_outputs_path_contribution_metrics(tmp_path: Path) -> None:
    summary = run_path_dependency_benchmark(
        output_dir=tmp_path,
        modes=("full", "no_memory_path", "no_state_path", "no_rule_path", "structure_only"),
        seeds=(202,),
        samples_per_label=20,
        train_per_label=12,
        seq_lens=(32,),
        model_sizes=("medium",),
        training_steps=12,
        device="cpu",
    )

    assert set(summary["modes"]) == {"full", "no_memory_path", "no_state_path", "no_rule_path", "structure_only"}
    assert set(summary["stage_gates"]) == {
        "full_engineering_failures_zero",
        "full_dependency_average_accuracy",
        "no_memory_path_drop",
        "no_state_path_drop",
        "no_rule_path_drop",
        "structure_only_drop",
        "surface_group_flip_accuracy",
    }
    assert "memory_required_two_hop" in summary["scenario_average_accuracy"]["full"]
    assert "memory_required_drop" in summary["comparison"]["no_memory_path"]
    assert isinstance(summary["weak_modules"], list)
    assert isinstance(summary["passes_stage_gate"], bool)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "path_metrics.csv").exists()
    assert (tmp_path / "ablation_drop.csv").exists()
    assert (tmp_path / "surface_group_flips.csv").exists()
    assert (tmp_path / "trace_contribution.csv").exists()
    assert (tmp_path / "failure_cases.json").exists()
