from pathlib import Path

from civilization.research.torch_line.analysis import HARD_LOGIC_SCENARIOS, LOGIC_LABELS, build_hard_logic_datasets
from civilization.research.torch_line.analysis.run_chain_state_repair import run_chain_state_repair
from civilization.research.torch_line.analysis.run_hard_logic_benchmark import run_hard_logic_benchmark


def test_chain_supervision_metadata_is_deterministic_and_complete() -> None:
    first, tokenizer = build_hard_logic_datasets(samples_per_label=40, max_seq_len=32, seed=202, include_chain_supervision=True)
    second, second_tokenizer = build_hard_logic_datasets(samples_per_label=40, max_seq_len=32, seed=202, include_chain_supervision=True)

    assert first == second
    assert tokenizer.vocab == second_tokenizer.vocab
    assert set(first) == set(HARD_LOGIC_SCENARIOS)
    for scenario, samples in first.items():
        assert len(samples) == len(LOGIC_LABELS) * 40
        for sample in samples:
            assert len(sample.token_ids) == 32
            assert max(sample.token_ids) < tokenizer.vocab_size
            assert min(sample.token_ids) >= 0
            assert sample.chain_nodes
            assert sample.final_target
            if scenario == "two_hop_logic":
                assert sample.logic_depth == 2
                assert len(sample.chain_edges) == 2
            if scenario == "three_hop_logic":
                assert sample.logic_depth == 3
                assert len(sample.chain_edges) == 3
            if scenario == "mixed_logic_priority":
                assert sample.primary_label_rule
                assert len(sample.secondary_labels) >= 1
            if scenario == "masked_keywords_hard":
                tokens = set(tokenizer.tokenize(sample.text))
                assert tokens.isdisjoint({"because", "therefore", "not", "never", "always", "critical", "priority", "if", "then"})


def test_hard_benchmark_chain_state_alignment_mode_outputs_chain_metrics(tmp_path: Path) -> None:
    summary = run_hard_logic_benchmark(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=40,
        train_per_label=25,
        seq_lens=(32,),
        model_sizes=("medium",),
        training_steps=30,
        device="cpu",
        run_stage16_regression=False,
        training_mode="chain_state_alignment",
    )

    assert summary["training_mode"] == "chain_state_alignment"
    assert summary["num_training_runs"] == 1
    assert summary["losses_decreased"]
    assert summary["chain_metrics"]["chain_step_accuracy"] >= 0.0
    assert summary["chain_metrics"]["final_target_accuracy"] >= 0.0
    assert summary["chain_metrics"]["priority_control_accuracy"] >= 0.0
    assert all(value == 0 for value in summary["failures"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "scenario_metrics.csv").exists()


def test_chain_state_repair_improves_stage17_failure_scenarios(tmp_path: Path) -> None:
    summary = run_chain_state_repair(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=40,
        train_per_label=25,
        seq_lens=(32,),
        model_sizes=("medium",),
        baseline_training_steps=20,
        repair_training_steps=30,
        device="cpu",
        run_stage16_regression=False,
    )

    assert summary["comparison"]["two_hop_logic"]["absolute_improvement"] >= 0.25
    assert summary["comparison"]["three_hop_logic"]["absolute_improvement"] >= 0.25
    assert summary["comparison"]["mixed_logic_priority"]["absolute_improvement"] >= 0.10
    assert summary["repaired"]["average_accuracy_by_scenario"]["two_hop_logic"] >= 0.60
    assert summary["repaired"]["average_accuracy_by_scenario"]["three_hop_logic"] >= 0.60
    assert summary["repaired"]["average_accuracy_by_scenario"]["mixed_logic_priority"] >= 0.60
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "baseline_vs_repaired.csv").exists()
    assert (tmp_path / "chain_metrics.json").exists()
    assert (tmp_path / "failure_cases.json").exists()
    assert (tmp_path / "trace_failures.json").exists()
