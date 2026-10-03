from pathlib import Path

from experiments.civilization_transformer_torch.analysis.dataset import HARD_LOGIC_SCENARIOS
from experiments.civilization_transformer_torch.analysis.run_chain_state_full_benchmark import FOCUS_SCENARIOS, run_chain_state_full_benchmark


def test_chain_state_full_benchmark_smoke_outputs_stability_artifacts(tmp_path: Path) -> None:
    summary = run_chain_state_full_benchmark(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=40,
        train_per_label=25,
        seq_lens=(32,),
        model_sizes=("medium",),
        training_steps=30,
        device="cpu",
        run_stage16_regression=False,
    )

    assert summary["training_mode"] == "chain_state_alignment"
    assert summary["num_training_runs"] == 1
    assert set(summary["scenario_stability"]) == set(HARD_LOGIC_SCENARIOS)
    assert set(summary["seed_stability"]) == {"202"}
    assert set(summary["size_comparison"]) == {"medium"}
    assert set(summary["seq_len_comparison"]) == {"32"}
    assert summary["training_summary"]["num_training_runs"] == 1
    assert summary["training_summary"]["all_total_loss_decreased"]
    assert all(key in summary["stability_gates"] for key in ("focus_seed_min_accuracy", "focus_seed_std", "large_toy_not_below_medium", "seq64_not_below_seq32"))
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "runs.json").exists()
    assert (tmp_path / "scenario_metrics.csv").exists()
    assert (tmp_path / "seed_stability.csv").exists()
    assert (tmp_path / "size_comparison.csv").exists()
    assert (tmp_path / "seq_len_comparison.csv").exists()
    assert (tmp_path / "failure_cases.json").exists()
    assert (tmp_path / "trace_failures.json").exists()


def test_chain_state_full_benchmark_medium_two_seed_matrix_passes(tmp_path: Path) -> None:
    summary = run_chain_state_full_benchmark(
        output_dir=tmp_path,
        seeds=(202, 303),
        samples_per_label=120,
        train_per_label=80,
        seq_lens=(48,),
        model_sizes=("medium",),
        training_steps=40,
        device="cpu",
        run_stage16_regression=False,
    )

    assert summary["num_training_runs"] == 2
    assert summary["num_scenario_rows"] == 2 * len(HARD_LOGIC_SCENARIOS)
    assert summary["training_summary"]["all_total_loss_decreased"]
    assert summary["training_summary"]["all_classification_loss_decreased"]
    assert summary["training_summary"]["all_chain_state_loss_decreased"]
    assert summary["training_summary"]["all_priority_control_loss_decreased"]
    assert all(summary["scenario_stability"][scenario]["mean"] >= 0.60 for scenario in FOCUS_SCENARIOS)
    assert all(value == 0 for value in summary["failures"].values())
    assert summary["passes_stage_gate"]
