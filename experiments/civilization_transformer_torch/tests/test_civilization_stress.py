from pathlib import Path

from experiments.civilization_transformer_torch.analysis.run_civilization_stress_test import run_civilization_stress_test


def test_civilization_stress_small_and_medium_single_seed_pass(tmp_path: Path) -> None:
    summary = run_civilization_stress_test(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        seq_lens=(18,),
        model_sizes=("small", "medium"),
        training_steps=35,
        device="cpu",
        run_regressions=False,
    )

    assert summary["num_training_runs"] == 2
    assert summary["num_context_rows"] == 128
    assert set(summary["average_accuracy_by_size"]) == {"small", "medium"}
    assert summary["losses_decreased"]
    assert summary["stage11_regression"]["allows_hidden_state_injection_planning"]
    assert all(value == 0 for value in summary["failures"].values())
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "runs.json").exists()
    assert (tmp_path / "context_metrics.csv").exists()
