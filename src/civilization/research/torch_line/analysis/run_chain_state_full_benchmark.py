from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .dataset import HARD_LOGIC_SCENARIOS
from .run_chain_state_repair import REPAIRED_GATES
from .run_hard_logic_benchmark import run_hard_logic_benchmark


FOCUS_SCENARIOS = ("two_hop_logic", "three_hop_logic", "mixed_logic_priority", "ood_surface", "synonym_hard")


def run_chain_state_full_benchmark(
    output_dir: str | Path = "artifacts/torch-line/chain_state_full_benchmark",
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606, 707, 808, 909, 1001, 1112),
    samples_per_label: int = 300,
    train_per_label: int = 200,
    seq_lens: tuple[int, ...] = (32, 48, 64),
    model_sizes: tuple[str, ...] = ("medium", "large_toy"),
    training_steps: int = 45,
    device: str = "cpu",
    run_stage16_regression: bool = True,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hard_summary = run_hard_logic_benchmark(
        output_dir=output_path,
        seeds=seeds,
        samples_per_label=samples_per_label,
        train_per_label=train_per_label,
        seq_lens=seq_lens,
        model_sizes=model_sizes,
        training_steps=training_steps,
        device=device,
        run_stage16_regression=run_stage16_regression,
        training_mode="chain_state_alignment",
    )
    scenario_rows = _read_scenario_rows(output_path / "scenario_metrics.csv")
    run_rows = _read_runs(output_path / "runs.json")
    scenario_stability = _scenario_stability(scenario_rows)
    seed_stability = _group_stability(scenario_rows, "seed")
    size_comparison = _group_stability(scenario_rows, "size")
    seq_len_comparison = _group_stability(scenario_rows, "seq_len")
    training_summary = _training_summary(run_rows)
    stability_gates = _stability_gates(seed_stability, size_comparison, seq_len_comparison)
    passes_stage_gate = bool(
        hard_summary["passes_stage_gate"]
        and training_summary["all_total_loss_decreased"]
        and training_summary["all_classification_loss_decreased"]
        and training_summary["all_chain_state_loss_decreased"]
        and training_summary["all_priority_control_loss_decreased"]
        and all(stability_gates.values())
    )
    summary = {
        **hard_summary,
        "scenario_stability": scenario_stability,
        "seed_stability": seed_stability,
        "size_comparison": size_comparison,
        "seq_len_comparison": seq_len_comparison,
        "training_summary": training_summary,
        "stability_gates": stability_gates,
        "allows_large_model_migration_planning": passes_stage_gate,
        "passes_stage_gate": passes_stage_gate,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_group_csv(output_path / "seed_stability.csv", seed_stability)
    _write_group_csv(output_path / "size_comparison.csv", size_comparison)
    _write_group_csv(output_path / "seq_len_comparison.csv", seq_len_comparison)
    return summary


def _read_scenario_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "seed": int(row["seed"]),
                "size": row["size"],
                "seq_len": int(row["seq_len"]),
                "scenario": row["scenario"],
                "accuracy": float(row["accuracy"]),
                "macro_accuracy": float(row["macro_accuracy"]),
                "adjusted_rand_score": float(row["adjusted_rand_score"]),
                "hidden_norm": float(row["hidden_norm"]),
                "trace_count": int(row["trace_count"]),
            }
            for row in reader
        ]


def _read_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_stability(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for scenario in HARD_LOGIC_SCENARIOS:
        values = np.array([row["accuracy"] for row in rows if row["scenario"] == scenario], dtype=float)
        result[scenario] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
            "passes_gate": bool(values.mean() >= REPAIRED_GATES.get(scenario, 0.0)),
        }
    return result


def _group_stability(rows: list[dict], group_key: str) -> dict[str, dict]:
    groups = sorted({row[group_key] for row in rows})
    result: dict[str, dict] = {}
    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        scenario_values = {
            scenario: float(np.mean([row["accuracy"] for row in group_rows if row["scenario"] == scenario]))
            for scenario in HARD_LOGIC_SCENARIOS
        }
        focus_values = {scenario: scenario_values[scenario] for scenario in FOCUS_SCENARIOS}
        result[str(group)] = {
            "average_accuracy": float(np.mean(list(scenario_values.values()))),
            "scenario_accuracy": scenario_values,
            "focus_scenario_accuracy": focus_values,
            "focus_min_accuracy": float(min(focus_values.values())),
        }
    return result


def _training_summary(runs: list[dict]) -> dict:
    training_rows = [run["training"] for run in runs]
    return {
        "num_training_runs": len(training_rows),
        "all_total_loss_decreased": all(row["total_loss_decreased"] for row in training_rows),
        "all_classification_loss_decreased": all(row["classification_loss_decreased"] for row in training_rows),
        "all_chain_state_loss_decreased": all(row.get("chain_state_loss_decreased", False) for row in training_rows),
        "all_priority_control_loss_decreased": all(row.get("priority_control_loss_decreased", False) for row in training_rows),
        "average_chain_step_accuracy": float(np.mean([row.get("chain_step_accuracy", 0.0) for row in training_rows])),
        "average_final_target_accuracy": float(np.mean([row.get("final_target_accuracy", 0.0) for row in training_rows])),
        "average_priority_control_accuracy": float(np.mean([row.get("priority_control_accuracy", 0.0) for row in training_rows])),
    }


def _stability_gates(seed_stability: dict[str, dict], size_comparison: dict[str, dict], seq_len_comparison: dict[str, dict]) -> dict[str, bool]:
    focus_values_by_seed = [values["focus_min_accuracy"] for values in seed_stability.values()]
    focus_std_by_scenario = []
    for scenario in FOCUS_SCENARIOS:
        focus_std_by_scenario.append(float(np.std([values["scenario_accuracy"][scenario] for values in seed_stability.values()])))
    medium_avg = size_comparison.get("medium", {}).get("average_accuracy", 0.0)
    large_avg = size_comparison.get("large_toy", {}).get("average_accuracy", 0.0)
    seq32_avg = seq_len_comparison.get("32", {}).get("average_accuracy", 0.0)
    seq64_avg = seq_len_comparison.get("64", {}).get("average_accuracy", 0.0)
    return {
        "focus_seed_min_accuracy": bool(min(focus_values_by_seed) >= 0.55),
        "focus_seed_std": bool(max(focus_std_by_scenario) <= 0.12),
        "large_toy_not_below_medium": True if "large_toy" not in size_comparison or "medium" not in size_comparison else bool(large_avg + 0.05 >= medium_avg),
        "seq64_not_below_seq32": True if "64" not in seq_len_comparison or "32" not in seq_len_comparison else bool(seq64_avg + 0.08 >= seq32_avg),
    }


def _write_group_csv(path: Path, grouped: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "average_accuracy", "focus_min_accuracy", *HARD_LOGIC_SCENARIOS])
        for group, values in grouped.items():
            writer.writerow([
                group,
                values["average_accuracy"],
                values["focus_min_accuracy"],
                *[values["scenario_accuracy"][scenario] for scenario in HARD_LOGIC_SCENARIOS],
            ])


def main() -> None:
    summary = run_chain_state_full_benchmark()
    print("scenario_stability", summary["scenario_stability"])
    print("training_summary", summary["training_summary"])
    print("stability_gates", summary["stability_gates"])
    print("num_training_runs", summary["num_training_runs"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
