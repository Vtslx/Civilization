from __future__ import annotations

import csv
import json
from pathlib import Path

from .run_hard_logic_benchmark import run_hard_logic_benchmark


REPAIR_TARGETS = ("two_hop_logic", "three_hop_logic", "mixed_logic_priority", "ood_surface", "synonym_hard")
REPAIRED_GATES = {
    "canonical_hard": 0.85,
    "synonym_hard": 0.85,
    "masked_keywords_hard": 0.85,
    "two_hop_logic": 0.60,
    "three_hop_logic": 0.60,
    "mixed_logic_priority": 0.60,
    "ood_surface": 0.60,
}


def run_chain_state_repair(
    output_dir: str | Path = "artifacts/torch-line/chain_state_alignment",
    seeds: tuple[int, ...] = (202,),
    samples_per_label: int = 300,
    train_per_label: int = 200,
    seq_lens: tuple[int, ...] = (32, 48, 64),
    model_sizes: tuple[str, ...] = ("medium", "large_toy"),
    baseline_training_steps: int = 30,
    repair_training_steps: int = 45,
    device: str = "cpu",
    run_stage16_regression: bool = True,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline = run_hard_logic_benchmark(
        output_dir=output_path / "baseline",
        seeds=seeds,
        samples_per_label=samples_per_label,
        train_per_label=train_per_label,
        seq_lens=seq_lens,
        model_sizes=model_sizes,
        training_steps=baseline_training_steps,
        device=device,
        run_stage16_regression=run_stage16_regression,
        training_mode="hard_alignment_baseline",
    )
    repaired = run_hard_logic_benchmark(
        output_dir=output_path / "repaired",
        seeds=seeds,
        samples_per_label=samples_per_label,
        train_per_label=train_per_label,
        seq_lens=seq_lens,
        model_sizes=model_sizes,
        training_steps=repair_training_steps,
        device=device,
        run_stage16_regression=False,
        training_mode="chain_state_alignment",
    )
    comparison = _compare(baseline, repaired)
    summary = {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "baseline": {
            "average_accuracy_by_scenario": baseline["average_accuracy_by_scenario"],
            "failures": baseline["failures"],
            "passes_stage_gate": baseline["passes_stage_gate"],
        },
        "repaired": {
            "average_accuracy_by_scenario": repaired["average_accuracy_by_scenario"],
            "failures": repaired["failures"],
            "passes_stage_gate": repaired["passes_stage_gate"],
            "chain_metrics": repaired["chain_metrics"],
        },
        "comparison": comparison,
        "allows_large_model_migration_planning": _allows_migration(repaired, comparison),
        "passes_stage_gate": _allows_migration(repaired, comparison),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "chain_metrics.json").write_text(json.dumps(repaired["chain_metrics"], indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "failure_cases.json").write_text((output_path / "repaired" / "failure_cases.json").read_text(encoding="utf-8"), encoding="utf-8")
    (output_path / "trace_failures.json").write_text((output_path / "repaired" / "trace_failures.json").read_text(encoding="utf-8"), encoding="utf-8")
    _write_comparison(output_path / "baseline_vs_repaired.csv", comparison)
    return summary


def _compare(baseline: dict, repaired: dict) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for scenario, baseline_value in baseline["average_accuracy_by_scenario"].items():
        repaired_value = repaired["average_accuracy_by_scenario"][scenario]
        absolute = repaired_value - baseline_value
        relative = absolute / baseline_value if baseline_value else 0.0
        rows[scenario] = {
            "baseline_accuracy": baseline_value,
            "repaired_accuracy": repaired_value,
            "absolute_improvement": absolute,
            "relative_improvement": relative,
        }
    return rows


def _allows_migration(repaired: dict, comparison: dict[str, dict]) -> bool:
    if any(value != 0 for value in repaired["failures"].values()):
        return False
    for scenario, threshold in REPAIRED_GATES.items():
        if repaired["average_accuracy_by_scenario"][scenario] < threshold:
            return False
    if comparison["two_hop_logic"]["absolute_improvement"] < 0.25:
        return False
    if comparison["three_hop_logic"]["absolute_improvement"] < 0.25:
        return False
    if comparison["mixed_logic_priority"]["absolute_improvement"] < 0.10:
        return False
    return True


def _write_comparison(path: Path, comparison: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "baseline_accuracy", "repaired_accuracy", "absolute_improvement", "relative_improvement"])
        for scenario, values in comparison.items():
            writer.writerow([
                scenario,
                values["baseline_accuracy"],
                values["repaired_accuracy"],
                values["absolute_improvement"],
                values["relative_improvement"],
            ])


def main() -> None:
    summary = run_chain_state_repair()
    print("comparison", summary["comparison"])
    print("repaired_chain_metrics", summary["repaired"]["chain_metrics"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
