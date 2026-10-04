from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..model import CivilizationAblationConfig
from .dataset import HARD_LOGIC_SCENARIOS
from .run_chain_state_full_benchmark import FOCUS_SCENARIOS
from .run_hard_logic_benchmark import run_hard_logic_benchmark


ABLATION_MODES: dict[str, CivilizationAblationConfig] = {
    "full": CivilizationAblationConfig(),
    "no_memory_path": CivilizationAblationConfig(use_memory_path=False),
    "no_state_path": CivilizationAblationConfig(use_state_path=False),
    "no_rule_path": CivilizationAblationConfig(use_rule_path=False),
    "no_chain_state_loss": CivilizationAblationConfig(use_chain_state_loss=False, use_final_state_loss=False),
    "no_priority_control_loss": CivilizationAblationConfig(use_priority_control_loss=False),
    "no_centroid_separation_loss": CivilizationAblationConfig(use_centroid_separation_loss=False),
    "no_hard_negative_loss": CivilizationAblationConfig(use_hard_negative_loss=False),
    "structure_only": CivilizationAblationConfig(
        use_chain_state_loss=False,
        use_final_state_loss=False,
        use_priority_control_loss=False,
        use_centroid_separation_loss=False,
        use_hard_negative_loss=False,
    ),
}


def run_civilization_ablation_study(
    output_dir: str | Path = "artifacts/torch-line/civilization_ablation_study",
    modes: tuple[str, ...] = tuple(ABLATION_MODES),
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 300,
    train_per_label: int = 200,
    seq_lens: tuple[int, ...] = (32, 64),
    model_sizes: tuple[str, ...] = ("medium", "large_toy"),
    training_steps: int = 45,
    device: str = "cpu",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mode_summaries: dict[str, dict] = {}
    for mode in modes:
        if mode not in ABLATION_MODES:
            raise ValueError(f"unknown ablation mode {mode}")
        summary = run_hard_logic_benchmark(
            output_dir=output_path / mode,
            seeds=seeds,
            samples_per_label=samples_per_label,
            train_per_label=train_per_label,
            seq_lens=seq_lens,
            model_sizes=model_sizes,
            training_steps=training_steps,
            device=device,
            run_stage16_regression=False,
            training_mode="chain_state_alignment",
            ablation_config=ABLATION_MODES[mode],
        )
        mode_summaries[mode] = summary
    comparison = _compare_modes(mode_summaries)
    trace_contribution = _trace_contribution(mode_summaries)
    gates = _stage_gates(mode_summaries, comparison)
    summary = {
        "modes": list(modes),
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "seq_lens": list(seq_lens),
        "model_sizes": list(model_sizes),
        "mode_summaries": _compact_mode_summaries(mode_summaries),
        "comparison": comparison,
        "trace_contribution": trace_contribution,
        "stage_gates": gates,
        "weak_contribution_modules": _weak_modules(comparison),
        "passes_stage_gate": all(gates.values()),
        "allows_large_model_migration_planning": all(gates.values()),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "failure_cases.json").write_text(json.dumps(_collect_failure_cases(output_path, modes), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_ablation_metrics(output_path / "ablation_metrics.csv", mode_summaries)
    _write_mode_comparison(output_path / "mode_comparison.csv", comparison)
    _write_scenario_drop(output_path / "scenario_drop.csv", comparison)
    _write_trace_contribution(output_path / "trace_contribution.csv", trace_contribution)
    return summary


def _compact_mode_summaries(mode_summaries: dict[str, dict]) -> dict[str, dict]:
    return {
        mode: {
            "average_accuracy_by_scenario": summary["average_accuracy_by_scenario"],
            "failures": summary["failures"],
            "losses_decreased": summary["losses_decreased"],
            "chain_metrics": summary["chain_metrics"],
            "passes_stage_gate": summary["passes_stage_gate"],
        }
        for mode, summary in mode_summaries.items()
    }


def _compare_modes(mode_summaries: dict[str, dict]) -> dict[str, dict]:
    full = mode_summaries["full"]["average_accuracy_by_scenario"]
    comparison: dict[str, dict] = {}
    for mode, summary in mode_summaries.items():
        values = summary["average_accuracy_by_scenario"]
        drops = {scenario: full[scenario] - values[scenario] for scenario in HARD_LOGIC_SCENARIOS}
        focus_drops = {scenario: drops[scenario] for scenario in FOCUS_SCENARIOS}
        comparison[mode] = {
            "scenario_drop": drops,
            "focus_scenario_drop": focus_drops,
            "focus_average_drop": float(np.mean(list(focus_drops.values()))),
            "average_drop": float(np.mean(list(drops.values()))),
        }
    return comparison


def _trace_contribution(mode_summaries: dict[str, dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for mode, summary in mode_summaries.items():
        failures = summary["failures"]
        result[mode] = {
            "trace_failures": failures["trace_failures"],
            "nan_inf": failures["nan_inf"],
            "memory_attention_bad": failures.get("memory_attention_bad", 0),
            "rule_path_active": mode != "no_rule_path",
            "state_path_active": mode != "no_state_path",
            "memory_path_active": mode != "no_memory_path",
        }
    return result


def _stage_gates(mode_summaries: dict[str, dict], comparison: dict[str, dict]) -> dict[str, bool]:
    full = mode_summaries["full"]
    full_failures_ok = all(value == 0 for value in full["failures"].values())
    full_focus_ok = all(full["average_accuracy_by_scenario"][scenario] >= 0.60 for scenario in ("two_hop_logic", "three_hop_logic", "mixed_logic_priority", "ood_surface"))
    return {
        "full_mode_passes": bool(full_failures_ok and full_focus_ok),
        "no_chain_state_loss_drops_two_hop": comparison["no_chain_state_loss"]["scenario_drop"]["two_hop_logic"] >= 0.20,
        "no_chain_state_loss_drops_three_hop": comparison["no_chain_state_loss"]["scenario_drop"]["three_hop_logic"] >= 0.20,
        "no_priority_control_loss_drops_mixed": comparison["no_priority_control_loss"]["scenario_drop"]["mixed_logic_priority"] >= 0.10,
        "structure_only_focus_drop": comparison["structure_only"]["focus_average_drop"] >= 0.25,
    }


def _weak_modules(comparison: dict[str, dict]) -> list[str]:
    weak: list[str] = []
    for mode, module in (("no_memory_path", "memory_path"), ("no_state_path", "state_path"), ("no_rule_path", "rule_path")):
        if mode in comparison and abs(comparison[mode]["focus_average_drop"]) < 0.02:
            weak.append(module)
    return weak


def _collect_failure_cases(output_path: Path, modes: tuple[str, ...]) -> dict[str, list]:
    failures: dict[str, list] = {}
    for mode in modes:
        failure_path = output_path / mode / "failure_cases.json"
        failures[mode] = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else []
    return failures


def _write_ablation_metrics(path: Path, mode_summaries: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "passes_stage_gate", "losses_decreased", *HARD_LOGIC_SCENARIOS])
        for mode, summary in mode_summaries.items():
            writer.writerow([
                mode,
                summary["passes_stage_gate"],
                summary["losses_decreased"],
                *[summary["average_accuracy_by_scenario"][scenario] for scenario in HARD_LOGIC_SCENARIOS],
            ])


def _write_mode_comparison(path: Path, comparison: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "average_drop", "focus_average_drop", *[f"{scenario}_drop" for scenario in FOCUS_SCENARIOS]])
        for mode, values in comparison.items():
            writer.writerow([
                mode,
                values["average_drop"],
                values["focus_average_drop"],
                *[values["focus_scenario_drop"][scenario] for scenario in FOCUS_SCENARIOS],
            ])


def _write_scenario_drop(path: Path, comparison: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", *HARD_LOGIC_SCENARIOS])
        for mode, values in comparison.items():
            writer.writerow([mode, *[values["scenario_drop"][scenario] for scenario in HARD_LOGIC_SCENARIOS]])


def _write_trace_contribution(path: Path, trace_contribution: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "trace_failures", "nan_inf", "memory_attention_bad", "memory_path_active", "state_path_active", "rule_path_active"])
        for mode, values in trace_contribution.items():
            writer.writerow([
                mode,
                values["trace_failures"],
                values["nan_inf"],
                values["memory_attention_bad"],
                values["memory_path_active"],
                values["state_path_active"],
                values["rule_path_active"],
            ])


def main() -> None:
    summary = run_civilization_ablation_study()
    print("stage_gates", summary["stage_gates"])
    print("weak_contribution_modules", summary["weak_contribution_modules"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
