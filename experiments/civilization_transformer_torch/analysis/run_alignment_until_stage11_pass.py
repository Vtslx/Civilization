from __future__ import annotations

import json
from pathlib import Path

from .codebook import run_codebook_generalization_matrix


def run_alignment_until_stage11_pass(
    output_dir: str | Path = "experiments/civilization_transformer_torch/artifacts/codebook_generalization/alignment_until_pass",
    seeds: tuple[int, ...] = (202, 303, 404),
    initial_samples_per_label: int = 60,
    initial_train_per_label: int = 40,
    max_seq_len: int = 18,
    device: str = "cpu",
    max_iterations: int | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    iteration = 1
    samples_per_label = initial_samples_per_label
    train_per_label = initial_train_per_label

    while True:
        iteration_dir = output_path / f"iteration_{iteration:02d}"
        summary = run_codebook_generalization_matrix(
            output_dir=iteration_dir,
            seeds=seeds,
            samples_per_label=samples_per_label,
            train_per_label=train_per_label,
            device=device,
            template_bank="expanded_v1",
            model_training_mode="cross_template_alignment",
            max_seq_len=max_seq_len,
        )
        record = {
            "iteration": iteration,
            "summary_path": str(iteration_dir / "summary.json"),
            "samples_per_label": samples_per_label,
            "train_per_label": train_per_label,
            "trained_mean_average_accuracy_by_variant": summary["trained_mean_average_accuracy_by_variant"],
            "masked_keyword_drop": summary["masked_keyword_drop"],
            "allows_hidden_state_injection_planning": summary["allows_hidden_state_injection_planning"],
            "worst_variant": summary["worst_variant"],
            "continue_reason": None,
        }
        if summary["allows_hidden_state_injection_planning"]:
            record["continue_reason"] = "stage 11 gate passed"
            history.append(record)
            final = {"passed": True, "iterations": history, "final_summary": summary}
            (output_path / "iteration_history.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
            return final

        record["continue_reason"] = _failure_reason(summary)
        history.append(record)
        (output_path / "iteration_history.json").write_text(json.dumps({"passed": False, "iterations": history}, indent=2, ensure_ascii=False), encoding="utf-8")
        if max_iterations is not None and iteration >= max_iterations:
            raise RuntimeError(f"stage 11 did not pass after {max_iterations} iterations: {record['continue_reason']}")

        iteration += 1
        samples_per_label += 20
        train_per_label = max(40, int(samples_per_label * 2 / 3))
        if train_per_label >= samples_per_label:
            train_per_label = samples_per_label - 10


def _failure_reason(summary: dict) -> str:
    reasons: list[str] = []
    if not summary["passes_canonical_gate"]:
        reasons.append("canonical below 0.60")
    if not summary["passes_synonym_gate"]:
        reasons.append("synonym below 0.50")
    if not summary["passes_perturbed_gate"]:
        reasons.append("perturbed below 0.50")
    if summary["template_leakage_indicated"]:
        reasons.append("masked keyword drop above 0.20")
    return "; ".join(reasons) if reasons else "unknown gate failure"


def main() -> None:
    result = run_alignment_until_stage11_pass()
    final_summary = result["final_summary"]
    print("passed", result["passed"])
    print("iterations", len(result["iterations"]))
    print("trained_mean_average_accuracy_by_variant", final_summary["trained_mean_average_accuracy_by_variant"])
    print("masked_keyword_drop", final_summary["masked_keyword_drop"])
    print("allows_hidden_state_injection_planning", final_summary["allows_hidden_state_injection_planning"])
    print("worst_variant", final_summary["worst_variant"])


if __name__ == "__main__":
    main()
