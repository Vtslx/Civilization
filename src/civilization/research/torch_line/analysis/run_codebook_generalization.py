from __future__ import annotations

from .codebook import run_codebook_generalization_matrix


def main() -> None:
    summary = run_codebook_generalization_matrix()
    print("seeds", summary["seeds"])
    print("samples_per_label", summary["samples_per_label"])
    print("train_per_label", summary["train_per_label"])
    print("num_runs", summary["num_runs"])
    print("trained_mean_average_accuracy_by_variant", summary["trained_mean_average_accuracy_by_variant"])
    print("masked_keyword_drop", summary["masked_keyword_drop"])
    print("passes_canonical_gate", summary["passes_canonical_gate"])
    print("passes_synonym_gate", summary["passes_synonym_gate"])
    print("passes_perturbed_gate", summary["passes_perturbed_gate"])
    print("template_leakage_indicated", summary["template_leakage_indicated"])
    print("allows_hidden_state_injection_planning", summary["allows_hidden_state_injection_planning"])
    print("worst_variant", summary["worst_variant"])


if __name__ == "__main__":
    main()
