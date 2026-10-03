from __future__ import annotations

from .codebook import run_codebook_matrix


def main() -> None:
    summary = run_codebook_matrix()
    print("seeds", summary["seeds"])
    print("samples_per_label", summary["samples_per_label"])
    print("num_runs", summary["num_runs"])
    print("trained_mean_average_accuracy", summary["trained_mean_average_accuracy"])
    print("trained_mean_min_accuracy", summary["trained_mean_min_accuracy"])
    print("passes_stage_gate", summary["passes_stage_gate"])
    for run in summary["runs"]:
        print(
            "run",
            run["seed"],
            run["model_state"],
            run["pooling"],
            "accuracy",
            run["nearest_neighbor_accuracy"],
            "macro",
            run["macro_accuracy"],
            "ari",
            run["adjusted_rand_score"],
            "confusion",
            run["easiest_confusion_pair"],
        )


if __name__ == "__main__":
    main()
