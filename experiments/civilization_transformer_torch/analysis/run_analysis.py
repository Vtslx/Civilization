from __future__ import annotations

from .pipeline import run_hidden_state_analysis


def main() -> None:
    result = run_hidden_state_analysis()
    print("metrics_path", result.metrics_path)
    print("csv_path", result.csv_path)
    print("pca_plot_path", result.pca_plot_path)
    print("umap_plot_path", result.umap_plot_path)
    print("adjusted_rand_score", result.metrics["analysis"]["adjusted_rand_score"])
    print("initial_loss", result.metrics["training"]["initial_loss"])
    print("final_loss", result.metrics["training"]["final_loss"])
    print("loss_decreased", result.metrics["training"]["loss_decreased"])


if __name__ == "__main__":
    main()
