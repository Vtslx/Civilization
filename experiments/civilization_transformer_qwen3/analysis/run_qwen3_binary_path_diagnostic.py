from __future__ import annotations

from .binary_path_diagnostic_benchmark import run_qwen3_binary_path_diagnostic


def main() -> None:
    summary = run_qwen3_binary_path_diagnostic()
    print(
        "qwen3_binary_path_diagnostic_complete "
        f"passes_stage_gate={summary['passes_stage_gate']} "
        f"allows_real_task_remigration={summary['allows_real_task_remigration']}"
    )


if __name__ == "__main__":
    main()
