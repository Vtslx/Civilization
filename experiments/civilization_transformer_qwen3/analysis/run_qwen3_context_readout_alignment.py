from __future__ import annotations

from .context_readout_alignment import run_qwen3_context_readout_alignment


def main() -> None:
    summary = run_qwen3_context_readout_alignment()
    print(
        "qwen3_context_readout_alignment_complete "
        f"passes_stage_gate={summary['passes_stage_gate']} "
        f"allows_stage32_rerun={summary['allows_stage32_rerun']}"
    )


if __name__ == "__main__":
    main()
