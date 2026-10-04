from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage45_adapter_package import (
    DEFAULT_OUTPUT_DIR,
    build_stage45_adapter_package,
    run_stage45_eval_harness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the Stage45 Qwen3 Civilization Adapter package")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage44c-summary", default="artifacts/civilization/stage44c_real_task_runner/summary.json")
    parser.add_argument("--stage44a-root", default="artifacts/civilization/stage44a_local_multiclass_integration")
    parser.add_argument("--stage44b6-root", default="artifacts/civilization/stage44b6_external_recovery_training")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--skip-task-checkpoints", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    output_dir = Path(args.output_dir)
    manifest = build_stage45_adapter_package(
        output_dir=output_dir,
        stage44c_summary_path=args.stage44c_summary,
        stage44a_root=args.stage44a_root,
        stage44b6_root=args.stage44b6_root,
        seeds=seeds,
        include_task_checkpoints=not args.skip_task_checkpoints,
    )
    summary = run_stage45_eval_harness(
        manifest_path=output_dir / "package_manifest.json",
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "package_manifest": str(output_dir / "package_manifest.json"),
                "checkpoint_count": len(manifest.checkpoints),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
