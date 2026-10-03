from __future__ import annotations

import argparse
import json

from .stage58_service_doctor import (
    DEFAULT_OUTPUT_DIR,
    Stage58DoctorConfig,
    generate_stage58_runbook,
    run_stage58_doctor,
)


def _config(args: argparse.Namespace) -> Stage58DoctorConfig:
    return Stage58DoctorConfig(
        host=args.host,
        port=args.port,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
        repo_root=args.repo_root,
        python_bin=args.python_bin,
        model_path=args.model_path,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        daemon_output_dir=args.daemon_output_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        pid_file=args.pid_file,
        log_file=args.log_file,
        systemd_unit_name=args.systemd_unit_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage58 service doctor and runbook generator")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--write-runbook", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--repo-root", default="/home/yike/AoNeb-01")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--model-path", default="/home/yike/AoNeb-01/Models/Qwen3-0.6B")
    parser.add_argument("--package-manifest", default="experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "experiments/civilization_transformer_qwen3/artifacts/"
            "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
        ),
    )
    parser.add_argument("--daemon-output-dir", default="experiments/civilization_transformer_qwen3/artifacts/stage56_daemon_operations")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default="experiments/civilization_transformer_qwen3/artifacts/logs")
    parser.add_argument("--pid-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.pid")
    parser.add_argument("--log-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.log")
    parser.add_argument("--systemd-unit-name", default="aoneb-qwen3-civilization.service")
    args = parser.parse_args()
    if args.doctor == args.write_runbook:
        parser.error("select exactly one of --doctor or --write-runbook")
    config = _config(args)
    result = run_stage58_doctor(output_dir=args.output_dir, config=config) if args.doctor else generate_stage58_runbook(output_dir=args.output_dir, config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
