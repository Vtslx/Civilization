from __future__ import annotations

import argparse
import json

from .stage57_daemon_recovery import DEFAULT_OUTPUT_DIR, Stage57RecoveryConfig, run_stage57_recovery_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage57 daemon crash recovery audit")
    parser.add_argument("--real", action="store_true", help="run real daemon recovery audit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--model-path", default="/home/yike/AoNeb-01/Models/Qwen3-0.6B")
    parser.add_argument("--package-manifest", default="experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "experiments/civilization_transformer_qwen3/artifacts/"
            "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--daemon-output-dir", default="experiments/civilization_transformer_qwen3/artifacts/stage56_daemon_operations")
    parser.add_argument("--log-dir", default="experiments/civilization_transformer_qwen3/artifacts/logs")
    parser.add_argument("--pid-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.pid")
    parser.add_argument("--log-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.log")
    parser.add_argument("--readiness-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-log-bytes", type=int, default=1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    args = parser.parse_args()
    if not args.real:
        parser.error("--real is required; Stage57 recovery audit intentionally exercises daemon scripts")
    config = Stage57RecoveryConfig(
        host=args.host,
        port=args.port,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
        model_path=args.model_path,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        output_dir=args.output_dir,
        daemon_output_dir=args.daemon_output_dir,
        log_dir=args.log_dir,
        pid_file=args.pid_file,
        log_file=args.log_file,
        readiness_timeout_seconds=args.readiness_timeout_seconds,
        max_log_bytes=args.max_log_bytes,
        log_backup_count=args.log_backup_count,
        python_bin=args.python_bin,
    )
    summary = run_stage57_recovery_smoke(output_dir=args.output_dir, config=config, real=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
