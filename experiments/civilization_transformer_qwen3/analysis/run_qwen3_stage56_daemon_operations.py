from __future__ import annotations

import argparse
import json

from .stage56_daemon_operations import (
    DEFAULT_OUTPUT_DIR,
    Stage56DaemonConfig,
    port_available,
    read_pid_status,
    rotate_log_file,
    run_stage56_fake_daemon_smoke,
    validate_stage56_daemon_bundle,
    write_stage56_daemon_bundle,
)


def _config_from_args(args: argparse.Namespace) -> Stage56DaemonConfig:
    return Stage56DaemonConfig(
        host=args.host,
        port=args.port,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        pid_file=args.pid_file,
        log_file=args.log_file,
        python_bin=args.python_bin,
        readiness_timeout_seconds=args.readiness_timeout_seconds,
        stop_timeout_seconds=args.stop_timeout_seconds,
        max_log_bytes=args.max_log_bytes,
        log_backup_count=args.log_backup_count,
        offline=not args.allow_online,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage56 daemon operations packaging and audits")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-bundle", action="store_true")
    parser.add_argument("--validate-bundle", action="store_true")
    parser.add_argument("--rotate-log", action="store_true")
    parser.add_argument("--pid-status", action="store_true")
    parser.add_argument("--check-port", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--package-manifest", default="experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "experiments/civilization_transformer_qwen3/artifacts/"
            "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
        ),
    )
    parser.add_argument("--model-path", default="/home/yike/AoNeb-01/Models/Qwen3-0.6B")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default="experiments/civilization_transformer_qwen3/artifacts/logs")
    parser.add_argument("--pid-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.pid")
    parser.add_argument("--log-file", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.log")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--readiness-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-log-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=5)
    parser.add_argument("--allow-online", action="store_true")
    args = parser.parse_args()
    actions = [
        args.smoke,
        args.write_bundle,
        args.validate_bundle,
        args.rotate_log,
        args.pid_status,
        args.check_port,
    ]
    if sum(bool(action) for action in actions) != 1:
        parser.error("select exactly one action")
    if args.smoke:
        result = run_stage56_fake_daemon_smoke(output_dir=args.output_dir, port=0)
    elif args.write_bundle:
        result = write_stage56_daemon_bundle(output_dir=args.output_dir, config=_config_from_args(args))
    elif args.validate_bundle:
        result = validate_stage56_daemon_bundle(output_dir=args.output_dir)
    elif args.rotate_log:
        result = rotate_log_file(args.log_file, max_bytes=args.max_log_bytes, backup_count=args.log_backup_count)
    elif args.pid_status:
        result = read_pid_status(args.pid_file)
    else:
        available = port_available(args.host, args.port)
        result = {"host": args.host, "port": args.port, "available": available}
        if not available:
            raise SystemExit(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
