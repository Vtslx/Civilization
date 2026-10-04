from __future__ import annotations

import argparse
import json

from .stage55_service_deployment import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CENTROID_BUNDLE,
    DEFAULT_MODEL_PATH,
    DEFAULT_PACKAGE_MANIFEST,
    Stage55DeploymentConfig,
    Stage55ServiceClient,
    run_stage55_fake_deployment_smoke,
    serve_stage55_real_service,
    stop_pid_file,
    validate_stage55_deployment_bundle,
    write_stage55_deployment_bundle,
)


def _config_from_args(args: argparse.Namespace) -> Stage55DeploymentConfig:
    return Stage55DeploymentConfig(
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
        request_timeout_seconds=args.request_timeout_seconds,
        python_bin=args.python_bin,
        offline=not args.allow_online,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage55 service deployment packaging and daemon entrypoints")
    parser.add_argument("--smoke", action="store_true", help="run fake service deployment smoke")
    parser.add_argument("--write-bundle", action="store_true", help="write deployment scripts and manifest")
    parser.add_argument("--validate-bundle", action="store_true", help="validate deployment scripts and manifest")
    parser.add_argument("--serve", action="store_true", help="run the real managed service in foreground")
    parser.add_argument("--stop", action="store_true", help="stop process referenced by --pid-file")
    parser.add_argument("--health", action="store_true", help="query service health")
    parser.add_argument("--status", action="store_true", help="query service admin status")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--package-manifest", default=str(DEFAULT_PACKAGE_MANIFEST))
    parser.add_argument("--centroid-bundle", default=str(DEFAULT_CENTROID_BUNDLE))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default="artifacts/civilization/logs")
    parser.add_argument("--pid-file", default="artifacts/civilization/logs/stage55_service.pid")
    parser.add_argument("--log-file", default="artifacts/civilization/logs/stage55_service.log")
    parser.add_argument("--request-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--allow-online", action="store_true", help="do not force HF/Transformers offline env in generated scripts")
    args = parser.parse_args()
    config = _config_from_args(args)
    actions = [args.smoke, args.write_bundle, args.validate_bundle, args.serve, args.stop, args.health, args.status]
    if sum(bool(action) for action in actions) != 1:
        parser.error("select exactly one action")
    if args.smoke:
        summary = run_stage55_fake_deployment_smoke(output_dir=args.output_dir, port=0)
    elif args.write_bundle:
        summary = write_stage55_deployment_bundle(output_dir=args.output_dir, config=config)
    elif args.validate_bundle:
        summary = validate_stage55_deployment_bundle(output_dir=args.output_dir)
    elif args.stop:
        summary = stop_pid_file(args.pid_file)
    elif args.health:
        summary = Stage55ServiceClient(f"http://{args.host}:{args.port}", timeout=args.request_timeout_seconds).health()
    elif args.status:
        summary = Stage55ServiceClient(f"http://{args.host}:{args.port}", timeout=args.request_timeout_seconds).status()
    else:
        serve_stage55_real_service(config)
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
