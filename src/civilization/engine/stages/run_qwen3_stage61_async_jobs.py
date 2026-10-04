from __future__ import annotations

import argparse
import json

from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig
from .stage61_async_jobs import (
    DEFAULT_JOB_LOG,
    DEFAULT_OUTPUT_DIR,
    Stage61JobConfig,
    build_stage61_real_service,
    run_stage61_async_jobs_smoke,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage61 async job service")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--require-token-for-predict", action="store_true")
    parser.add_argument("--require-token-for-admin", action="store_true", default=True)
    parser.add_argument("--no-token-for-admin", action="store_false", dest="require_token_for_admin")
    parser.add_argument("--audit-log-path", default="artifacts/civilization/logs/stage61_audit.jsonl")
    parser.add_argument("--max-in-flight", type=int, default=2)
    parser.add_argument("--per-client-qps", type=int, default=4)
    parser.add_argument("--max-queued-jobs", type=int, default=128)
    parser.add_argument("--job-log-path", default=str(DEFAULT_JOB_LOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--package-manifest", default="artifacts/civilization/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "artifacts/civilization/"
            "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if args.smoke == args.serve:
        parser.error("select exactly one of --smoke or --serve")
    if args.smoke:
        summary = run_stage61_async_jobs_smoke(output_dir=args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    security = Stage59SecurityConfig(
        bearer_token=args.bearer_token,
        require_token_for_predict=args.require_token_for_predict,
        require_token_for_admin=args.require_token_for_admin,
        audit_log_path=args.audit_log_path,
    )
    queue_config = Stage60QueueConfig(max_in_flight=args.max_in_flight, per_client_qps=args.per_client_qps)
    job_config = Stage61JobConfig(max_queued_jobs=args.max_queued_jobs, job_log_path=args.job_log_path)
    service = build_stage61_real_service(
        port=args.port,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "stage": "stage61_async_jobs",
                "status": "starting",
                "port": args.port,
                "queue_config": queue_config.__dict__,
                "job_config": job_config.__dict__,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
