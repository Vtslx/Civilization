from __future__ import annotations

import argparse
import json

from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig
from .stage61_async_jobs import Stage61JobConfig
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import (
    DEFAULT_JOB_LOG,
    DEFAULT_JOB_STATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULT_DIR,
    Stage65BatchConfig,
    build_stage65_real_service,
    run_stage65_batch_jobs_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage65 batch job service")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--require-token-for-predict", action="store_true")
    parser.add_argument("--require-token-for-admin", action="store_true", default=True)
    parser.add_argument("--no-token-for-admin", action="store_false", dest="require_token_for_admin")
    parser.add_argument("--audit-log-path", default="experiments/civilization_transformer_qwen3/artifacts/logs/stage65_audit.jsonl")
    parser.add_argument("--max-in-flight", type=int, default=2)
    parser.add_argument("--per-client-qps", type=int, default=4)
    parser.add_argument("--max-queued-jobs", type=int, default=128)
    parser.add_argument("--job-log-path", default=str(DEFAULT_JOB_LOG))
    parser.add_argument("--job-state-path", default=str(DEFAULT_JOB_STATE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--disable-recover-incomplete-jobs", action="store_true")
    parser.add_argument("--result-ttl-seconds", type=float, default=3600.0)
    parser.add_argument("--max-persisted-jobs", type=int, default=1000)
    parser.add_argument("--max-list-limit", type=int, default=200)
    parser.add_argument("--inline-result-row-limit", type=int, default=0)
    parser.add_argument("--max-result-page-limit", type=int, default=200)
    parser.add_argument("--max-batch-submit", type=int, default=64)
    parser.add_argument("--max-batch-result-jobs", type=int, default=64)
    parser.add_argument("--keep-result-files-on-prune", action="store_true")
    parser.add_argument("--disable-cleanup-on-persist", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--package-manifest", default="experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "experiments/civilization_transformer_qwen3/artifacts/"
            "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
        ),
    )
    parser.add_argument("--model-path", default="/home/yike/AoNeb-01/Models/Qwen3-0.6B")
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if args.smoke == args.serve:
        parser.error("select exactly one of --smoke or --serve")
    if args.smoke:
        summary = run_stage65_batch_jobs_smoke(output_dir=args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    security = Stage59SecurityConfig(
        bearer_token=args.bearer_token,
        require_token_for_predict=args.require_token_for_predict,
        require_token_for_admin=args.require_token_for_admin,
        audit_log_path=args.audit_log_path,
    )
    queue_config = Stage60QueueConfig(max_in_flight=args.max_in_flight, per_client_qps=args.per_client_qps)
    job_config = Stage61JobConfig(
        max_queued_jobs=args.max_queued_jobs,
        job_log_path=args.job_log_path,
        result_ttl_seconds=args.result_ttl_seconds,
    )
    persistence_config = Stage62PersistenceConfig(
        job_state_path=args.job_state_path,
        recover_incomplete_jobs=not args.disable_recover_incomplete_jobs,
    )
    retention_config = Stage63RetentionConfig(
        max_persisted_jobs=args.max_persisted_jobs,
        cleanup_on_persist=not args.disable_cleanup_on_persist,
        max_list_limit=args.max_list_limit,
    )
    result_store_config = Stage64ResultStoreConfig(
        result_dir=args.result_dir,
        inline_result_row_limit=args.inline_result_row_limit,
        max_result_page_limit=args.max_result_page_limit,
        delete_result_file_on_job_prune=not args.keep_result_files_on_prune,
    )
    batch_config = Stage65BatchConfig(
        max_batch_submit=args.max_batch_submit,
        max_batch_result_jobs=args.max_batch_result_jobs,
    )
    service = build_stage65_real_service(
        port=args.port,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "stage": "stage65_batch_jobs",
                "status": "starting",
                "port": args.port,
                "queue_config": queue_config.__dict__,
                "job_config": job_config.__dict__,
                "persistence_config": persistence_config.__dict__,
                "retention_config": retention_config.__dict__,
                "result_store_config": result_store_config.__dict__,
                "batch_config": batch_config.__dict__,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
