from __future__ import annotations

import argparse
import json

from .stage59_access_control_audit import (
    DEFAULT_AUDIT_LOG,
    DEFAULT_OUTPUT_DIR,
    Stage59SecurityConfig,
    build_stage59_real_service,
    run_stage59_security_smoke,
)
from experiments.civilization_transformer_qwen3.model_paths import DEFAULT_MODEL_PATH


def _hosts(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage59 access-controlled service")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allowed-hosts", default="127.0.0.1,::1,localhost")
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--require-token-for-predict", action="store_true")
    parser.add_argument("--require-token-for-admin", action="store_true", default=True)
    parser.add_argument("--no-token-for-admin", action="store_false", dest="require_token_for_admin")
    parser.add_argument("--require-token-for-metrics", action="store_true")
    parser.add_argument("--audit-log-path", default=str(DEFAULT_AUDIT_LOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--package-manifest", default="experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
    parser.add_argument(
        "--centroid-bundle",
        default=(
            "experiments/civilization_transformer_qwen3/artifacts/"
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
        summary = run_stage59_security_smoke(output_dir=args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    security = Stage59SecurityConfig(
        allowed_hosts=_hosts(args.allowed_hosts),
        bearer_token=args.bearer_token,
        require_token_for_predict=args.require_token_for_predict,
        require_token_for_admin=args.require_token_for_admin,
        require_token_for_metrics=args.require_token_for_metrics,
        audit_log_path=args.audit_log_path,
    )
    service = build_stage59_real_service(
        port=args.port,
        security=security,
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "stage": "stage59_access_control_audit",
                "status": "starting",
                "host": args.host,
                "port": args.port,
                "bearer_token_configured": args.bearer_token is not None,
                "audit_log_path": args.audit_log_path,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
