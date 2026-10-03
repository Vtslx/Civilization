from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage72_resumable_download_client import (
    DEFAULT_OUTPUT_DIR,
    Stage72DownloadConfig,
    Stage72ResumableDownloadClient,
    run_stage72_resumable_download_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage72 resumable package download client")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download-url", default=None)
    parser.add_argument("--package-url", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--expected-size-bytes", type=int, default=None)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if args.smoke:
        summary = run_stage72_resumable_download_smoke(output_dir=args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.download_url or not args.output_path:
        parser.error("--download-url and --output-path are required unless --smoke is set")

    client = Stage72ResumableDownloadClient(
        Stage72DownloadConfig(
            chunk_bytes=args.chunk_bytes,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
    )
    result = client.download(
        download_url=args.download_url,
        package_url=args.package_url,
        output_path=Path(args.output_path),
        expected_sha256=args.expected_sha256,
        expected_size_bytes=args.expected_size_bytes,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
