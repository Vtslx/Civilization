from __future__ import annotations

import argparse
import json

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage47_centroid_batch_inference import run_stage47_centroid_batch_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage47 raw centroid bundle and run batch inference")
    parser.add_argument("--package-manifest", default="artifacts/civilization/stage45_adapter_package/package_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/civilization/stage47_centroid_batch_inference")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()
    summary = run_stage47_centroid_batch_smoke(
        package_manifest=args.package_manifest,
        output_dir=args.output_dir,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        seed=args.seed,
        smoke=args.smoke,
        include_external=args.include_external,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
