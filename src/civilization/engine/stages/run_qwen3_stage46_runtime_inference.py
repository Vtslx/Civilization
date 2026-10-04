from __future__ import annotations

import argparse
import json

from civilization.engine.model_paths import DEFAULT_MODEL_PATH

from .stage46_runtime_inference import run_stage46_runtime_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage46 single-sample Qwen3 Civilization runtime inference smoke")
    parser.add_argument("--package-manifest", default="artifacts/civilization/stage45_adapter_package/package_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/civilization/stage46_runtime_inference")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()
    summary = run_stage46_runtime_smoke(
        package_manifest=args.package_manifest,
        output_dir=args.output_dir,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
