from __future__ import annotations

import json

from .multilayer_adapter_benchmark import run_qwen3_multilayer_adapter


def main() -> None:
    summary = run_qwen3_multilayer_adapter()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
