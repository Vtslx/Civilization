from __future__ import annotations

import json

from .adapter_benchmark import run_qwen3_civilization_adapter


def main() -> None:
    summary = run_qwen3_civilization_adapter()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
