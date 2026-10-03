from __future__ import annotations

import json

from .surface_flip_benchmark import run_qwen3_surface_flip_repair


def main() -> None:
    print(json.dumps(run_qwen3_surface_flip_repair(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
