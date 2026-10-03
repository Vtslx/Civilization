from __future__ import annotations

import json

from .real_task_migration import run_qwen3_real_task_migration


if __name__ == "__main__":
    print(json.dumps(run_qwen3_real_task_migration(), ensure_ascii=False, indent=2))

