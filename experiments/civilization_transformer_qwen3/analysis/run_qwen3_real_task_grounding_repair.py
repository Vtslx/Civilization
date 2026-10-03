from __future__ import annotations

import json

from .real_task_grounding_repair import run_qwen3_real_task_grounding_repair


if __name__ == "__main__":
    print(json.dumps(run_qwen3_real_task_grounding_repair(), ensure_ascii=False, indent=2))
