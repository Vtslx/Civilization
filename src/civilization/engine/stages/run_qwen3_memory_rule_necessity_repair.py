from __future__ import annotations

import json

from .memory_rule_necessity_benchmark import run_qwen3_memory_rule_necessity_repair


if __name__ == "__main__":
    print(json.dumps(run_qwen3_memory_rule_necessity_repair(), ensure_ascii=False, indent=2))
