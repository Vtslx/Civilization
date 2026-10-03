from __future__ import annotations

import json

from .evidence_answer_benchmark import run_qwen3_evidence_answer_alignment


if __name__ == "__main__":
    print(json.dumps(run_qwen3_evidence_answer_alignment(), ensure_ascii=False, indent=2))
