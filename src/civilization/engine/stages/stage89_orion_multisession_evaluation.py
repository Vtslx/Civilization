from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from .stage74_orion_memory_service import build_stage74_real_service
from .stage84_orion_conflict_resolution import OrionConflictResolutionPolicy
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage89_orion_multisession_evaluation")


def _post(url: str, payload: dict) -> dict:
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:  # noqa: S310 - local CUDA evaluation
        return json.loads(response.read().decode())


def run_stage89_orion_multisession_evaluation(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR, model_path: str = str(DEFAULT_MODEL_PATH)) -> dict:
    service = build_stage74_real_service(port=0, model_path=model_path, preferred_device="cuda")
    winner = service.global_memory_store.write_cell(memory_system="semantic", content="shared global approval evidence", summary="shared winner", source="stage78", confidence=0.9)
    loser = service.global_memory_store.write_cell(memory_system="semantic", content="shared global rejection evidence", summary="shared loser", source="stage78", confidence=0.4)
    service.global_memory_store.mark_conflict(winner.cell_id, loser.cell_id, reason="stage89")
    OrionConflictResolutionPolicy().resolve(service.global_memory_store, winner.cell_id, loser.cell_id)
    server = service.start_background(); host, port = server.server_address
    rows = []
    try:
        for session_id in ("stage89-a", "stage89-b"):
            payload = {"id": session_id, "text": "Use shared global evidence for the operation.", "memory_items": [], "rule_items": ["keep Qwen frozen"], "state_values": [1.0, 0.0, 0.5], "answer_options": ["approve", "reject"], "task_name": "operation_decision", "seed": 202, "controls": ["full"], "orion_memory": {"session_id": session_id, "include_global": True, "allow_resolved_global": True}}
            row = _post(f"http://{host}:{port}/v1/predict", payload)["rows"][0]
            rows.append({"session_id": session_id, "tiers": row["orion_memory"]["retrieved_tiers"], "global_cell_ids": [cell_id for cell_id, tier in zip(row["orion_memory"]["retrieved_cell_ids"], row["orion_memory"]["retrieved_tiers"], strict=True) if tier == "global"], "qwen_frozen": row["response"]["trace"]["qwen_trainable_parameters"] == 0 and row["response"]["trace"]["qwen_gradients"] == 0 and row["response"]["trace"]["weights_unchanged"]})
    finally:
        service.shutdown()
    summary = {"stage": "stage89_orion_multisession_evaluation", "rows": rows, "stage_gates": {"both_sessions_inject_shared_winner": all(winner.cell_id in row["global_cell_ids"] for row in rows), "session_ids_isolated": len({row["session_id"] for row in rows}) == 2, "qwen_frozen": all(row["qwen_frozen"] for row in rows)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
