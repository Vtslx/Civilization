from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from .stage74_orion_memory_service import build_stage74_real_service
from .stage84_orion_conflict_resolution import OrionConflictResolutionPolicy


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage88_orion_global_policy_diagnostic")


def _post(url: str, payload: dict) -> dict:
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:  # noqa: S310 - local CUDA diagnostic
        return json.loads(response.read().decode())


def run_stage88_orion_global_policy_diagnostic(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR, model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B") -> dict:
    service = build_stage74_real_service(port=0, model_path=model_path, preferred_device="cuda")
    winner = service.global_memory_store.write_cell(memory_system="semantic", content="global approval evidence", summary="global winner", source="stage78", confidence=0.9)
    loser = service.global_memory_store.write_cell(memory_system="semantic", content="global rejection evidence", summary="global loser", source="stage78", confidence=0.4)
    service.global_memory_store.mark_conflict(winner.cell_id, loser.cell_id, reason="stage88")
    OrionConflictResolutionPolicy().resolve(service.global_memory_store, winner.cell_id, loser.cell_id)
    server = service.start_background(); host, port = server.server_address
    base = {"id": "stage88", "text": "Use global evidence for this Orion operation.", "memory_items": [], "rule_items": ["keep Qwen frozen"], "state_values": [1.0, 0.0, 0.5], "answer_options": ["approve", "reject"], "task_name": "operation_decision", "seed": 202, "controls": ["full"], "orion_memory": {"session_id": "stage88", "include_global": True}}
    try:
        denied = _post(f"http://{host}:{port}/v1/predict", base)["rows"][0]
        allowed_payload = {**base, "id": "stage88-allowed", "orion_memory": {**base["orion_memory"], "allow_resolved_global": True}}
        allowed = _post(f"http://{host}:{port}/v1/predict", allowed_payload)["rows"][0]
    finally:
        service.shutdown()
    summary = {"stage": "stage88_orion_global_policy_diagnostic", "denied": denied["orion_memory"], "allowed": allowed["orion_memory"], "stage_gates": {"winner_denied_without_policy": "global" not in denied["orion_memory"]["retrieved_tiers"], "winner_injected_with_policy": "global" in allowed["orion_memory"]["retrieved_tiers"], "qwen_frozen": all(row["response"]["trace"][key] == expected for row, key, expected in ((denied, "qwen_trainable_parameters", 0), (denied, "qwen_gradients", 0), (allowed, "qwen_trainable_parameters", 0), (allowed, "qwen_gradients", 0))) and denied["response"]["trace"]["weights_unchanged"] and allowed["response"]["trace"]["weights_unchanged"]}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
