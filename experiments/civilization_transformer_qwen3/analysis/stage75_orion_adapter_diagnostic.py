from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch

from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from .stage74_orion_memory_service import build_stage74_real_service
from .stage75_orion_adapter_context_bridge import OrionAdapterContextBridge


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage75_orion_adapter_diagnostic")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:  # noqa: S310 - local WSL diagnostic client
        return json.loads(response.read().decode("utf-8"))


def run_stage75_orion_adapter_diagnostic(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = "/home/yike/AoNeb-01/Models/Qwen3-0.6B",
    preferred_device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    query = "route the Orion semantic memory through the frozen adapter"
    store = OrionMemoryStore()
    seed = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="The semantic evidence supports approving the operation.",
        summary="Stage75 semantic retrieval seed",
        source="stage75_diagnostic",
    )
    bridge = OrionAdapterContextBridge(max_items=4, max_item_chars=512)
    items = bridge.items_for_results(store.read(query, memory_system=MemorySystem.SEMANTIC, limit=4))

    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    encoded = bridge.encode(FrozenQwenContextEncoder(backend), items)
    vector_audit = {
        "shape": list(encoded.vectors.shape),
        "mask_shape": list(encoded.mask.shape),
        "mask_all_true": bool(encoded.mask.all()),
        "vectors_finite": bool(torch.isfinite(encoded.vectors).all()),
        "cell_ids": [item.cell_id for item in encoded.items],
        "items": [item.to_dict() for item in encoded.items],
    }

    service = build_stage74_real_service(port=0, model_path=model_path, preferred_device=preferred_device)
    service.write_memory_cell(
        "stage75-diagnostic",
        {"memory_system": "semantic", "content": seed.content, "summary": seed.summary},
    )
    server = service.start_background()
    host, port = server.server_address
    payload = {
        "id": "stage75-diagnostic",
        "text": "Use the Orion semantic memory to decide the operation.",
        "memory_items": ["caller provided memory"],
        "rule_items": ["keep Qwen3 frozen"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "no_memory_path"],
        "orion_memory": {"session_id": "stage75-diagnostic"},
    }
    try:
        response = _post(f"http://{host}:{port}/v1/predict", payload)
    finally:
        service.shutdown()
    rows = {row["control_mode"]: row for row in response["rows"]}
    full = rows["full"]
    no_memory = rows["no_memory_path"]
    full_traces = full["response"]["trace"]["traces"]
    no_memory_traces = no_memory["response"]["trace"]["traces"]
    full_memory_delta = [float(trace["memory_delta_norm"]) for trace in full_traces.values()]
    no_memory_delta = [float(trace["memory_delta_norm"]) for trace in no_memory_traces.values()]
    summary = {
        "stage": "stage75_orion_adapter_diagnostic",
        "vector_audit": vector_audit,
        "service_rows": {
            mode: {
                "status": row["status"],
                "control_audit_passed": row["control_audit_passed"],
                "retrieved_cell_ids": row["orion_memory"]["retrieved_cell_ids"],
                "qwen_trainable_parameters": row["response"]["trace"]["qwen_trainable_parameters"],
                "qwen_gradients": row["response"]["trace"]["qwen_gradients"],
                "weights_unchanged": row["response"]["trace"]["weights_unchanged"],
            }
            for mode, row in rows.items()
        },
        "memory_delta_norms": {"full": full_memory_delta, "no_memory_path": no_memory_delta},
        "stage_gates": {
            "bridge_cell_provenance": vector_audit["cell_ids"] == [seed.cell_id],
            "adapter_vector_contract": vector_audit["shape"] == [1, 1, 1024] and vector_audit["mask_shape"] == [1, 1] and vector_audit["mask_all_true"] and vector_audit["vectors_finite"],
            "service_retrieves_same_cell": full["orion_memory"]["retrieved_cell_ids"] == ["semantic-000001"],
            "full_memory_path_active": all(value > 0.0 for value in full_memory_delta),
            "no_memory_path_zeroed": all(value == 0.0 for value in no_memory_delta),
            "control_audits_pass": full["control_audit_passed"] and no_memory["control_audit_passed"],
            "qwen_frozen": all(
                row["response"]["trace"]["qwen_trainable_parameters"] == 0
                and row["response"]["trace"]["qwen_gradients"] == 0
                and row["response"]["trace"]["weights_unchanged"]
                for row in rows.values()
            ),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    _write_json(output / "bridge_items.json", vector_audit["items"])
    _write_json(output / "vector_audit.json", vector_audit)
    _write_json(output / "summary.json", summary)
    return summary
