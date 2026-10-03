from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage103_trifid_episode_temporal_index import TrifidTemporalEpisodeRetriever
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage121_trifid_stage74_bridge")


class TrifidStage74Bridge:
    """Adapts existing Stage74 episodic write traces into per-session Trifid frames."""

    def __init__(self) -> None:
        self._binders: dict[str, TrifidEpisodeBinder] = {}
        self._retriever = TrifidTemporalEpisodeRetriever()

    def bind_request(self, store: OrionMemoryStore, *, session_id: str, trace: dict[str, Any]) -> dict[str, Any] | None:
        episodic_id = trace.get("episodic_cell_id")
        if not isinstance(episodic_id, str) or episodic_id not in store.cells:
            return None
        cell = store.cells[episodic_id]
        if cell.memory_system != MemorySystem.EPISODIC or cell.decay_state != "active":
            return None
        task_name = str(cell.metadata.get("task_name", "custom"))
        outcome = str(trace.get("outcome", cell.metadata.get("outcome", "unknown")))
        binder = self._binders.setdefault(session_id, TrifidEpisodeBinder())
        frame = binder.bind(
            store,
            source_cell_ids=[episodic_id],
            scene="stage74_request",
            entities=[session_id],
            goal=task_name,
            action="frozen_inference",
            outcome=outcome,
        )
        return {"session_id": session_id, "episode_id": frame.episode_id, "anchor_cell_id": frame.anchor_cell_id}

    def recall(self, *, session_id: str, query: str, time_index: float, window_seconds: float) -> dict[str, Any]:
        binder = self._binders.get(session_id)
        if binder is None:
            return {"session_id": session_id, "episode_ids": []}
        frames = self._retriever.retrieve(binder, query, time_index=time_index, window_seconds=window_seconds)
        return {"session_id": session_id, "episode_ids": [frame.episode_id for frame in frames]}


def run_stage121_trifid_stage74_bridge_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store = OrionMemoryStore(); bridge = TrifidStage74Bridge()
    episodic = store.write_cell(memory_system="episodic", content="task=classify; outcome=success", summary="classify inference success", source="stage74_session:alpha", time_index=100.0, metadata={"task_name": "classify", "outcome": "success"})
    bound = bridge.bind_request(store, session_id="alpha", trace={"episodic_cell_id": episodic.cell_id, "outcome": "success"})
    recalled = bridge.recall(session_id="alpha", query="alpha classify", time_index=100.0, window_seconds=1.0)
    isolated = bridge.recall(session_id="beta", query="beta classify", time_index=100.0, window_seconds=1.0)
    summary = {"stage": "stage121_trifid_stage74_bridge", "stage_gates": {"stage74_trace_bound": bound is not None and bound["anchor_cell_id"] in store.cells, "session_recall_works": recalled["episode_ids"] == [bound["episode_id"]] if bound else False, "session_isolation": isolated["episode_ids"] == [], "source_preserved": episodic.cell_id in store.cells}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
