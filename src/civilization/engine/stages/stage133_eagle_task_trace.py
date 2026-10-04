from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage133_eagle_task_trace")


@dataclass(frozen=True)
class EagleTaskTrace:
    trace_id: str
    task_name: str
    scene: str
    goal: str
    steps: tuple[str, ...]
    outcome: str
    source_cell_ids: tuple[str, ...]
    time_index: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = list(self.steps)
        payload["source_cell_ids"] = list(self.source_cell_ids)
        return payload


class EagleTaskTraceNormalizer:
    """Normalizes execution traces into deterministic, evidence-only task records."""

    def normalize(self, payload: dict[str, Any]) -> EagleTaskTrace:
        task_name = self._required_text(payload, "task_name")
        scene = self._required_text(payload, "scene")
        goal = self._required_text(payload, "goal")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps or any(not isinstance(step, str) or not step.strip() for step in raw_steps):
            raise ValueError("steps must be a non-empty list of non-empty strings")
        outcome = payload.get("outcome")
        if outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        raw_sources = payload.get("source_cell_ids")
        if not isinstance(raw_sources, list) or not raw_sources or any(not isinstance(item, str) or not item.strip() for item in raw_sources):
            raise ValueError("source_cell_ids must be a non-empty list")
        try:
            time_index = float(payload["time_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("time_index must be numeric") from error
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        trace = EagleTaskTrace(
            trace_id="",
            task_name=task_name,
            scene=scene,
            goal=goal,
            steps=tuple(step.strip() for step in raw_steps),
            outcome=outcome,
            source_cell_ids=tuple(raw_sources),
            time_index=time_index,
            metadata=dict(metadata),
        )
        canonical = json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return EagleTaskTrace(f"eagle-trace-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}", trace.task_name, trace.scene, trace.goal, trace.steps, trace.outcome, trace.source_cell_ids, trace.time_index, trace.metadata)

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be non-empty text")
        return value.strip()


def run_stage133_eagle_task_trace_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    normalizer = EagleTaskTraceNormalizer()
    raw = [
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-001"], "time_index": 100.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-002"], "time_index": 130.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-003"], "time_index": 160.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "drop sample"], "outcome": "failure", "source_cell_ids": ["episodic-004"], "time_index": 200.0},
    ]
    traces = [normalizer.normalize(item) for item in raw]
    rejected = 0
    for invalid in ({**raw[0], "steps": []}, {**raw[0], "outcome": "unknown"}, {**raw[0], "source_cell_ids": []}):
        try:
            normalizer.normalize(invalid)
        except ValueError:
            rejected += 1
    summary = {
        "stage": "stage133_eagle_task_trace",
        "traces": [trace.to_dict() for trace in traces],
        "stage_gates": {
            "four_traces_normalized": len(traces) == 4,
            "success_failure_preserved": [trace.outcome for trace in traces] == ["success", "success", "success", "failure"],
            "trace_ids_deterministic": traces[0].trace_id == normalizer.normalize(raw[0]).trace_id and len({trace.trace_id for trace in traces}) == 4,
            "invalid_inputs_rejected": rejected == 3,
            "time_and_sources_preserved": traces[0].time_index == 100.0 and traces[0].source_cell_ids == ("episodic-001",),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "task_traces.json").write_text(json.dumps(summary["traces"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
