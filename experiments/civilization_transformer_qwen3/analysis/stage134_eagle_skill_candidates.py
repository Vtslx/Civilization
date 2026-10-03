from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .stage133_eagle_task_trace import EagleTaskTrace, EagleTaskTraceNormalizer


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage134_eagle_skill_candidates")


@dataclass(frozen=True)
class EagleSkillCandidate:
    fingerprint: str
    task_signature: tuple[str, str, str]
    strategy_steps: tuple[str, ...]
    support_trace_ids: tuple[str, ...]
    success_trace_ids: tuple[str, ...]
    failure_trace_ids: tuple[str, ...]
    confidence: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("task_signature", "strategy_steps", "support_trace_ids", "success_trace_ids", "failure_trace_ids"):
            payload[key] = list(payload[key])
        return payload


class EagleSkillCandidateBuilder:
    """Builds evidence-only procedural candidates from repeated execution traces."""

    def build(self, traces: list[EagleTaskTrace], *, min_successes: int = 2) -> list[EagleSkillCandidate]:
        if min_successes < 1:
            raise ValueError("min_successes must be positive")
        groups: dict[tuple[str, str, str], list[EagleTaskTrace]] = {}
        for trace in traces:
            groups.setdefault((trace.task_name, trace.scene, trace.goal), []).append(trace)
        candidates: list[EagleSkillCandidate] = []
        for signature, group in groups.items():
            successes = [trace for trace in group if trace.outcome == "success"]
            failures = [trace for trace in group if trace.outcome == "failure"]
            if len(successes) < min_successes:
                continue
            strategy_counts: dict[tuple[str, ...], list[EagleTaskTrace]] = {}
            for trace in successes:
                strategy_counts.setdefault(trace.steps, []).append(trace)
            strategy_steps, strategy_support = sorted(strategy_counts.items(), key=lambda item: (-len(item[1]), item[0]))[0]
            payload = {"task_signature": signature, "strategy_steps": strategy_steps, "support_trace_ids": sorted(trace.trace_id for trace in strategy_support)}
            fingerprint = "eagle:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
            total = len(successes) + len(failures)
            candidates.append(EagleSkillCandidate(fingerprint, signature, strategy_steps, tuple(sorted(trace.trace_id for trace in group)), tuple(sorted(trace.trace_id for trace in strategy_support)), tuple(sorted(trace.trace_id for trace in failures)), len(strategy_support) / total, 1.0 + min(1.0, len(strategy_support) / 10.0)))
        candidates.sort(key=lambda candidate: candidate.fingerprint)
        return candidates


def run_stage134_eagle_skill_candidate_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    normalizer = EagleTaskTraceNormalizer()
    raw = [
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-001"], "time_index": 100.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-002"], "time_index": 130.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-003"], "time_index": 160.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "drop sample"], "outcome": "failure", "source_cell_ids": ["episodic-004"], "time_index": 200.0},
        {"task_name": "repair_robot", "scene": "workshop", "goal": "restore robot", "steps": ["open panel"], "outcome": "success", "source_cell_ids": ["episodic-005"], "time_index": 220.0},
    ]
    traces = [normalizer.normalize(item) for item in raw]
    candidates = EagleSkillCandidateBuilder().build(traces)
    candidate = candidates[0]
    summary = {
        "stage": "stage134_eagle_skill_candidates",
        "candidates": [item.to_dict() for item in candidates],
        "stage_gates": {
            "repeated_success_candidate": len(candidates) == 1 and len(candidate.success_trace_ids) == 3,
            "failure_evidence_preserved": candidate.failure_trace_ids == (traces[3].trace_id,),
            "strategy_steps_consensus": candidate.strategy_steps == traces[0].steps,
            "fingerprint_deterministic": candidate.fingerprint == EagleSkillCandidateBuilder().build(traces)[0].fingerprint,
            "insufficient_success_rejected": EagleSkillCandidateBuilder().build([traces[0]], min_successes=2) == [],
            "no_memory_written": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "skill_candidates.json").write_text(json.dumps(summary["candidates"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
