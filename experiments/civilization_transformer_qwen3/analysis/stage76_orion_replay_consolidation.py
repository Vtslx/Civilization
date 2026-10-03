from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage73_orion_memory_kernel import MemoryCell, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage76_orion_replay_consolidation")


@dataclass(frozen=True)
class Stage76ReplayConfig:
    min_episodes: int = 2
    min_success_episodes: int = 1


@dataclass(frozen=True)
class Stage76ReplayResult:
    accepted: bool
    reason: str | None
    source_episode_ids: tuple[str, ...]
    semantic_cell_id: str | None
    procedural_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrionReplayConsolidationPolicy:
    def __init__(self, config: Stage76ReplayConfig = Stage76ReplayConfig()) -> None:
        if config.min_episodes < 1 or config.min_success_episodes < 1:
            raise ValueError("minimum episode thresholds must be >= 1")
        self.config = config

    def apply(self, store: OrionMemoryStore, *, source: str, task_name: str) -> Stage76ReplayResult:
        episodes = [
            cell
            for cell in store.timeline(source=source)
            if cell.consolidation_state != "replayed" and cell.metadata.get("task_name") == task_name
        ]
        episode_ids = tuple(cell.cell_id for cell in episodes)
        if len(episodes) < self.config.min_episodes:
            return Stage76ReplayResult(False, "insufficient_episodes", episode_ids, None, ())
        successes = [cell for cell in episodes if cell.metadata.get("outcome") == "success"]
        if len(successes) < self.config.min_success_episodes:
            return Stage76ReplayResult(False, "insufficient_success_episodes", episode_ids, None, ())
        semantic = store.consolidate_episodic_to_semantic(
            [cell.cell_id for cell in successes],
            summary=f"{task_name} replayed semantic pattern",
            content=" | ".join(cell.content for cell in successes),
            source=source,
            confidence=sum(cell.confidence for cell in successes) / len(successes),
            importance=max(cell.importance for cell in successes),
            metadata={"stage": 76, "task_name": task_name, "replay_episode_ids": [cell.cell_id for cell in successes]},
        )
        procedural_ids: list[str] = []
        for outcome in ("success", "failure"):
            outcome_cells = [cell for cell in episodes if cell.metadata.get("outcome") == outcome]
            if not outcome_cells:
                continue
            procedural = store.write_procedural_from_task_trace(
                task_name=task_name,
                steps=["retrieve episodic outcomes", f"replay {len(outcome_cells)} {outcome} episodes", "record policy candidate"],
                outcome=outcome,
                source=source,
                confidence=sum(cell.confidence for cell in outcome_cells) / len(outcome_cells),
                importance=max(cell.importance for cell in outcome_cells),
                metadata={"stage": 76, "replay_episode_ids": [cell.cell_id for cell in outcome_cells], "episode_count": len(outcome_cells)},
            )
            procedural_ids.append(procedural.cell_id)
        return Stage76ReplayResult(True, None, episode_ids, semantic.cell_id, tuple(procedural_ids))


def run_stage76_orion_replay_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store = OrionMemoryStore()
    source = "stage76_smoke"
    for index, outcome in enumerate(("success", "success", "failure"), start=1):
        store.write_cell(
            memory_system=MemorySystem.EPISODIC,
            content=f"episode {index} outcome={outcome}",
            summary=f"stage76 task {outcome}",
            source=source,
            confidence=0.9 if outcome == "success" else 0.5,
            metadata={"task_name": "stage76_task", "outcome": outcome},
        )
    result = OrionReplayConsolidationPolicy().apply(store, source=source, task_name="stage76_task")
    summary = {
        **store.summary(),
        "replay": result.to_dict(),
        "stage_gates": {
            "accepted": result.accepted,
            "semantic_created": result.semantic_cell_id in store.cells,
            "procedural_outcomes_preserved": len(result.procedural_cell_ids) == 2,
            "source_episodes_preserved": all(cell_id in store.cells for cell_id in result.source_episode_ids),
            "replay_links_created": any(link.link_type.value == "replay" for link in store.links),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
