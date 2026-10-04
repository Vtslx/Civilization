from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import re
import time
from typing import Any


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage73_orion_memory_kernel")


class MemorySystem(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryLinkType(StrEnum):
    TEMPORAL = "temporal"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAUSAL = "causal"
    TASK = "task"
    PROCEDURE = "procedure"
    REPLAY = "replay"
    CONFLICT = "conflict"


class MemoryTraceAction(StrEnum):
    WRITE = "write"
    READ = "read"
    LINK = "link"
    UPDATE = "update"
    EXPIRE = "expire"
    CONSOLIDATE = "consolidate"
    CONFLICT = "conflict"


@dataclass
class MemoryCell:
    cell_id: str
    memory_system: MemorySystem
    content: str
    summary: str
    source: str
    confidence: float
    importance: float
    created_at: float
    updated_at: float
    time_index: float
    decay_state: str = "active"
    consolidation_state: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory_system"] = self.memory_system.value
        return payload


@dataclass(frozen=True)
class MemoryLink:
    source_cell_id: str
    target_cell_id: str
    link_type: MemoryLinkType
    weight: float
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["link_type"] = self.link_type.value
        return payload


@dataclass(frozen=True)
class MemoryTraceEvent:
    event_id: str
    action: MemoryTraceAction
    cell_ids: tuple[str, ...]
    created_at: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["cell_ids"] = list(self.cell_ids)
        return payload


@dataclass(frozen=True)
class MemoryReadResult:
    cell: MemoryCell
    score: float
    matched_terms: tuple[str, ...]
    link_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.to_dict(),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "link_score": self.link_score,
        }


def _coerce_system(memory_system: MemorySystem | str) -> MemorySystem:
    return memory_system if isinstance(memory_system, MemorySystem) else MemorySystem(memory_system)


def _coerce_link_type(link_type: MemoryLinkType | str) -> MemoryLinkType:
    return link_type if isinstance(link_type, MemoryLinkType) else MemoryLinkType(link_type)


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower()) if term}


class OrionMemoryStore:
    def __init__(self, *, now_fn=time.time) -> None:
        self._now_fn = now_fn
        self.cells: dict[str, MemoryCell] = {}
        self.links: list[MemoryLink] = []
        self.trace_events: list[MemoryTraceEvent] = []
        self._next_cell_id = 1
        self._next_event_id = 1

    def _now(self) -> float:
        return float(self._now_fn())

    def _new_cell_id(self, memory_system: MemorySystem) -> str:
        cell_id = f"{memory_system.value}-{self._next_cell_id:06d}"
        self._next_cell_id += 1
        return cell_id

    def _trace(self, action: MemoryTraceAction, cell_ids: tuple[str, ...], details: dict[str, Any] | None = None) -> None:
        event = MemoryTraceEvent(
            event_id=f"trace-{self._next_event_id:06d}",
            action=action,
            cell_ids=cell_ids,
            created_at=self._now(),
            details=details or {},
        )
        self._next_event_id += 1
        self.trace_events.append(event)

    def write_cell(
        self,
        *,
        memory_system: MemorySystem | str,
        content: str,
        summary: str,
        source: str,
        confidence: float = 1.0,
        importance: float = 1.0,
        time_index: float | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        consolidation_state: str = "raw",
    ) -> MemoryCell:
        system = _coerce_system(memory_system)
        if not content.strip():
            raise ValueError("content must not be empty")
        if not summary.strip():
            raise ValueError("summary must not be empty")
        now = self._now()
        cell_metadata = dict(metadata or {})
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be positive")
            cell_metadata["ttl_seconds"] = ttl_seconds
            cell_metadata["expires_at"] = now + ttl_seconds
        cell = MemoryCell(
            cell_id=self._new_cell_id(system),
            memory_system=system,
            content=content,
            summary=summary,
            source=source,
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, importance),
            created_at=now,
            updated_at=now,
            time_index=time_index if time_index is not None else now,
            metadata=cell_metadata,
            consolidation_state=consolidation_state,
        )
        self.cells[cell.cell_id] = cell
        self._trace(MemoryTraceAction.WRITE, (cell.cell_id,), {"memory_system": system.value, "source": source})
        return cell

    def update_cell(
        self,
        cell_id: str,
        *,
        content: str | None = None,
        summary: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        decay_state: str | None = None,
        consolidation_state: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCell:
        cell = self._required_cell(cell_id)
        if content is not None:
            if not content.strip():
                raise ValueError("content must not be empty")
            cell.content = content
        if summary is not None:
            if not summary.strip():
                raise ValueError("summary must not be empty")
            cell.summary = summary
        if confidence is not None:
            cell.confidence = max(0.0, min(1.0, confidence))
        if importance is not None:
            cell.importance = max(0.0, importance)
        if decay_state is not None:
            cell.decay_state = decay_state
        if consolidation_state is not None:
            cell.consolidation_state = consolidation_state
        if metadata:
            cell.metadata.update(metadata)
        cell.updated_at = self._now()
        self._trace(MemoryTraceAction.UPDATE, (cell_id,), {"decay_state": cell.decay_state, "consolidation_state": cell.consolidation_state})
        return cell

    def link_cells(
        self,
        source_cell_id: str,
        target_cell_id: str,
        *,
        link_type: MemoryLinkType | str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryLink:
        self._required_cell(source_cell_id)
        self._required_cell(target_cell_id)
        typed_link = _coerce_link_type(link_type)
        link = MemoryLink(
            source_cell_id=source_cell_id,
            target_cell_id=target_cell_id,
            link_type=typed_link,
            weight=max(0.0, weight),
            created_at=self._now(),
            metadata=dict(metadata or {}),
        )
        self.links.append(link)
        action = MemoryTraceAction.CONFLICT if typed_link == MemoryLinkType.CONFLICT else MemoryTraceAction.LINK
        self._trace(action, (source_cell_id, target_cell_id), {"link_type": typed_link.value, "weight": link.weight})
        return link

    def mark_conflict(self, source_cell_id: str, target_cell_id: str, *, reason: str, weight: float = 1.0) -> MemoryLink:
        return self.link_cells(
            source_cell_id,
            target_cell_id,
            link_type=MemoryLinkType.CONFLICT,
            weight=weight,
            metadata={"reason": reason, "preserves_original_cells": True},
        )

    def expire_working_memory(self) -> list[MemoryCell]:
        now = self._now()
        expired: list[MemoryCell] = []
        for cell in self.cells.values():
            if cell.memory_system != MemorySystem.WORKING or cell.decay_state != "active":
                continue
            expires_at = cell.metadata.get("expires_at")
            if isinstance(expires_at, (int, float)) and expires_at <= now:
                cell.decay_state = "expired"
                cell.updated_at = now
                cell.metadata["transfer_candidate"] = True
                expired.append(cell)
                self._trace(MemoryTraceAction.EXPIRE, (cell.cell_id,), {"transfer_candidate": True, "expired_at": now})
        return expired

    def read(
        self,
        query: str,
        *,
        memory_system: MemorySystem | str | None = None,
        source: str | None = None,
        include_expired: bool = False,
        limit: int = 5,
    ) -> list[MemoryReadResult]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.expire_working_memory()
        system = _coerce_system(memory_system) if memory_system is not None else None
        query_terms = _terms(query)
        results: list[MemoryReadResult] = []
        for cell in self.cells.values():
            if system is not None and cell.memory_system != system:
                continue
            if source is not None and cell.source != source:
                continue
            if not include_expired and cell.decay_state != "active":
                continue
            cell_terms = _terms(f"{cell.summary} {cell.content} {cell.source}")
            matched = tuple(sorted(query_terms & cell_terms))
            link_score = self._link_score(cell.cell_id)
            lexical = len(matched) / max(len(query_terms), 1)
            system_score = 0.15 if system is not None and cell.memory_system == system else 0.0
            score = lexical + system_score + 0.2 * cell.importance + 0.2 * cell.confidence + link_score
            if not query_terms or matched or system is not None or source is not None:
                results.append(MemoryReadResult(cell=cell, score=score, matched_terms=matched, link_score=link_score))
        results.sort(key=lambda item: (-item.score, item.cell.time_index, item.cell.cell_id))
        selected = results[:limit]
        self._trace(
            MemoryTraceAction.READ,
            tuple(result.cell.cell_id for result in selected),
            {"query": query, "memory_system": system.value if system else None, "source": source, "result_count": len(selected)},
        )
        return selected

    def read_cells(
        self,
        cell_ids: list[str] | tuple[str, ...],
        *,
        include_expired: bool = False,
        trace_details: dict[str, Any] | None = None,
    ) -> list[MemoryCell]:
        """Read known cells in caller order and emit one auditable read event."""
        self.expire_working_memory()
        requested_cell_ids = tuple(cell_ids)
        selected = [
            self.cells[cell_id]
            for cell_id in requested_cell_ids
            if cell_id in self.cells
            and (include_expired or self.cells[cell_id].decay_state == "active")
        ]
        self._trace(
            MemoryTraceAction.READ,
            tuple(cell.cell_id for cell in selected),
            {
                "read_by_id": True,
                "requested_cell_ids": list(requested_cell_ids),
                "include_expired": include_expired,
                **(trace_details or {}),
            },
        )
        return selected

    def timeline(self, *, source: str | None = None, include_expired: bool = False) -> list[MemoryCell]:
        self.expire_working_memory()
        cells = [
            cell
            for cell in self.cells.values()
            if cell.memory_system == MemorySystem.EPISODIC
            and (source is None or cell.source == source)
            and (include_expired or cell.decay_state == "active")
        ]
        cells.sort(key=lambda cell: (cell.time_index, cell.cell_id))
        self._trace(MemoryTraceAction.READ, tuple(cell.cell_id for cell in cells), {"timeline": True, "source": source})
        return cells

    def consolidate_episodic_to_semantic(
        self,
        episodic_cell_ids: list[str],
        *,
        summary: str,
        content: str,
        source: str = "stage73_consolidation",
        confidence: float = 0.75,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCell:
        if not episodic_cell_ids:
            raise ValueError("episodic_cell_ids must not be empty")
        for cell_id in episodic_cell_ids:
            cell = self._required_cell(cell_id)
            if cell.memory_system != MemorySystem.EPISODIC:
                raise ValueError("consolidation input must be episodic cells")
        semantic = self.write_cell(
            memory_system=MemorySystem.SEMANTIC,
            content=content,
            summary=summary,
            source=source,
            confidence=confidence,
            importance=importance,
            metadata={**(metadata or {}), "consolidated_from": list(episodic_cell_ids)},
            consolidation_state="consolidated",
        )
        for cell_id in episodic_cell_ids:
            self.link_cells(cell_id, semantic.cell_id, link_type=MemoryLinkType.REPLAY, weight=1.0, metadata={"consolidation": True})
            self.update_cell(cell_id, consolidation_state="replayed")
        self._trace(MemoryTraceAction.CONSOLIDATE, tuple([*episodic_cell_ids, semantic.cell_id]), {"semantic_cell_id": semantic.cell_id})
        return semantic

    def write_procedural_from_task_trace(
        self,
        *,
        task_name: str,
        steps: list[str],
        outcome: str,
        source: str,
        confidence: float = 0.8,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCell:
        if outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        if not steps:
            raise ValueError("steps must not be empty")
        cell = self.write_cell(
            memory_system=MemorySystem.PROCEDURAL,
            content=" -> ".join(steps),
            summary=f"{task_name} {outcome} procedure",
            source=source,
            confidence=confidence,
            importance=importance,
            metadata={**(metadata or {}), "task_name": task_name, "outcome": outcome, "steps": list(steps)},
            consolidation_state="skill_candidate",
        )
        self._trace(MemoryTraceAction.CONSOLIDATE, (cell.cell_id,), {"procedural_from_task_trace": True, "outcome": outcome})
        return cell

    def summary(self) -> dict[str, Any]:
        counts = {system.value: 0 for system in MemorySystem}
        active_counts = {system.value: 0 for system in MemorySystem}
        for cell in self.cells.values():
            counts[cell.memory_system.value] += 1
            if cell.decay_state == "active":
                active_counts[cell.memory_system.value] += 1
        return {
            "stage": "stage73_orion_memory_kernel",
            "cell_counts": counts,
            "active_cell_counts": active_counts,
            "link_count": len(self.links),
            "trace_count": len(self.trace_events),
            "trace_actions": sorted({event.action.value for event in self.trace_events}),
            "link_types": sorted({link.link_type.value for link in self.links}),
        }

    def write_artifacts(self, output_dir: str | Path, *, summary: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "memory_cells.json").write_text(
            json.dumps([cell.to_dict() for cell in self.cells.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "memory_links.json").write_text(
            json.dumps([link.to_dict() for link in self.links], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output / "trace_events.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.trace_events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        (output / "summary.json").write_text(json.dumps(summary or self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _required_cell(self, cell_id: str) -> MemoryCell:
        try:
            return self.cells[cell_id]
        except KeyError as error:
            raise KeyError(f"unknown memory cell: {cell_id}") from error

    def _link_score(self, cell_id: str) -> float:
        return 0.05 * sum(link.weight for link in self.links if link.source_cell_id == cell_id or link.target_cell_id == cell_id)


def run_stage73_orion_memory_kernel_smoke(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    clock = {"now": 1000.0}
    store = OrionMemoryStore(now_fn=lambda: clock["now"])
    working = store.write_cell(
        memory_system=MemorySystem.WORKING,
        content="Current task is to route Orion memory through a four-system kernel.",
        summary="current Orion stage focus",
        source="stage73_smoke",
        ttl_seconds=5.0,
    )
    episodic_a = store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="Stage73 began by separating working, episodic, semantic, and procedural memory.",
        summary="stage73 kickoff event",
        source="stage73_smoke",
        time_index=1001.0,
    )
    episodic_b = store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="The first implementation keeps Qwen3 frozen and avoids service API changes.",
        summary="stage73 constraint event",
        source="stage73_smoke",
        time_index=1002.0,
    )
    store.link_cells(episodic_a.cell_id, episodic_b.cell_id, link_type=MemoryLinkType.TEMPORAL, weight=0.8)
    semantic = store.consolidate_episodic_to_semantic(
        [episodic_a.cell_id, episodic_b.cell_id],
        summary="Orion Stage73 kernel principle",
        content="The memory kernel must be auditable before service or adapter integration.",
    )
    procedural = store.write_procedural_from_task_trace(
        task_name="stage73_kernel_smoke",
        steps=["write four systems", "link cells", "consolidate episodes", "verify trace"],
        outcome="success",
        source="stage73_smoke",
    )
    store.link_cells(semantic.cell_id, procedural.cell_id, link_type=MemoryLinkType.PROCEDURE, weight=0.7)
    conflict_left = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="Stage73 should not modify the Stage49-72 inference service.",
        summary="service isolation",
        source="stage73_smoke",
    )
    conflict_right = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="A later stage may attach memory to the Stage49-72 inference service.",
        summary="future service integration",
        source="stage73_smoke",
    )
    store.mark_conflict(conflict_left.cell_id, conflict_right.cell_id, reason="current boundary versus future stage")
    clock["now"] = 1006.0
    expired = store.expire_working_memory()
    active_working_results = store.read("current Orion stage focus", memory_system=MemorySystem.WORKING)
    episodic_timeline = store.timeline(source="stage73_smoke")
    procedural_results = store.read("stage73 success procedure", memory_system=MemorySystem.PROCEDURAL)
    summary = {
        **store.summary(),
        "stage_gates": {
            "four_systems_written": all(count >= 1 for count in store.summary()["cell_counts"].values()),
            "working_expired": len(expired) == 1 and expired[0].cell_id == working.cell_id,
            "expired_working_excluded": active_working_results == [],
            "episodic_timeline_ordered": [cell.cell_id for cell in episodic_timeline] == [episodic_a.cell_id, episodic_b.cell_id],
            "semantic_consolidated": semantic.memory_system == MemorySystem.SEMANTIC and semantic.consolidation_state == "consolidated",
            "procedural_task_trace_written": procedural_results and procedural_results[0].cell.metadata.get("outcome") == "success",
            "conflict_preserves_cells": conflict_left.cell_id in store.cells and conflict_right.cell_id in store.cells,
            "trace_covers_required_actions": {
                action.value for action in MemoryTraceAction
            }.issubset({event.action.value for event in store.trace_events}),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
