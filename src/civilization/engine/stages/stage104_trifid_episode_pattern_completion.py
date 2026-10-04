from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage103_trifid_episode_temporal_index import TrifidTemporalEpisodeRetriever
from .stage73_orion_memory_kernel import MemoryCell, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage104_trifid_episode_pattern_completion")


@dataclass(frozen=True)
class EpisodeCompletion:
    episode: EpisodeFrame
    anchor_cell: MemoryCell
    source_cells: tuple[MemoryCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": asdict(self.episode),
            "anchor_cell": self.anchor_cell.to_dict(),
            "source_cells": [cell.to_dict() for cell in self.source_cells],
        }


class TrifidEpisodePatternCompleter:
    """Completes one time-safe episode into its anchor and active source cells."""

    def __init__(self) -> None:
        self._temporal_retriever = TrifidTemporalEpisodeRetriever()

    def complete(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        time_index: float,
        window_seconds: float,
    ) -> EpisodeCompletion | None:
        matches = self._temporal_retriever.retrieve(
            binder,
            query,
            time_index=time_index,
            window_seconds=window_seconds,
            limit=1,
        )
        if not matches:
            return None
        return self.complete_frame(
            store,
            matches[0],
            query=query,
            time_index=time_index,
            window_seconds=window_seconds,
        )

    @staticmethod
    def complete_frame(
        store: OrionMemoryStore,
        episode: EpisodeFrame,
        *,
        query: str,
        time_index: float,
        window_seconds: float,
    ) -> EpisodeCompletion | None:
        cells = store.read_cells(
            [episode.anchor_cell_id, *episode.source_cell_ids],
            trace_details={
                "trifid_pattern_completion": True,
                "episode_id": episode.episode_id,
                "query": query,
                "time_index": time_index,
                "window_seconds": window_seconds,
            },
        )
        if len(cells) != len(episode.source_cell_ids) + 1:
            return None
        return EpisodeCompletion(
            episode=episode,
            anchor_cell=cells[0],
            source_cells=tuple(cells[1:]),
        )


def run_stage104_trifid_pattern_completion_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=102.0, ttl_seconds=60.0)
    episode = binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
    completer = TrifidEpisodePatternCompleter()
    completion = completer.complete(store, binder, "robot laboratory", time_index=101.0, window_seconds=5.0)
    outside_window = completer.complete(store, binder, "robot laboratory", time_index=150.0, window_seconds=5.0)
    read_traces = [
        event
        for event in store.trace_events
        if event.details.get("trifid_pattern_completion")
    ]
    summary = {
        "stage": "stage104_trifid_episode_pattern_completion",
        "completion": completion.to_dict() if completion else None,
        "stage_gates": {
            "complete_episode_recalled": completion is not None and completion.episode.episode_id == episode.episode_id,
            "anchor_recalled": completion is not None and completion.anchor_cell.cell_id == episode.anchor_cell_id,
            "source_order_preserved": completion is not None and [cell.cell_id for cell in completion.source_cells] == [arrival.cell_id, collection.cell_id],
            "outside_window_rejected": outside_window is None,
            "completion_read_traced": len(read_traces) == 1 and read_traces[0].cell_ids == (episode.anchor_cell_id, arrival.cell_id, collection.cell_id),
            "source_cells_preserved": arrival.cell_id in store.cells and collection.cell_id in store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
