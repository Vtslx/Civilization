from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder, _terms
from .stage102_trifid_episode_disambiguation import TrifidEpisodeDisambiguator
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage103_trifid_episode_temporal_index")


class TrifidTemporalEpisodeRetriever:
    """Retrieves Stage102-safe episode candidates inside a deterministic time window."""

    def __init__(self) -> None:
        self._disambiguator = TrifidEpisodeDisambiguator()

    def retrieve(
        self,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        time_index: float,
        window_seconds: float,
        limit: int = 5,
    ) -> list[EpisodeFrame]:
        if window_seconds < 0:
            raise ValueError("window_seconds must not be negative")
        candidates = self._disambiguator.retrieve(binder, query, limit=len(binder.frames))
        window_start, window_end = time_index - window_seconds, time_index + window_seconds
        candidates = [
            frame
            for frame in candidates
            if frame.started_at is not None
            and frame.ended_at is not None
            and frame.started_at <= window_end
            and frame.ended_at >= window_start
        ]
        query_terms = _terms(query)
        candidates.sort(
            key=lambda frame: (
                -len(query_terms & set(frame.cues)),
                abs((frame.time_index or 0.0) - time_index),
                frame.sequence or 0,
                frame.episode_id,
            )
        )
        return candidates[:limit]


def run_stage103_trifid_temporal_index_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    early_a = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    early_b = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=102.0)
    late_a = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=200.0)
    late_b = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=202.0)
    early = binder.bind(store, source_cell_ids=[early_a.cell_id, early_b.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
    late = binder.bind(store, source_cell_ids=[late_a.cell_id, late_b.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
    rejected_discontinuous = False
    try:
        binder.bind(store, source_cell_ids=[early_a.cell_id, late_a.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="invalid", max_source_gap_seconds=5.0)
    except ValueError:
        rejected_discontinuous = True
    retriever = TrifidTemporalEpisodeRetriever()
    early_hit = retriever.retrieve(binder, "robot laboratory", time_index=101.0, window_seconds=5.0)
    late_hit = retriever.retrieve(binder, "robot laboratory", time_index=201.0, window_seconds=5.0)
    gap_hit = retriever.retrieve(binder, "robot laboratory", time_index=150.0, window_seconds=5.0)
    summary = {
        "stage": "stage103_trifid_episode_temporal_index",
        "episodes": [asdict(early), asdict(late)],
        "stage_gates": {
            "time_bounds_aggregated": early.started_at == 100.0 and early.ended_at == 102.0 and early.time_index == 101.0,
            "early_episode_selected": [frame.episode_id for frame in early_hit] == [early.episode_id],
            "late_episode_selected": [frame.episode_id for frame in late_hit] == [late.episode_id],
            "gap_query_rejected": gap_hit == [],
            "discontinuous_sources_rejected": rejected_discontinuous,
            "sources_preserved": all(cell_id in store.cells for cell_id in (early_a.cell_id, early_b.cell_id, late_a.cell_id, late_b.cell_id)),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
