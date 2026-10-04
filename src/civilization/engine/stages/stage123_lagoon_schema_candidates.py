from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationCluster, LagoonConsolidationClusterer
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage123_lagoon_schema_candidates")


@dataclass(frozen=True)
class LagoonSchemaCandidate:
    fingerprint: str
    summary: str
    signature: tuple[str, tuple[str, ...], str, str, str]
    support_episode_ids: tuple[str, ...]
    confidence: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonSchemaCandidateBuilder:
    """Abstracts a verified consolidation cluster into an evidence-only schema proposal."""

    def build(self, clusters: list[LagoonConsolidationCluster]) -> list[LagoonSchemaCandidate]:
        candidates = [self._candidate(cluster) for cluster in clusters]
        candidates.sort(key=lambda item: item.fingerprint)
        return candidates

    @staticmethod
    def _candidate(cluster: LagoonConsolidationCluster) -> LagoonSchemaCandidate:
        scene, entities, goal, action, outcome = cluster.signature
        fingerprint = "lagoon:" + "|".join(cluster.episode_ids)
        return LagoonSchemaCandidate(fingerprint, f"{scene} {' '.join(entities)} {goal} {outcome} schema", cluster.signature, cluster.episode_ids, cluster.average_confidence, cluster.average_importance)


def run_stage123_lagoon_schema_candidate_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder(); source_ids: list[str] = []
    for stamp in (100.0, 130.0):
        source = store.write_cell(memory_system="episodic", content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
        frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
        completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
        assert completion is not None; TrifidEpisodeReplayer.record_completion(store, completion); source_ids.append(source.cell_id)
    clusters = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)
    candidates = LagoonSchemaCandidateBuilder().build(clusters)
    summary = {"stage": "stage123_lagoon_schema_candidates", "candidates": [candidate.to_dict() for candidate in candidates], "stage_gates": {"cluster_abstracted": len(candidates) == 1 and len(candidates[0].support_episode_ids) == 2, "fingerprint_stable": len(candidates) == 1 and candidates[0].fingerprint.startswith("lagoon:"), "no_semantic_written": all(cell.memory_system != MemorySystem.SEMANTIC for cell in store.cells.values()), "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); store.write_artifacts(output_dir, summary=summary); return summary
