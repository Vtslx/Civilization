from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationCluster, LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidate, LagoonSchemaCandidateBuilder
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage124_lagoon_approved_consolidation")


@dataclass(frozen=True)
class LagoonConsolidationResult:
    semantic_cell_id: str
    fingerprint: str
    created: bool

    def to_dict(self) -> dict:
        return asdict(self)


class LagoonApprovedConsolidator:
    def consolidate(self, store: OrionMemoryStore, cluster: LagoonConsolidationCluster, candidate: LagoonSchemaCandidate, *, approval_id: str) -> LagoonConsolidationResult:
        if not approval_id.strip() or candidate.fingerprint != "lagoon:" + "|".join(cluster.episode_ids):
            raise ValueError("invalid approval or candidate")
        existing = next((cell for cell in store.cells.values() if cell.memory_system == MemorySystem.SEMANTIC and cell.metadata.get("lagoon_fingerprint") == candidate.fingerprint), None)
        if existing:
            return LagoonConsolidationResult(existing.cell_id, candidate.fingerprint, False)
        semantic = store.consolidate_episodic_to_semantic(list(cluster.anchor_cell_ids), summary=candidate.summary, content=" | ".join(candidate.support_episode_ids), source="lagoon_stage124", confidence=candidate.confidence, importance=candidate.importance, metadata={"lagoon_fingerprint": candidate.fingerprint, "lagoon_support_episode_ids": list(candidate.support_episode_ids), "lagoon_signature": [candidate.signature[0], list(candidate.signature[1]), candidate.signature[2], candidate.signature[3], candidate.signature[4]], "lagoon_schema_outcome": candidate.signature[4], "lagoon_approval_id": approval_id})
        return LagoonConsolidationResult(semantic.cell_id, candidate.fingerprint, True)


def run_stage124_lagoon_approved_consolidation_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder(); sources = []
    for stamp in (100.0, 130.0):
        source = store.write_cell(memory_system="episodic", content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
        frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
        completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
        assert completion; TrifidEpisodeReplayer.record_completion(store, completion); sources.append(source.cell_id)
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]; consolidator = LagoonApprovedConsolidator(); rejected = False
    try: consolidator.consolidate(store, cluster, candidate, approval_id="")
    except ValueError: rejected = True
    first = consolidator.consolidate(store, cluster, candidate, approval_id="stage124"); second = consolidator.consolidate(store, cluster, candidate, approval_id="again")
    summary = {"stage": "stage124_lagoon_approved_consolidation", "stage_gates": {"approval_required": rejected, "semantic_created": first.created and first.semantic_cell_id in store.cells, "provenance_preserved": store.cells[first.semantic_cell_id].metadata.get("lagoon_support_episode_ids") == list(candidate.support_episode_ids), "idempotent": not second.created and second.semantic_cell_id == first.semantic_cell_id, "sources_preserved": all(item in store.cells for item in sources)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); store.write_artifacts(output_dir, summary=summary); return summary
