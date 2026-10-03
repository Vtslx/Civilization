from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .stage122_lagoon_consolidation_clusters import run_stage122_lagoon_cluster_smoke
from .stage123_lagoon_schema_candidates import run_stage123_lagoon_schema_candidate_smoke
from .stage124_lagoon_approved_consolidation import run_stage124_lagoon_approved_consolidation_smoke
from .stage125_lagoon_schema_retrieval import run_stage125_lagoon_schema_retrieval_smoke
from .stage126_lagoon_provenance_query import run_stage126_lagoon_provenance_query_smoke
from .stage127_lagoon_schema_conflict_guard import run_stage127_lagoon_conflict_smoke
from .stage128_lagoon_conflict_aware_recall import run_stage128_lagoon_conflict_aware_recall_smoke
from .stage129_lagoon_conflict_review import run_stage129_lagoon_conflict_review_smoke
from .stage130_lagoon_conflict_decision import run_stage130_lagoon_conflict_decision_smoke
from .stage131_lagoon_decision_aware_recall import run_stage131_lagoon_decision_aware_recall_smoke


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage132_lagoon_release_gate")


_STAGE_RUNNERS: tuple[tuple[str, Callable[..., dict]], ...] = (
    ("stage122", run_stage122_lagoon_cluster_smoke),
    ("stage123", run_stage123_lagoon_schema_candidate_smoke),
    ("stage124", run_stage124_lagoon_approved_consolidation_smoke),
    ("stage125", run_stage125_lagoon_schema_retrieval_smoke),
    ("stage126", run_stage126_lagoon_provenance_query_smoke),
    ("stage127", run_stage127_lagoon_conflict_smoke),
    ("stage128", run_stage128_lagoon_conflict_aware_recall_smoke),
    ("stage129", run_stage129_lagoon_conflict_review_smoke),
    ("stage130", run_stage130_lagoon_conflict_decision_smoke),
    ("stage131", run_stage131_lagoon_decision_aware_recall_smoke),
)


def run_stage132_lagoon_release_gate(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stage_summaries: dict[str, dict] = {}
    for stage_name, runner in _STAGE_RUNNERS:
        stage_summaries[stage_name] = runner(output_dir=output / stage_name)
    stage_gates = {
        "all_lagoon_stages_pass": all(summary.get("passes_stage_gate") is True for summary in stage_summaries.values()),
        "all_stage_summaries_present": all((output / stage_name / "summary.json").exists() for stage_name, _runner in _STAGE_RUNNERS),
        "all_core_artifacts_present": all(
            all((output / stage_name / filename).exists() for filename in ("memory_cells.json", "memory_links.json", "trace_events.jsonl"))
            for stage_name, _runner in _STAGE_RUNNERS
        ),
        "release_stage_count_complete": len(stage_summaries) == 10 and set(stage_summaries) == {f"stage{number}" for number in range(122, 132)},
        "facts_decisions_and_reads_covered": all(
            any(key in summary.get("stage_gates", {}) for key in expected)
            for summary, expected in (
                (stage_summaries["stage122"], ("no_semantic_written",)),
                (stage_summaries["stage124"], ("semantic_created",)),
                (stage_summaries["stage127"], ("conflict_traced",)),
                (stage_summaries["stage130"], ("procedural_decision_created",)),
                (stage_summaries["stage131"], ("recall_is_traced",)),
            )
        ),
    }
    summary = {
        "stage": "stage132_lagoon_release_gate",
        "version": "v0.00.04",
        "stage_summaries": stage_summaries,
        "stage_gates": stage_gates,
    }
    summary["passes_stage_gate"] = all(stage_gates.values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
