from __future__ import annotations

import json
from pathlib import Path

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder, _terms
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage102_trifid_episode_disambiguation")


class TrifidEpisodeDisambiguator:
    def retrieve(self, binder: TrifidEpisodeBinder, query: str, *, limit: int = 5) -> list[EpisodeFrame]:
        query_terms = _terms(query)
        required = min(2, len(query_terms))
        ranked = [(len(query_terms & set(frame.cues)), frame) for frame in binder.frames.values()]
        ranked = [item for item in ranked if item[0] >= required]
        ranked.sort(key=lambda item: (-item[0], item[1].episode_id))
        return [frame for _score, frame in ranked[:limit]]


def run_stage102_trifid_disambiguation_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    a = store.write_cell(memory_system="episodic", content="robot lab sample", summary="robot", source="obs")
    b = store.write_cell(memory_system="episodic", content="patient clinic chart", summary="patient", source="obs")
    robot = binder.bind(store, source_cell_ids=[a.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
    patient = binder.bind(store, source_cell_ids=[b.cell_id], scene="clinic", entities=["patient"], goal="review chart", action="inspect", outcome="reviewed")
    disambiguator = TrifidEpisodeDisambiguator()
    robot_hit = disambiguator.retrieve(binder, "robot laboratory")
    patient_hit = disambiguator.retrieve(binder, "patient clinic")
    mixed = disambiguator.retrieve(binder, "robot clinic")
    summary = {"stage": "stage102_trifid_episode_disambiguation", "stage_gates": {"robot_disambiguated": [frame.episode_id for frame in robot_hit] == [robot.episode_id], "patient_disambiguated": [frame.episode_id for frame in patient_hit] == [patient.episode_id], "mixed_cue_rejected": mixed == [], "sources_preserved": a.cell_id in store.cells and b.cell_id in store.cells}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
