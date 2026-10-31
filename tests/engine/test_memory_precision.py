from __future__ import annotations

import json
from pathlib import Path

import pytest

from civilization.cli import main
from civilization.engine.memory.precision import (
    MISSING_PROVENANCE,
    ROLE_VIOLATION,
    SUPPORTED,
    UNRESOLVED_DERIVATION,
    UNSUPPORTED,
    MemoryPrecisionError,
    MemoryPrecisionPolicy,
    TranscriptSpan,
    assert_memory_precision,
    audit_memory_cells,
    load_transcript,
)
from civilization.engine.persistence.sink import PersistenceConfig
from civilization.engine.persistence.store import SessionPersistence
from civilization.engine.stages.stage73_orion_memory_kernel import MemoryCell, MemorySystem


def _cell(
    cell_id: str,
    content: str,
    *,
    metadata: dict | None = None,
    source: str = "session",
) -> MemoryCell:
    return MemoryCell(
        cell_id=cell_id,
        memory_system=MemorySystem.EPISODIC,
        content=content,
        summary=content,
        source=source,
        confidence=1.0,
        importance=1.0,
        created_at=1.0,
        updated_at=1.0,
        time_index=1.0,
        metadata=dict(metadata or {}),
    )


def _provenance(source_id: str, quote: str, role: str = "user") -> dict:
    return {"provenance": {"source_id": source_id, "quote": quote, "role": role}}


SPANS = (
    TranscriptSpan(source_id="turn-1", role="user", text="I bought two blue chairs on 2023-03-04."),
    TranscriptSpan(source_id="turn-2", role="assistant", text="You could consider a matching table."),
    TranscriptSpan(source_id="turn-3", role="user", text="My sister moved to Lisbon last spring."),
)


def test_supported_cell_is_precise() -> None:
    report = audit_memory_cells(
        [_cell("episodic-000001", "two blue chairs", metadata=_provenance("turn-1", "two blue chairs"))],
        SPANS,
    )

    assert report.precision == 1.0
    assert report.supported_cells == 1
    assert report.target_met is True
    assert report.findings == ()


def test_fabricated_quote_is_unsupported() -> None:
    report = audit_memory_cells(
        [_cell("episodic-000001", "two red chairs", metadata=_provenance("turn-1", "two red chairs"))],
        SPANS,
    )

    assert report.precision == 0.0
    assert report.status_counts == {UNSUPPORTED: 1}
    assert report.findings[0].status == UNSUPPORTED


def test_missing_source_span_is_unsupported() -> None:
    report = audit_memory_cells(
        [_cell("episodic-000001", "two blue chairs", metadata=_provenance("turn-999", "two blue chairs"))],
        SPANS,
    )

    assert report.status_counts == {UNSUPPORTED: 1}
    assert "not in the transcript" in report.findings[0].reason


def test_missing_provenance_is_unsupported() -> None:
    report = audit_memory_cells([_cell("episodic-000001", "two blue chairs")], SPANS)

    assert report.precision == 0.0
    assert report.status_counts == {MISSING_PROVENANCE: 1}


def test_user_fact_cannot_be_supported_by_an_assistant_span() -> None:
    report = audit_memory_cells(
        [
            _cell(
                "episodic-000001",
                "the user owns two blue chairs",
                metadata=_provenance("turn-2", "matching table", role="assistant"),
            )
        ],
        SPANS,
        policy=MemoryPrecisionPolicy(require_user_authorship=True),
    )

    assert report.status_counts == {ROLE_VIOLATION: 1}
    assert "user-authored claim" in report.findings[0].reason


def test_derived_cell_is_supported_by_supported_parents() -> None:
    cells = [
        _cell("episodic-000001", "two blue chairs", metadata=_provenance("turn-1", "two blue chairs")),
        _cell("episodic-000002", "sister in Lisbon", metadata=_provenance("turn-3", "moved to Lisbon")),
        _cell(
            "semantic-000003",
            "the user furnished a home and has family abroad",
            metadata={"derived_from_cell_ids": ["episodic-000001", "episodic-000002"]},
        ),
    ]

    report = audit_memory_cells(cells, SPANS)

    assert report.precision == 1.0
    assert report.status_counts == {SUPPORTED: 3}


def test_derived_cell_with_unsupported_parent_fails_closed() -> None:
    cells = [
        _cell("episodic-000001", "two blue chairs"),
        _cell(
            "semantic-000002",
            "the user furnished a home",
            metadata={"derived_from_cell_ids": ["episodic-000001"]},
        ),
    ]

    report = audit_memory_cells(cells, SPANS)

    assert report.precision == 0.0
    assert report.status_counts == {MISSING_PROVENANCE: 1, UNRESOLVED_DERIVATION: 1}


def test_cyclic_derivation_is_unresolved() -> None:
    cells = [
        _cell("semantic-000001", "claim a", metadata={"derived_from_cell_ids": ["semantic-000002"]}),
        _cell("semantic-000002", "claim b", metadata={"derived_from_cell_ids": ["semantic-000001"]}),
    ]

    report = audit_memory_cells(cells, SPANS)

    assert report.precision == 0.0
    assert {finding.status for finding in report.findings} == {UNRESOLVED_DERIVATION}


def test_empty_store_is_vacuously_precise_but_marked_empty() -> None:
    report = audit_memory_cells([], SPANS)

    assert report.precision == 1.0
    assert report.target_met is True
    assert report.empty is True
    assert report.total_cells == 0


def test_assert_memory_precision_raises_with_cell_detail() -> None:
    report = audit_memory_cells([_cell("episodic-000001", "fabricated")], SPANS)

    with pytest.raises(MemoryPrecisionError) as error:
        assert_memory_precision(report)

    assert "episodic-000001" in str(error.value)
    assert "below target" in str(error.value)


def test_report_serializes_without_source_text() -> None:
    report = audit_memory_cells(
        [_cell("episodic-000001", "two blue chairs", metadata=_provenance("turn-1", "two blue chairs"))],
        SPANS,
    )
    payload = report.to_dict()

    assert payload["metric"] == "stored_memory_factual_precision"
    assert payload["precision"] == 1.0
    assert "I bought two blue chairs" not in json.dumps(payload)


def test_load_transcript_supports_jsonl_and_json_arrays(tmp_path: Path) -> None:
    jsonl = tmp_path / "spans.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"source_id": "turn-1", "role": "user", "text": "hello"},
                {"id": "turn-2", "role": "assistant", "text": "hi"},
            )
        ),
        encoding="utf-8",
    )
    array = tmp_path / "spans.json"
    array.write_text(json.dumps([{"source_id": "turn-1", "role": "user", "text": "hello"}]), encoding="utf-8")

    assert [span.source_id for span in load_transcript(jsonl)] == ["turn-1", "turn-2"]
    assert [span.source_id for span in load_transcript(array)] == ["turn-1"]


def test_cli_reports_100_percent_for_a_provenance_complete_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_dir = tmp_path / "state"
    persistence = SessionPersistence(PersistenceConfig(mode="in_process", state_dir=str(state_dir), fsync="never"))
    store = persistence.hydrate("session-1")
    store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="two blue chairs",
        summary="two blue chairs",
        source="session-1",
        metadata=_provenance("turn-1", "two blue chairs"),
    )
    persistence.flush()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"source_id": "turn-1", "role": "user", "text": "I bought two blue chairs on 2023-03-04."}),
        encoding="utf-8",
    )
    output = tmp_path / "precision.json"

    exit_code = main(
        [
            "memory-precision",
            "--state-dir",
            str(state_dir),
            "--session-id",
            "session-1",
            "--transcript",
            str(transcript),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["precision"] == 1.0
    assert payload["target_met"] is True
    assert payload["total_cells"] == 1


def test_cli_exits_nonzero_when_precision_is_below_target(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    persistence = SessionPersistence(PersistenceConfig(mode="in_process", state_dir=str(state_dir), fsync="never"))
    store = persistence.hydrate("session-1")
    store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="a fabricated fact",
        summary="a fabricated fact",
        source="session-1",
        metadata=_provenance("turn-1", "a fabricated fact"),
    )
    persistence.flush()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"source_id": "turn-1", "role": "user", "text": "I bought two blue chairs on 2023-03-04."}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "memory-precision",
            "--state-dir",
            str(state_dir),
            "--session-id",
            "session-1",
            "--transcript",
            str(transcript),
        ]
    )

    assert exit_code == 1
