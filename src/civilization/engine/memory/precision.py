"""Stored-memory precision: verify every memory cell against its sources.

The LongMemEval answerer measures answer accuracy. That is a different claim
from the one this module supports: whether the cells a session stores are
factually supported by the transcript they came from. A memory system can answer
well while silently storing invented facts, and it can store only true facts
while answering poorly. The two need separate, separately reproducible
measurements.

The audit here is deliberately strict and local:

* a cell is **supported** only when every claimed provenance quote is a
  normalized substring of the referenced source span, and the span's role is
  compatible with the cell's claim (unless the policy allows assistant
  evidence);
* a **derived** cell (a consolidation summary) is supported when every parent
  cell is supported; the summary itself is not re-quoted because it is not a
  verbatim claim;
* a cell with no provenance and no derivations is **missing_provenance**;
* a fabricated quote is **unsupported** even when its shape looks valid.

``precision = supported / total``. A store with no cells is vacuously precise at
1.0 and says so with ``empty=True``; it must not be used as evidence for a
production corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
MISSING_PROVENANCE = "missing_provenance"
ROLE_VIOLATION = "role_violation"
UNRESOLVED_DERIVATION = "unresolved_derivation"

USER_ROLES = frozenset({"user", "human"})
ASSISTANT_ROLES = frozenset({"assistant", "agent", "system", "tool"})


class MemoryPrecisionError(RuntimeError):
    """Raised when a store misses the configured precision target."""


@dataclass(frozen=True)
class TranscriptSpan:
    """One addressable turn of source text."""

    source_id: str
    role: str
    text: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")

    @property
    def normalized_role(self) -> str:
        return self.role.strip().casefold()


@dataclass(frozen=True)
class MemoryPrecisionPolicy:
    """Which provenance shapes count as proof."""

    require_user_authorship: bool = False
    allow_assistant_evidence: bool = True
    allow_derived_cells: bool = True
    require_every_quote: bool = True
    normalize_unicode: bool = True


@dataclass(frozen=True)
class CellPrecisionFinding:
    cell_id: str
    status: str
    reason: str
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "status": self.status,
            "reason": self.reason,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class MemoryPrecisionReport:
    session_id: str
    total_cells: int
    supported_cells: int
    precision: float
    target: float
    target_met: bool
    empty: bool
    status_counts: dict[str, int] = field(default_factory=dict)
    findings: tuple[CellPrecisionFinding, ...] = ()
    policy: MemoryPrecisionPolicy = field(default_factory=MemoryPrecisionPolicy)
    transcript_span_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "metric": "stored_memory_factual_precision",
            "definition": "supported memory cells / all stored memory cells",
            "total_cells": self.total_cells,
            "supported_cells": self.supported_cells,
            "precision": self.precision,
            "target": self.target,
            "target_met": self.target_met,
            "empty": self.empty,
            "status_counts": dict(self.status_counts),
            "transcript_span_count": self.transcript_span_count,
            "policy": {
                "require_user_authorship": self.policy.require_user_authorship,
                "allow_assistant_evidence": self.policy.allow_assistant_evidence,
                "allow_derived_cells": self.policy.allow_derived_cells,
                "require_every_quote": self.policy.require_every_quote,
                "normalize_unicode": self.policy.normalize_unicode,
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _normalize(text: str, *, normalize_unicode: bool = True) -> str:
    value = unicodedata.normalize("NFKC", text) if normalize_unicode else text
    return " ".join(value.casefold().split())


def _metadata(cell: Any) -> dict[str, Any]:
    value = getattr(cell, "metadata", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _provenance_entries(cell: Any) -> list[dict[str, str]]:
    """Return the cell's claimed source quotes in a single normalized shape."""

    metadata = _metadata(cell)
    entries: list[dict[str, str]] = []
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping):
        spans = provenance.get("spans")
        if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes)):
            for span in spans:
                if isinstance(span, Mapping):
                    entries.append(
                        {
                            "source_id": str(span.get("source_id", "")),
                            "quote": str(span.get("quote", "")),
                            "role": str(span.get("role", "")),
                        }
                    )
        else:
            entries.append(
                {
                    "source_id": str(provenance.get("source_id", "")),
                    "quote": str(provenance.get("quote", "")),
                    "role": str(provenance.get("role", "")),
                }
            )
    if not entries:
        source_id = metadata.get("source_id")
        quote = metadata.get("source_quote") or metadata.get("evidence_quote")
        role = metadata.get("source_role")
        if source_id or quote or role:
            entries.append(
                {
                    "source_id": str(source_id or ""),
                    "quote": str(quote or ""),
                    "role": str(role or ""),
                }
            )
    return [entry for entry in entries if any(entry.values())]


def _derived_from(cell: Any) -> tuple[str, ...]:
    metadata = _metadata(cell)
    value = metadata.get("derived_from_cell_ids") or metadata.get("parent_cell_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _entry_status(
    entry: Mapping[str, str],
    spans: Mapping[str, TranscriptSpan],
    policy: MemoryPrecisionPolicy,
) -> tuple[str | None, str, tuple[str, ...]]:
    source_id = entry.get("source_id", "").strip()
    quote = entry.get("quote", "").strip()
    role = entry.get("role", "").strip().casefold()
    if not source_id or not quote:
        return MISSING_PROVENANCE, "provenance needs both source_id and quote", (source_id,) if source_id else ()
    span = spans.get(source_id)
    if span is None:
        return UNSUPPORTED, f"source span {source_id!r} is not in the transcript", (source_id,)
    span_role = role or span.normalized_role
    if policy.require_user_authorship and span_role not in USER_ROLES:
        return ROLE_VIOLATION, f"user-authored claim cites a {span_role or 'unknown'} span", (source_id,)
    if not policy.allow_assistant_evidence and span_role in ASSISTANT_ROLES:
        return ROLE_VIOLATION, f"assistant evidence is disallowed but {source_id!r} is {span_role}", (source_id,)
    normalized_quote = _normalize(quote, normalize_unicode=policy.normalize_unicode)
    normalized_span = _normalize(span.text, normalize_unicode=policy.normalize_unicode)
    if normalized_quote not in normalized_span:
        return UNSUPPORTED, f"quote is not a normalized substring of {source_id!r}", (source_id,)
    return None, "", (source_id,)


def audit_memory_cells(
    cells: Iterable[Any],
    spans: Iterable[TranscriptSpan],
    *,
    session_id: str = "default",
    target: float = 1.0,
    policy: MemoryPrecisionPolicy | None = None,
) -> MemoryPrecisionReport:
    """Audit every cell and return a deterministic precision report."""

    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    policy = policy or MemoryPrecisionPolicy()
    span_index = {span.source_id: span for span in spans}
    cell_list = list(cells)
    statuses: dict[str, str] = {}
    reasons: dict[str, str] = {}
    source_ids: dict[str, tuple[str, ...]] = {}

    # A cell's status can depend on its parents, so resolve derivations with an
    # explicit visiting set. Cycles fail closed instead of recursing forever.
    def status_of(cell_id: str, visiting: frozenset[str]) -> tuple[str, str, tuple[str, ...]]:
        if cell_id in statuses:
            return statuses[cell_id], reasons[cell_id], source_ids.get(cell_id, ())
        if cell_id in visiting:
            return UNRESOLVED_DERIVATION, "cyclic derived-memory provenance", (cell_id,)
        cell = by_id.get(cell_id)
        if cell is None:
            return UNRESOLVED_DERIVATION, f"parent cell {cell_id!r} is not in the store", (cell_id,)
        entries = _provenance_entries(cell)
        if entries:
            collected: list[str] = []
            for entry in entries:
                status, reason, cited = _entry_status(entry, span_index, policy)
                collected.extend(cited)
                if status is not None:
                    statuses[cell_id] = status
                    reasons[cell_id] = reason
                    source_ids[cell_id] = tuple(dict.fromkeys(collected))
                    return status, reason, source_ids[cell_id]
                if not policy.require_every_quote:
                    break
            statuses[cell_id] = SUPPORTED
            reasons[cell_id] = ""
            source_ids[cell_id] = tuple(dict.fromkeys(collected))
            return SUPPORTED, "", source_ids[cell_id]
        parents = _derived_from(cell)
        if parents and policy.allow_derived_cells:
            collected = []
            for parent in parents:
                status, reason, cited = status_of(parent, visiting | {cell_id})
                collected.extend(cited)
                if status != SUPPORTED:
                    statuses[cell_id] = UNRESOLVED_DERIVATION
                    reasons[cell_id] = f"parent {parent!r} is {status}"
                    source_ids[cell_id] = tuple(dict.fromkeys(collected))
                    return UNRESOLVED_DERIVATION, reasons[cell_id], source_ids[cell_id]
            statuses[cell_id] = SUPPORTED
            reasons[cell_id] = ""
            source_ids[cell_id] = tuple(dict.fromkeys(collected))
            return SUPPORTED, "", source_ids[cell_id]
        statuses[cell_id] = MISSING_PROVENANCE
        reasons[cell_id] = "cell has neither a source quote nor supported parents"
        source_ids[cell_id] = ()
        return MISSING_PROVENANCE, reasons[cell_id], ()

    by_id = {}
    for cell in cell_list:
        cell_id = str(getattr(cell, "cell_id", "") or "")
        if not cell_id:
            raise ValueError("every audited cell needs a cell_id")
        if cell_id in by_id:
            raise ValueError(f"duplicate cell_id {cell_id!r}")
        by_id[cell_id] = cell

    findings: list[CellPrecisionFinding] = []
    counts: dict[str, int] = {}
    for cell in cell_list:
        cell_id = str(cell.cell_id)
        status, reason, cited = status_of(cell_id, frozenset())
        counts[status] = counts.get(status, 0) + 1
        if status != SUPPORTED:
            findings.append(
                CellPrecisionFinding(
                    cell_id=cell_id,
                    status=status,
                    reason=reason,
                    source_ids=cited,
                )
            )

    total = len(cell_list)
    supported = counts.get(SUPPORTED, 0)
    precision = 1.0 if total == 0 else supported / total
    return MemoryPrecisionReport(
        session_id=session_id,
        total_cells=total,
        supported_cells=supported,
        precision=precision,
        target=target,
        target_met=precision + 1e-12 >= target,
        empty=total == 0,
        status_counts=counts,
        findings=tuple(findings),
        policy=policy,
        transcript_span_count=len(span_index),
    )


def assert_memory_precision(report: MemoryPrecisionReport) -> None:
    """Fail closed when the report misses its target."""

    if report.target_met:
        return
    unsupported = report.total_cells - report.supported_cells
    detail = "; ".join(
        f"{finding.cell_id}: {finding.status} ({finding.reason})"
        for finding in report.findings[:5]
    )
    raise MemoryPrecisionError(
        f"stored-memory precision {report.precision:.4f} is below target "
        f"{report.target:.4f} ({unsupported} of {report.total_cells} cells unsupported)"
        + (f": {detail}" if detail else "")
    )


def load_transcript(path: str | Path) -> list[TranscriptSpan]:
    """Load JSONL spans (``{source_id, role, text}``) or a JSON list."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    rows: list[Any]
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    spans: list[TranscriptSpan] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"transcript row {index} is not an object")
        source_id = row.get("source_id") or row.get("id")
        if not source_id:
            raise ValueError(f"transcript row {index} is missing source_id")
        spans.append(
            TranscriptSpan(
                source_id=str(source_id),
                role=str(row.get("role", "unknown")),
                text=str(row.get("text", "")),
            )
        )
    return spans
