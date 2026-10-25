"""Preserve distinct source sessions for temporal Orion reads."""

from __future__ import annotations

from typing import Any, Iterable

from .stage164_orion_temporal_coverage import is_temporal_coverage_query

_MAX_LOOKAHEAD = 2


def reorder_for_source_session_coverage(
    results: Iterable[Any], query: str
) -> tuple[list[Any], bool]:
    """Move a new source session ahead of nearby same-session repeats.

    The relevance ranker still chooses the candidate set. This pass only
    changes order for temporal or multi-event questions, and advances a new
    source session by at most two ranks. Cells without source-session metadata
    keep their existing order; date coverage is handled by the preceding
    temporal-coverage stage.
    """

    ordered = list(results)
    if len(ordered) < 2 or not is_temporal_coverage_query(query):
        return ordered, False

    session_keys = [_source_session_key(result) for result in ordered]
    seen_sessions: set[str] = set()
    applied = False
    index = 0
    while index < len(ordered):
        session_key = session_keys[index]
        if session_key is None or session_key not in seen_sessions:
            if session_key is not None:
                seen_sessions.add(session_key)
            index += 1
            continue

        stop = min(len(ordered), index + 1 + _MAX_LOOKAHEAD)
        candidate_index = next(
            (
                candidate
                for candidate in range(index + 1, stop)
                if session_keys[candidate] is not None
                and session_keys[candidate] not in seen_sessions
            ),
            None,
        )
        if candidate_index is None:
            index += 1
            continue

        result = ordered.pop(candidate_index)
        candidate_key = session_keys.pop(candidate_index)
        ordered.insert(index, result)
        session_keys.insert(index, candidate_key)
        seen_sessions.add(candidate_key)
        applied = True
        index += 1

    return ordered, applied


def _source_session_key(result: Any) -> str | None:
    cell = getattr(result, "cell", None)
    if cell is None:
        nested_result = getattr(result, "result", None)
        cell = getattr(nested_result, "cell", None)
    if cell is None:
        return None

    metadata = getattr(cell, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    source_session_id = metadata.get("source_session_id")
    if not isinstance(source_session_id, str) or not source_session_id.strip():
        return None

    source_user_id = metadata.get("source_user_id")
    if isinstance(source_user_id, str) and source_user_id:
        return f"{source_user_id}\x1f{source_session_id}"
    return source_session_id
