"""Coverage-first ordering for multi-event and temporal memory questions."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Iterable


_AGGREGATION_MARKERS = frozenset(
    {
        "after",
        "ago",
        "all",
        "before",
        "between",
        "date",
        "days",
        "day",
        "duration",
        "earlier",
        "earliest",
        "every",
        "first",
        "howmany",
        "howoften",
        "later",
        "long",
        "month",
        "months",
        "ordered",
        "sequence",
        "timeline",
        "time",
        "times",
        "week",
        "weeks",
        "when",
        "year",
        "years",
    }
)
_LATEST_MARKERS = frozenset({"current", "currently", "latest", "newest", "recent", "recently", "now"})
_CJK_AGGREGATION_MARKERS = ("之前", "之后", "最早", "顺序", "时间", "时间线", "多少次", "多久", "每次")
_CJK_LATEST_MARKERS = ("当前", "目前", "现在", "最新", "最近")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def reorder_for_temporal_coverage(results: Iterable[Any], query: str) -> tuple[list[Any], bool]:
    """Place the best cell from each observed date before same-date repeats.

    The relevance ranker still selects candidates and breaks date ties. This
    pass only changes the order for questions that ask for a sequence or an
    aggregate across events. Queries asking for the latest/current fact keep
    their existing recency-first ranking.
    """

    ordered = list(results)
    if len(ordered) < 2 or not is_temporal_coverage_query(query):
        return ordered, False

    first_per_date: list[Any] = []
    repeats: list[Any] = []
    seen_dates: set[str] = set()
    for result in ordered:
        date_key = _date_key(result)
        if date_key is None or date_key not in seen_dates:
            first_per_date.append(result)
            if date_key is not None:
                seen_dates.add(date_key)
        else:
            repeats.append(result)
    return [*first_per_date, *repeats], len(seen_dates) > 1


def is_temporal_coverage_query(query: str) -> bool:
    """Return whether a query benefits from multi-event evidence coverage."""

    text = query.casefold()
    words = set(_WORD_RE.findall(text))
    if words & _LATEST_MARKERS or any(marker in text for marker in _CJK_LATEST_MARKERS):
        return False
    normalized = "".join(character for character in text if character.isalnum())
    if any(marker in normalized for marker in ("howmany", "howoften")):
        return True
    return bool(words & _AGGREGATION_MARKERS) or any(marker in text for marker in _CJK_AGGREGATION_MARKERS)


def _date_key(result: Any) -> str | None:
    cell = getattr(result, "cell", None)
    if cell is None:
        nested_result = getattr(result, "result", None)
        cell = getattr(nested_result, "cell", None)
    if cell is None:
        return None
    metadata = getattr(cell, "metadata", {})
    if isinstance(metadata, dict):
        observation_date = metadata.get("observation_date")
        if isinstance(observation_date, str) and _DATE_RE.match(observation_date):
            return observation_date[:10]
    time_index = getattr(cell, "time_index", None)
    try:
        timestamp = float(time_index)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None
