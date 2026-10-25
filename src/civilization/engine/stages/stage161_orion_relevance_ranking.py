"""Deterministic, dependency-free fielded ranking for Orion memory search.

The ranker gives summaries more weight than long body text and uses inverse
document frequency plus length normalization. It intentionally avoids model
calls: callers may add a semantic candidate source later without changing the
Orion store contract or making reads depend on an external provider.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9_\u4e00-\u9fff]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an and are as at be been being between by can could did do does doing many "
    "for from had has have how i in into is it its may me my of on or our shall should "
    "that the their them then there these they this those through to was we were what "
    "when where which who why will with would you your".split()
)
_RECENCY_MARKERS = frozenset({"current", "currently", "latest", "newest", "recent", "recently", "now"})
_CJK_RECENCY_MARKERS = ("当前", "目前", "现在", "最新", "最近")
_FIELD_WEIGHTS = {
    "summary": 2.0,
    "content": 1.0,
    "source": 0.2,
    "user_content": 1.0,
    "assistant_content": 1.0,
    "other_content": 0.5,
}


def tokenize(text: str) -> tuple[str, ...]:
    """Return normalized word tokens and CJK character bigrams."""

    output: list[str] = []
    for raw_token in _TOKEN_RE.findall(text):
        token = raw_token.casefold()
        if all("\u4e00" <= character <= "\u9fff" for character in token):
            if len(token) == 1:
                output.append(token)
            else:
                output.extend(token[offset : offset + 2] for offset in range(len(token) - 1))
        else:
            output.append(token)
    return tuple(output)


def has_recency_intent(query: str, query_tokens: tuple[str, ...] | None = None) -> bool:
    query_tokens = query_tokens if query_tokens is not None else tokenize(query)
    return bool(_RECENCY_MARKERS.intersection(query_tokens)) or any(
        marker in query.casefold() for marker in _CJK_RECENCY_MARKERS
    )


@dataclass(frozen=True)
class RankedMemoryCell:
    cell: Any
    score: float
    matched_terms: tuple[str, ...]
    link_score: float
    score_breakdown: dict[str, float]


@dataclass(frozen=True)
class _IndexedCell:
    cell: Any
    fields: dict[str, tuple[str, ...]]
    term_counts: dict[str, Counter[str]]
    term_sets: dict[str, frozenset[str]]
    all_terms: frozenset[str]
    has_role_fields: bool


class OrionRelevanceRanker:
    """Fielded BM25F ranking with bounded priors and stable temporal ordering."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, phrase_bonus: float = 0.35) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if phrase_bonus < 0:
            raise ValueError("phrase_bonus must be non-negative")
        self.k1 = float(k1)
        self.b = float(b)
        self.phrase_bonus = float(phrase_bonus)

    def rank(
        self,
        query: str,
        cells: Iterable[Any],
        *,
        link_score_for: Callable[[Any], float],
        ranking_context: Iterable[Any] = (),
    ) -> list[RankedMemoryCell]:
        candidates = [self._index_cell(cell) for cell in cells]
        if not candidates:
            return []
        candidate_object_ids = {id(document.cell) for document in candidates}
        context = [
            self._index_cell(cell)
            for cell in ranking_context
            if id(cell) not in candidate_object_ids
        ]
        documents = [*candidates, *context]

        query_tokens = tokenize(query)
        query_terms = tuple(dict.fromkeys(term for term in query_tokens if term not in _STOPWORDS))
        if not query_terms:
            query_terms = tuple(dict.fromkeys(query_tokens))
        recency_intent = has_recency_intent(query, query_tokens)
        role_aware_corpus = any(document.has_role_fields for document in documents)
        field_weights = self._field_weights_for_query(query, query_tokens) if role_aware_corpus else dict(_FIELD_WEIGHTS)
        document_count = len(documents)
        field_averages = {
            name: max(1.0, sum(len(document.fields[name]) for document in documents) / document_count)
            for name in _FIELD_WEIGHTS
        }
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(document.all_terms)

        ranked: list[RankedMemoryCell] = []
        for document in candidates:
            matched_terms = tuple(sorted(set(query_tokens) & document.all_terms))
            bm25f = self._bm25f_score(
                document,
                query_terms,
                document_count,
                document_frequency,
                field_averages,
                field_weights,
            )
            phrase = self._phrase_match(query_terms, document) if len(query_terms) > 1 else False
            phrase_component = self.phrase_bonus if phrase else 0.0
            importance = self._bounded_prior(getattr(document.cell, "importance", 0.0)) * 0.05
            confidence = self._bounded_prior(getattr(document.cell, "confidence", 0.0)) * 0.05
            link_score = self._nonnegative(link_score_for(document.cell))
            score = bm25f + phrase_component + importance + confidence + link_score
            ranked.append(
                RankedMemoryCell(
                    cell=document.cell,
                    score=score,
                    matched_terms=matched_terms,
                    link_score=link_score,
                    score_breakdown={
                        "bm25f": bm25f,
                        "phrase": phrase_component,
                        "importance_prior": importance,
                        "confidence_prior": confidence,
                        "link": link_score,
                        "user_role_weight": field_weights["user_content"],
                        "assistant_role_weight": field_weights["assistant_content"],
                    },
                )
            )

        ranked.sort(key=lambda item: self._sort_key(item, recency_intent))
        return ranked

    def _fields_for_cell(self, cell: Any) -> dict[str, tuple[str, ...]]:
        return {
            "summary": tokenize(str(getattr(cell, "summary", ""))),
            "content": tokenize(str(getattr(cell, "content", ""))),
            "source": tokenize(str(getattr(cell, "source", ""))),
            "user_content": (),
            "assistant_content": (),
            "other_content": (),
        }

    @staticmethod
    def _field_weights_for_query(query: str, query_tokens: tuple[str, ...]) -> dict[str, float]:
        return dict(_FIELD_WEIGHTS)

    def _index_cell(self, cell: Any) -> _IndexedCell:
        fields = self._fields_for_cell(cell)
        term_counts = {name: Counter(tokens) for name, tokens in fields.items()}
        term_sets = {name: frozenset(tokens) for name, tokens in fields.items()}
        return _IndexedCell(
            cell=cell,
            fields=fields,
            term_counts=term_counts,
            term_sets=term_sets,
            all_terms=frozenset().union(*term_sets.values()),
            has_role_fields=bool(term_sets["user_content"] or term_sets["assistant_content"]),
        )

    def _bm25f_score(
        self,
        document: _IndexedCell,
        query_terms: tuple[str, ...],
        document_count: int,
        document_frequency: Counter[str],
        field_averages: dict[str, float],
        field_weights: dict[str, float],
    ) -> float:
        score = 0.0
        for term in query_terms:
            weighted_frequency = 0.0
            for field_name, field_weight in field_weights.items():
                tokens = document.fields[field_name]
                frequency = document.term_counts[field_name].get(term, 0)
                if frequency == 0:
                    continue
                denominator = 1.0 - self.b + self.b * len(tokens) / field_averages[field_name]
                weighted_frequency += field_weight * frequency / denominator
            if weighted_frequency <= 0:
                continue
            frequency = document_frequency[term]
            inverse_frequency = math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
            score += inverse_frequency * (weighted_frequency * (self.k1 + 1.0)) / (weighted_frequency + self.k1)
        return score

    @staticmethod
    def _phrase_match(query_terms: tuple[str, ...], document: _IndexedCell) -> bool:
        if len(query_terms) < 2:
            return False
        for field_name in document.fields:
            tokens = document.fields[field_name]
            width = len(query_terms)
            if any(tokens[offset : offset + width] == query_terms for offset in range(len(tokens) - width + 1)):
                return True
        return False

    @staticmethod
    def _bounded_prior(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, number)) if math.isfinite(number) else 0.0

    @staticmethod
    def _nonnegative(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, number) if math.isfinite(number) else 0.0

    @staticmethod
    def _sort_key(item: RankedMemoryCell, recency_intent: bool) -> tuple[float, float, str]:
        time_index = getattr(item.cell, "time_index", 0.0)
        try:
            timestamp = float(time_index)
        except (TypeError, ValueError):
            timestamp = 0.0
        if not math.isfinite(timestamp):
            timestamp = 0.0
        temporal_key = -timestamp if recency_intent else timestamp
        return (-item.score, temporal_key, str(item.cell.cell_id))
