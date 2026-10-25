"""Speaker-aware relevance ranking for conversational memory cells.

When a cell preserves ``user:`` and ``assistant:`` turns, first-person questions
can prefer user evidence over assistant advice, and explicit recommendation
questions can prefer assistant evidence. Cells without speaker provenance keep
the Stage161 fielded BM25F behavior.
"""

from __future__ import annotations

import re
from typing import Any

from .stage161_orion_relevance_ranking import (
    OrionRelevanceRanker,
    tokenize,
)


_ROLE_MARKER = re.compile(r"(?im)^([a-z][a-z0-9_-]{0,31})\s*:\s*")
_USER_ROLES = frozenset({"user", "human"})
_ASSISTANT_ROLES = frozenset({"assistant", "agent"})
_USER_ACTIONS = frozenset({"ask", "asked", "say", "said", "tell", "told", "request", "requested"})
_ASSISTANT_INTENT = frozenset({"assistant", "suggest", "suggested", "suggestion", "recommend", "recommended", "advice", "advised"})
_PERSONAL_PRONOUNS = frozenset({"i", "me", "my", "mine", "we", "our", "ours"})


class OrionRoleAwareRanker(OrionRelevanceRanker):
    """Use speaker provenance as a query-dependent BM25F field weight."""

    version = "stage163_role_aware_bm25f"

    def _fields_for_cell(self, cell: Any) -> dict[str, tuple[str, ...]]:
        summary_text = str(getattr(cell, "summary", ""))
        content_text = str(getattr(cell, "content", ""))
        source_text = str(getattr(cell, "source", ""))
        fields = {
            "summary": tokenize(summary_text),
            "content": tokenize(content_text),
            "source": tokenize(source_text),
            "user_content": (),
            "assistant_content": (),
            "other_content": (),
        }

        role_matches = list(_ROLE_MARKER.finditer(content_text))
        if role_matches:
            user_parts: list[str] = []
            assistant_parts: list[str] = []
            other_parts = [content_text[: role_matches[0].start()]]
            for index, marker in enumerate(role_matches):
                end = role_matches[index + 1].start() if index + 1 < len(role_matches) else len(content_text)
                role_text = content_text[marker.end() : end]
                role = marker.group(1).casefold()
                if role in _USER_ROLES:
                    user_parts.append(role_text)
                elif role in _ASSISTANT_ROLES:
                    assistant_parts.append(role_text)
                else:
                    other_parts.append(role_text)
            fields["user_content"] = tokenize(" ".join(user_parts))
            fields["assistant_content"] = tokenize(" ".join(assistant_parts))
            fields["other_content"] = tokenize(" ".join(other_parts))
            # The benchmark adapter stores the full conversation in both
            # summary and content. Avoid counting that duplicate a second time.
            if summary_text.strip() == content_text.strip():
                fields["summary"] = tokenize(" ".join(other_parts))
            fields["content"] = ()
            return fields

        metadata = getattr(cell, "metadata", {})
        role = ""
        if isinstance(metadata, dict):
            role = str(metadata.get("speaker") or metadata.get("role") or "").casefold()
        if role in _USER_ROLES | _ASSISTANT_ROLES:
            if summary_text.strip() == content_text.strip():
                fields["summary"] = ()
            fields["content"] = ()
            role_field = "user_content" if role in _USER_ROLES else "assistant_content"
            fields[role_field] = tokenize(content_text)
        return fields

    def _field_weights_for_query(self, query: str, query_tokens: tuple[str, ...]) -> dict[str, float]:
        weights = super()._field_weights_for_query(query, query_tokens)
        tokens = set(query_tokens)
        lowered = query.casefold()
        user_action_query = bool(tokens & _PERSONAL_PRONOUNS) and bool(tokens & _USER_ACTIONS) and "assistant" in tokens
        assistant_query = bool(tokens & _ASSISTANT_INTENT) or any(
            marker in lowered for marker in ("你推荐", "助手建议", "助手推荐")
        )
        personal_query = bool(tokens & _PERSONAL_PRONOUNS) or "我" in lowered

        if user_action_query or (personal_query and not assistant_query):
            weights["user_content"] = 2.0
            weights["assistant_content"] = 0.25
        elif assistant_query:
            weights["user_content"] = 0.25
            weights["assistant_content"] = 2.0
        return weights
