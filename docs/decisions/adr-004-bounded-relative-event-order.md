# ADR-004: Bound Relative Event Ordering to Cited Date Intervals

**Status:** Accepted  
**Date:** 2026-10-26

## Context

Stage227 QID `gpt4_88806d6e` asks whether Mark and Sarah or Tom were met first. The cited user statements say Tom was met “a few months ago” and Mark and Sarah “about a month ago,” both relative to 2023-05-28. The generated temporal event strings included the anchor date, but the parser failed to recognize “few” and treated the embedded anchor date as Tom’s event date. That gave the wrong event order.

## Decision

- Keep the existing exact-date path.
- If a relative event-date string contains an unsupported relative marker such as “ago”, “before”, or “after”, do not fall through to an absolute date embedded in the same string.
- Resolve approximate intervals only from exact, validated user-authored citation quotes and an unambiguous memory observation date.
- Model “a few” as a broad 2–4 unit interval and “about N” as N ± 0.5 units. Compare interval bounds; update a two-option order answer only when one interval is fully earlier and safely separated from the other.
- Leave the response unchanged for overlapping intervals, missing observation dates, invalid citations, or assistant-only evidence.

## Consequences

- The resolver no longer treats an anchor date as an event date when relative phrasing is unparsed.
- Two-option event ordering can use user-authored relative quotes even when the model cannot provide an exact calendar day.
- The algorithm deliberately abstains when the intervals overlap; it does not fabricate a precise date.
- Stage229 frozen-output pilot changed two of 133 temporal answers: one wrong answer was corrected, one already-correct answer remained correct, and no regression occurred.
- Stage230 full answer generation scored 427/500, versus Stage227’s 433/500. Reranker selection varied in 372/374 rows between runs, so the full-run difference cannot be attributed to this resolver.
