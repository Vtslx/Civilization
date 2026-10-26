# ADR-005: Optional Question-Type Answer Verification

**Status:** Accepted  
**Date:** 2026-10-26

## Context

Stage230 missed 6 of 70 single-session user questions and 5 of 56 assistant questions. The existing benchmark verifier covered cross-session, knowledge-update, temporal, and preference cases; it did not independently review these single-session answers.

In a frozen-input pilot over all 126 rows, a citation-backed verifier changed 23 answers, corrected 4 benchmark errors, and caused no regressions. The slice moved from 115/126 to 119/126.

## Decision

- Add `--answer-verification-question-types` as an opt-in list of additional question types.
- Preserve existing verification defaults and the separate preference-verification switch.
- Accept a revision only when its exact-quote citations and full answer schema validate; otherwise keep the original answer.
- Record the selected question types in result metadata so evaluations remain auditable.
- Keep the new categories disabled by default; the benefit is supported by a frozen-slice pilot, while a full-rerun score delta remains confounded by gateway reranker variance.

## Consequences

- Operators can target independent answer review without changing existing default request volume.
- A fresh 500-question Stage232 run with the single-session allowlist scored 432/500 (86.4%) versus Stage230’s 427/500, with 27 candidate-only and 22 baseline-only outcomes. Single-session user rose 64/70→68/70; assistant rose 51/56→52/56.
- Reranker selections matched in only 3/374 rows across those full runs. The aggregate difference cannot be attributed solely to answer verification; use the frozen pilot as the isolated evidence.
- Best full score remains Stage227 at 433/500 (86.6%), 52 below the 485/500 release gate.
