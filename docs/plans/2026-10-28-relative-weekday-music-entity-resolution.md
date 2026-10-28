# Relative Weekday Music Entity Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover an unnamed music artist/band description when a user-authored event on the exact weekday requested is present in Top-20 evidence.

**Architecture:** Add a default-off answer resolver. It computes the previous occurrence of a weekday from the question date, then requires a cited temporal event whose date is re-derived from an exact user-authored quote and equals that weekday. It extracts only the music entity description contained in the quote, rejects multiple distinct entities, and validates the final answer with the shared evidence validator. It does not infer dates from retrieval order or accept assistant text.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest/pytest.

---

## Evidence and scope

Stage268 row `gpt4_fa19884d` asks which artist the user started listening to last Friday. The answerer abstained, but its Top-20 temporal audit cites a user-authored quote dated Friday, 2023-03-31: “I recently discovered a bluegrass band that features a banjo player and started enjoying their music today.” The question date is Wednesday, 2023-04-05, so last Friday is 2023-03-31. The descriptive candidate “A bluegrass band that features a banjo player” passed the configured judge with seed 42 accepted.

## Task 1: Exact user-source weekday resolver

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`
- Test `tests/test_longmemeval_prompts.py`

**Acceptance:**
- Support “last <weekday>” only for artist/band/music entity questions.
- Resolve the previous weekday from the question date, using the prior week if the question itself is on that weekday.
- Require an exact source date derived from user-authored event citations; it must equal the computed weekday.
- Extract the artist/band descriptor from the exact user quote, not from assistant text or retrieval position.
- Reject missing dates, multiple distinct entity candidates, unsupported source roles, and invalid citations.
- Validate every rewrite with `validate_longmemeval_answer`.

## Task 2: Default-off pipeline wiring

Add `--relative-weekday-event-answer-resolution`, per-question audit status, and run metadata. Add tests for the disabled default, enabled pipeline, ambiguous identities, and assistant-only evidence.

## Task 3: Evaluation and delivery

Replay Stage268 saved answers, rejudge only changed rows with the configured judge and seed 42, and verify zero replay regressions. If the focused replay passes, run a fresh 500-answer candidate over the exact frozen Stage268 retrieval snapshot with this feature and the existing Stage273 tenure resolver enabled. Compare paired outcomes, retrieval order, invalid audits, and feature trigger count. Commit adapter code/tests with an English message and commit the Civilization scorecard separately.
