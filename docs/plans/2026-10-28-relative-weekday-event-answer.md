# Relative Weekday Event Answer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Answer an unnamed artist/band question when an exact user-authored event description matches the weekday named in the question.

**Architecture:** Add a default-off resolver for artist/band questions using “last <weekday>”. Compute the previous weekday from the question date, then require a cited user event whose date is independently re-derived from an exact user quote and equals that day. Extract only the band/artist descriptor contained in the quote and pass it through the existing answer validator. Ambiguous dates, multiple distinct entities, assistant-only evidence, or missing event evidence leave the original answer unchanged.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest/pytest.

---

## Evidence and scope

Stage268 row `gpt4_fa19884d` asks for the artist the user started listening to last Friday. Top-20 includes a user-authored quote dated Friday, 2023-03-31: “I recently discovered a bluegrass band that features a banjo player and started enjoying their music today.” The question date is Wednesday, 2023-04-05, so the previous Friday is 2023-03-31. Stage268 abstained and was judged FAIL. A direct candidate answer, “A bluegrass band that features a banjo player,” was independently judged correct by `group/auto-gpt-6-luna` with seed 42 accepted.

The resolver must not infer an event date from retrieval rank or assistant text. It uses the prior-weekday date only to validate an explicitly user-dated event, and retains the exact supporting quote as the answer citation.

## Task 1: Strict weekday-date resolution

**Files:**
- Modify `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Test `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`

**Acceptance:**
- Trigger only when the question asks for an artist/band and names “last <weekday>”.
- Compute the previous occurrence of that weekday from the question date; “last Friday” on Friday means the prior week.
- Require an exact, user-authored source citation whose event date re-derives to that date.
- Extract a descriptive phrase only from the cited user quote; do not invent an artist name.
- Reject missing/ambiguous dates, conflicting entities, assistant-only claims, and invalid citations.
- Rewritten answer passes `validate_longmemeval_answer`.

## Task 2: Opt-in pipeline wiring

Add `--relative-weekday-event-answer-resolution`, default off. Record the flag and resolver status in answer audit and run metadata. Test default-off behavior and enabled single-row behavior with a fake answerer.

## Task 3: Frozen replay and full candidate

Run a deterministic replay over Stage268 saved answers, rejudging only changed rows with the configured judge and seed 42. Require zero replay regressions. If the replay passes, rerun all 500 answers over the exact Stage268 frozen retrieval snapshot with the feature enabled; compare paired outcomes and audit validity. Keep the previous full score unchanged unless the full candidate provides valid evidence to replace it.

## Task 4: Commit and record evidence

Commit adapter code and tests with an English message. Add a Civilization scorecard and append the verified result to the dedicated project archive. Do not report a saved-answer replay as a fresh full score or as memory-precision measurement.
