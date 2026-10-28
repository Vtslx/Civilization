# Research Poster Conference Venue Resolution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover a conference venue for a thesis poster when separate user-authored memories identify the same uniquely named event.

**Architecture:** Add a default-off resolver for thesis-poster venue questions. It joins only exact user sentences that share the unique event key “my first research conference”: one sentence must establish a poster presentation on thesis research, and another must identify the university hosting that same conference. A single unique venue is required, and both source sentences remain cited. If the event key is absent, assistant-only, or maps to multiple venues, preserve the answer.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest/pytest.

---

## Evidence and scope

Stage268 row `a96c20ee` asks which university hosted the user's thesis poster. Top-20 includes a user-authored statement about presenting a thesis-research poster at “my first research conference” and another user-authored statement about attending “my first research conference” at Harvard University. The answerer abstained, but the shared event wording uniquely connects the two facts. A direct answer of “Harvard University” was accepted by `group/auto-gpt-6-luna` with seed 42 accepted.

The nearby abstention question `a96c20ee_abs` asks about a poster for an undergraduate course research project. This resolver must not fire for that different event.

## Task 1: Source-bound event alias resolver

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`
- Test `tests/test_longmemeval_prompts.py`

**Acceptance:**
- Trigger only for thesis-research poster venue questions.
- Require a user-authored poster sentence and venue sentence that both name the same unique first research conference.
- Extract one unique university name from a user-authored quote; assistant-only venue mentions are ignored.
- Preserve both exact citations and pass the shared answer validator.
- Reject absent aliases, conflicting locations, invalid citations, and undergraduate-course poster questions.

## Task 2: Default-off pipeline wiring

Add `--research-poster-conference-venue-resolution`, run metadata, and per-question audit status. Test that the default preserves abstention and the enabled path returns Harvard only for the exact linked user sources.

## Task 3: Replay and full candidate

Replay Stage268's saved answer for `a96c20ee`, rejudge changed rows with seed 42, and verify zero regressions. If the replay passes, run all 500 questions over the same Stage268 frozen retrieval and compare paired outcomes, order equivalence, and invalid audits.

## Task 4: Commit and record

Commit code/tests with an English message and add a Civilization scorecard. Keep saved-answer replay distinct from fresh full-generation accuracy and from memory-precision measurement.
