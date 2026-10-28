# Undergraduate Course Poster Venue Evidence Guard Plan

> **For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.**

**Goal:** Prevent a venue from being transferred from a thesis-research conference poster to a different undergraduate-course poster question without matching user evidence.

**Architecture:** Add a default-off answer guard scoped to university questions about an undergraduate-course research poster. It preserves an answer only when user-authored evidence directly names the university for that course-project poster. If the answer uses a venue supported only for another poster/event, replace it with a cited abstention explaining that the memories do not link the course project to a university. Assistant text does not qualify as venue evidence. The guard does not infer or assign a university.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and constraints

Stage280 row `a96c20ee_abs` asks where the user presented a poster for an undergraduate-course research project. Top-20 does not contain a user-authored sentence for that course-project poster. It does contain a user-authored thesis-poster statement naming the first research conference and a separate statement that the user attended that conference at Harvard. Stage280 transferred Harvard to the course-project question and failed. A cited abstention distinguishing those events passed a live judge call with seed 42 accepted.

The existing Stage277 thesis-poster resolver is opt-in and must remain scoped to thesis-research conference questions. This new guard is separate and abstains on unsupported course-poster venue claims; it must not answer with Harvard based on the unrelated conference.

## Task 1: Add guard tests

**Files:**
- Modify `tests/test_longmemeval_prompts.py`

**Acceptance:** Cover the Stage280 cross-event false positive, a direct user-authored course-poster quote that includes its venue, assistant-only claims, existing abstentions, and unrelated thesis-poster questions. Validate all rewritten citations.

## Task 2: Implement the default-off source-bound guard

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`

**Acceptance:** Trigger only for questions asking where an undergraduate-course research poster was presented. Preserve a venue only when a user-authored course-poster source directly supports that venue. Otherwise, if the answer asserts a university based on another event, abstain with exact user citations that make the evidence mismatch clear. Leave the response unchanged if no valid citations can be constructed.

## Task 3: Wire the answer pipeline

**Files:**
- Modify `benchmarks/longmemeval/run.py`
- Modify `tests/test_longmemeval_run.py`
- Modify `README.md`

**Acceptance:** Add default-off `--undergrad-poster-venue-evidence-guard`, per-question audit status, and run metadata. Test both default behavior and enabled abstention.

## Task 4: Verify and benchmark

Run the full adapter suite, `compileall`, and `git diff --check`. Replay `a96c20ee_abs` on Stage268 frozen Top-20 evidence and judge the revised abstention. Then run all 500 questions on frozen Stage268 Top-200 retrieval using the established model, seed, cutoff, and Stage279 feature flags plus the guard.

## Task 5: Commit and archive

Commit the adapter change with an English message. Write a Stage281 scorecard with the target replay, full-run pairings, exact resolver trigger count, and artifact hashes. Preserve Stage277 as best if Stage281 does not exceed 441/500.
