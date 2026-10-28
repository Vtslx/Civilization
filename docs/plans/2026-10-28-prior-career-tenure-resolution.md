# Prior Career Tenure Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover a prior-career duration when user-authored memories give total professional experience and current-employer tenure for the same observation date.

**Architecture:** Add a default-off, deterministic resolver to the LongMemEval adapter. It extracts durations only from exact user-authored sentences, requires one unique total and one unique current-employer tenure on the same observation date, subtracts them in months, then validates citations before replacing the answer. Missing, conflicting, assistant-only, or differently dated evidence leaves the generated answer unchanged.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval benchmark adapter, unittest/pytest.

---

## Evidence and scope

Stage268 row `gpt4_93159ced` asks how long the user had worked before starting the current NovaTech job. The answerer abstained, although Top-20 contains two exact user-authored statements in separate memories with the same 2023-05-25 observation date: nine years of professional work and about four years three months at NovaTech. The difference is about four years nine months. This pilot does not add a prompt-only general fact ledger, extra model call, or memory write.

## Task 1: Source-bound tenure arithmetic

**Files:**
- Modify `/Volumes/Net/memory-benchmarks-stage272/benchmarks/longmemeval/prompts.py`
- Test in `/Volumes/Net/memory-benchmarks-stage272/tests/test_longmemeval_prompts.py`

**Acceptance:**
- Trigger only for a question asking for work duration before a named current employer.
- Parse simple years/months durations from user-authored sentences, never assistant text.
- Require exact employer match, one unique duration pair, and the same observation date.
- Compute the difference in integer months and cite both exact user sentences.
- Reject negative/zero duration, missing evidence, conflicting values, and date mismatches.
- Run the normal answer validator before accepting a rewrite.

## Task 2: Opt-in pipeline wiring

**Files:**
- Modify `/Volumes/Net/memory-benchmarks-stage272/benchmarks/longmemeval/run.py`
- Modify `/Volumes/Net/memory-benchmarks-stage272/tests/test_longmemeval_run.py`

Add `--prior-career-tenure-resolution`, default off. Record the flag and a compact status in per-question audit and run metadata. Verify disabled mode preserves the abstention and enabled mode rewrites only when the resolver returns a validated candidate.

## Task 3: Frozen replay and full candidate gate

Use Stage268's saved Top-20 evidence and answer for `gpt4_93159ced`. Rejudge the unchanged/rewritten answer with the configured judge and seed 42. Record whether the row changes FAIL→PASS and verify no unrelated rows changed in the replay. If the focused replay passes, run all 500 questions with the feature enabled over the same frozen retrieval snapshot; compare paired outcomes and invalid audits against Stage268. Do not claim a fresh full-score gain from the one-row replay.

## Task 4: Commit and evidence

Commit the adapter code and tests with an English message. Add a Civilization scorecard and append the verified result to the dedicated Civilization project archive when the system volume permits writes. Keep memory precision and disaster-recovery claims separate from LongMemEval answer accuracy.
