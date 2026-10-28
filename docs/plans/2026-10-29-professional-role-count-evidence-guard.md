# Professional Role Count Evidence Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent a count tied to one explicit job title from answering a question about a different job title.

**Architecture:** Add a default-off answer guard limited to questions asking how many engineers the user leads in a named professional role. When the cited user statement explicitly names a different role and gives a team size, replace the numeric answer with a citation-backed abstention. Preserve answers supported by a user-authored statement naming the requested role; ambiguous role phrases or assistant-only evidence leave the answer unchanged. Do not add model calls or memory writes.

**Tech Stack:** Python, existing Pydantic answer/evidence models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage282 row `031748ae_abs` asks how many engineers the user leads after starting a new role as Software Engineer Manager. The answerer returns “4 engineers” from a user-authored sentence saying the user leads four engineers in a new role as Senior Software Engineer. The benchmark reference says the available evidence does not establish a Software Engineer Manager role. The source is valid and numeric; the mismatch is the professional title.

Options considered:

1. **Role-mismatch abstention guard (recommended):** use exact user citations to detect a different explicit title and abstain only on the narrow engineer-count question. This directly addresses the evidence mismatch and is auditable.
2. **General title similarity resolver:** map related titles such as senior engineer and engineering manager. This could transfer responsibilities across distinct jobs and would be unsafe for personal facts.
3. **Prompt-only instruction:** tell the answerer to distinguish job titles. This is easy to add but cannot reliably catch title drift and has no deterministic audit result.

Keep the selected guard opt-in until the target replay and full-run comparison pass.

## Task 1: Add focused resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the Stage282 role mismatch and require a cited abstention with a valid answer audit. Verify that a user-authored quote naming the exact requested title preserves its count, assistant-only title claims do not establish the role, ambiguous role phrases do not rewrite answers, and unrelated engineer counts remain unchanged. Test default-off CLI behavior, enabled pipeline behavior, and per-answer audit status.

## Task 2: Implement the opt-in title-mismatch guard

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--professional-role-count-evidence-guard`. Match only engineer-count questions that explicitly ask about the user's role as a named title. Require an exact user-authored cited sentence that names a different role and states a team count, and require no user-authored cited sentence to name the requested role. The abstention must cite the mismatched user sentence and clear the stale count audit. Leave ambiguous, uncited, assistant-only, and directly supported cases unchanged.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay Stage282 row `031748ae_abs` with the guard enabled and judge the revised answer. Retain only if the target changes from FAIL to PASS with valid citations. Then run a fresh 500-question answerer-and-judge candidate over the same frozen Stage268 Top-200 retrieval using Stage282 settings plus this flag; compare every row with Stage282 and record the feature trigger count.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record target replay, full candidate, pairing, artifact hashes, and remaining accuracy, memory-precision, and disaster-recovery gates in the bilingual scorecard and separate Civilization project archive.
