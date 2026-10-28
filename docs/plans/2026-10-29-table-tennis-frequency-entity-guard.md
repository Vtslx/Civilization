# Table Tennis Frequency Entity Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent a tennis schedule from being transferred to a different sport when answering a table-tennis frequency question.

**Architecture:** Add a default-off guard limited to current table-tennis frequency questions. If the answer's user-authored schedule evidence names tennis but no user-authored memory identifies table tennis or ping-pong practice, replace the frequency claim with a cited abstention explaining the sport mismatch. Preserve answers with direct table-tennis/ping-pong frequency evidence and ordinary tennis questions. Do not map sport aliases beyond the explicit table-tennis/ping-pong pair; do not add model calls or memory writes.

**Tech Stack:** Python, existing Pydantic evidence models and citation validators, LongMemEval adapter, unittest.

---

## Evidence and options

Stage283 row `f685340e_abs` asks: “How often do I play table tennis with my friends at the local park?” The answer is “Every other week,” citing a user-authored quote that says the user plans to play tennis with friends at the local park every other week. The benchmark reference says the evidence establishes tennis but not table tennis. This is a direct named-entity mismatch.

Options considered:

1. **Narrow cited-entity guard (recommended):** abstain only when a table-tennis question is answered from tennis-only frequency evidence and no user quote supports table tennis or ping-pong. This preserves the source distinction.
2. **Sport alias expansion:** infer that generic “tennis” might include table tennis. This conflates two different activities and is not supported by the user quote.
3. **Prompt-only reminder:** ask the answerer to separate sports names. This may not catch the specific transfer and has no deterministic audit status.

Keep the selected guard opt-in until a frozen replay corrects the target with no regression.

## Task 1: Add resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test tennis-only frequency evidence for a table-tennis question, direct table-tennis and ping-pong evidence, an ordinary tennis question, assistant-only claims, and unrelated frequency answers. Verify abstention citations validate, the guard is off by default, and the pipeline records its status.

## Task 2: Implement the opt-in sport-entity guard

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--table-tennis-frequency-evidence-guard`. Apply only to questions explicitly asking about table-tennis frequency when the asserted answer has user-authored tennis frequency citations and the retrieved user evidence contains no table-tennis/ping-pong frequency quote. Return a cited abstention with an audit status; preserve every ambiguous or directly supported case.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay the Stage283 saved answers with the flag and judge changed rows; require `f685340e_abs` to move FAIL→PASS with valid citations and zero regressions. If accepted, run a fresh 500-question candidate over the frozen Stage268 Top-200 retrieval with Stage283 settings plus the flag and record the trigger count and paired outcomes.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record frozen and fresh-run results, hashes, and unresolved 97%, memory-precision, and disaster-recovery goals in the bilingual scorecard and separate Civilization project archive.
