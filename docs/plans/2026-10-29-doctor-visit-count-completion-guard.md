# Doctor Visit Count Completion Evidence Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent a planned medical consultation from being counted as a completed doctor visit.

**Architecture:** Add a default-off resolver restricted to “how many different doctors did I visit?” questions. It may remove an item described as an unnamed doctor only when its exact user-authored citation asks what to ask during a consultation before a procedure and no cited user quote confirms the encounter happened. It then recomputes the count only if the remaining doctor items have valid user citations, compatible units, and distinct identities. Ambiguous, assistant-authored, or completed-visit evidence leaves the answer unchanged. No model calls or memory writes are added.

**Tech Stack:** Python, existing Pydantic answer/count evidence models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage283 row `gpt4_f2262a51` asks how many different doctors the user visited. The answerer says four, counting Dr. Patel, Dr. Lee, Dr. Smith, and an unnamed doctor. The fourth item's only quote asks what to ask “my doctor during the consultation before the procedure”; it does not state that a visit occurred. The other three items are named and user-cited. The reference answer is three.

Options considered:

1. **Planned-consultation completion guard (recommended):** remove only an anonymous item backed by explicit future/planning language and no completed encounter evidence. This targets the observed false count with a clear user-source boundary.
2. **Broad doctor-name deduplication:** infer doctor identities from specialties and appointments. This risks merging distinct providers and cannot fix the uncompleted consultation by itself.
3. **Prompt-only reminder:** ask the count auditor to exclude plans. This depends on model behavior and is not a deterministic correction.

Keep the chosen guard opt-in until the frozen multi-session replay and fresh full run show no relevant regressions.

## Task 1: Add focused resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Cover the Stage283 planned-consultation item, a completed doctor appointment that must remain counted, assistant-only planning text, unsupported or duplicate remaining provider identities, and unrelated count questions. Verify the default-off CLI flag, pipeline propagation, recomputed count/evidence, and final validator result.

## Task 2: Implement the opt-in completion guard

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--doctor-visit-count-exclude-planned-consultations`. Apply it only to doctor-visit count questions. Exclude an anonymous item only if every relevant citation is an exact user quote with explicit consultation-planning language, and no cited user quote confirms a completed visit. Recompute only when all retained items are user-cited, unit-compatible, and distinct; otherwise preserve the original response. Record per-answer status.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay all 133 Stage283 multi-session saved answers, judge any changed rows, and require the target to move FAIL→PASS with zero regressions and valid citations. If accepted, run a fresh 500-question answerer-and-judge candidate over the same frozen Stage268 Top-200 retrieval using Stage283 settings plus this flag; compare every question with Stage283 and record the guard's exact trigger count.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the frozen replay, full candidate pairing, artifact hashes, and remaining 97%, memory-precision, and disaster-recovery gates in the bilingual scorecard and separate Civilization project archive.
