# Monetary Total Audit Consistency Guard Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's 97% LongMemEval release gate and factual precision target.

**Goal:** Prevent a numeric answer from presenting a top-level money total that contradicts its validated item audit.

**Architecture:** Add a default-off post-generation guard for explicit monetary aggregate questions. It may reconcile the answer only when every counted item has an exact user-authored citation containing its amount, the validated item quantities sum to `count_value`, and the opening total claim explicitly disagrees with that sum. Ambiguous claims, invalid citations, missing quoted amounts, non-currency totals, and comparison questions remain unchanged.

**Tech Stack:** Python 3.10+, Pydantic answer models, existing currency/count helpers, pytest, existing LongMemEval judge.

---

### Task 1: Add failing count and runtime tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** A $3,000 prose total backed by exact user-cited $800+$1,200+$500 items is reconciled to $2,500; non-currency totals, unsupported item amounts, and already-consistent answers stay unchanged. Verify the option is default-off and is called after answer verification.

### Task 2: Implement the opt-in monetary total resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** Reconcile only a clear opening total claim and only when exact user-authored item amounts sum to the existing validated `count_value`; run full validation after rewriting and otherwise leave the original response untouched.

### Task 3: Verify and replay frozen evidence

Run focused tests, the complete adapter suite, compileall, and `git diff --check`. Replay Stage263 saved answers, judge every changed row with seed 42, and report the paired denominator.

### Task 4: Run a full 500-question candidate

Regenerate all 500 answers over Stage263 frozen retrieval with only this guard newly enabled. Record paired outcomes, per-type scores, invalid audits, and artifact hashes. Commit the adapter module with an English message and record the evidence in Civilization.

## Stage264 outcome

- Adapter module commit: `d611e37 fix: reconcile monetary totals with cited items`.
- Adapter verification: 165 tests and 25 subtests passed; `compileall` and `git diff --check` passed.
- On Stage263's frozen saved answers, the guard changed `36b9f61e` from FAIL to PASS: the three exact user-cited item amounts summed to $2,500 while the answer's leading total claimed $3,000. Citation validation and the live judge passed; no other saved answer changed.
- The full 500-answer regeneration over frozen Stage263 retrieval scored 431/500 (86.2%), versus Stage263 433/500 and the best Stage243 score 438/500. Paired outcomes against Stage263 were 21 candidate-only and 23 baseline-only; the full-run delta does not establish an overall gain. The target answer was already $2,500 in the fresh generation, so the guard did not trigger there. Keep the feature opt-in.
- Scorecard: `docs/operations/data/longmemeval-stage264-monetary-total-audit-consistency-top20-2026-10-28.json`.
