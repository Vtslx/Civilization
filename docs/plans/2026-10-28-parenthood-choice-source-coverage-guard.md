# Parenthood Choice Source-Coverage Guard Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's Top-20 97% release gate and factual-precision target.

**Goal:** Prevent unsupported temporal comparisons from assigning parenthood events to a person who is not linked to a parenthood event in user-authored evidence.

**Architecture:** Extend the existing opt-in temporal-choice audit consistency guard for parenthood-order questions. Parse the two compared names from the question and scan only user-authored source sentences for those names. If one named option is absent from all user-authored evidence but the answer asserts an order, abstain and retain valid partial citations. Leave questions outside the parenthood comparison family unchanged.

**Tech Stack:** Python 3.10+, existing choice parser and citation validator, pytest, OpenAI-compatible LongMemEval judge.

---

### Task 1: Add failing guard and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** A non-abstaining “Tom or Alex” answer is changed to a validated abstention when user-authored evidence mentions Alex but never mentions Tom. Assistant-only mentions of Tom do not count; when both names appear in user-authored text, unrelated temporal-event logic remains responsible for comparing them.

### Task 2: Extend the existing opt-in temporal-choice guard

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** Keep the existing `--temporal-choice-audit-consistency` default-off boundary. Add named-option source coverage for parenthood choice questions even when the model omits `temporal_events`; abstention text and retained evidence must pass the existing answer validator.

### Task 3: Replay frozen temporal answers and run a full candidate

Replay all Stage263 temporal answers, live-judge changed rows with seed 42, and report the paired denominator. Then regenerate all 500 answers over frozen Stage267 retrieval with the existing feature set and the expanded choice guard enabled. Record retrieval-order equality, per-type scores, pairwise outcomes, and artifact hashes. Commit the adapter code separately with an English message.

## Stage268 outcome

- Adapter module commit: `fca8995 fix: abstain on unnamed parenthood comparison options`.
- Adapter verification: 173 tests and 25 subtests passed; `compileall` and `git diff --check` passed.
- On Stage267 saved answers, `gpt4_fe651585_abs` changed from FAIL to a validated abstention because user-authored Top-20 text did not mention Tom; the judge returned PASS, with zero saved-answer regressions (431/500→432/500).
- The full 500-answer regeneration over frozen Stage267 retrieval scored 440/500 (88.0%), the best observed full score so far and two above Stage243's 438/500. However, the guard itself fired on zero rows in the fresh generation because the model independently abstained on the target. The 27 candidate-only and 18 baseline-only paired outcomes span 271 changed answers, so the +9 full-run delta is not attributed to this guard.
- Scorecard: `docs/operations/data/longmemeval-stage268-parenthood-choice-source-coverage-top20-2026-10-28.json`.
