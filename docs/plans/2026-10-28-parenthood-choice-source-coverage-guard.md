# Parenthood Choice Source-Coverage Guard Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's Top-20 97% release gate and factual-precision target.

**Goal:** Prevent unsupported temporal comparisons from assigning parenthood events to a person who is not linked to a parenthood event in user-authored evidence.

**Architecture:** Extend the existing opt-in temporal-choice audit consistency guard for parenthood-order questions. Parse the two compared people from the question, scan only user-authored source sentences for a completed parenthood event explicitly associated with each person, and abstain when an asserted comparison lacks direct source coverage for either person. Preserve valid partial citations and leave questions outside the parenthood comparison family unchanged.

**Tech Stack:** Python 3.10+, existing choice parser and citation validator, pytest, OpenAI-compatible LongMemEval judge.

---

### Task 1: Add failing guard and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** A non-abstaining “Tom or Alex” answer is changed to a validated abstention when user-authored evidence supports Alex's adoption but contains no Tom parenthood event. Both user-sourced events leave the response unchanged; assistant-only events do not count; unrelated temporal choices are unaffected.

### Task 2: Extend the existing opt-in temporal-choice guard

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** Keep the existing `--temporal-choice-audit-consistency` default-off boundary. Add direct user-source coverage for parenthood choice questions even when the model omits `temporal_events`; abstention text and retained evidence must pass the existing answer validator.

### Task 3: Replay frozen temporal answers and run a full candidate

Replay all Stage263 temporal answers, live-judge changed rows with seed 42, and report the paired denominator. Then regenerate all 500 answers over frozen Stage267 retrieval with the existing feature set and the expanded choice guard enabled. Record retrieval-order equality, per-type scores, pairwise outcomes, and artifact hashes. Commit the adapter code separately with an English message.
