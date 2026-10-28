# Explicit Education Year-Span Resolver Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's 97% LongMemEval release gate and factual precision target.

**Goal:** Resolve an education-duration answer from explicit user-authored start and completion years when the model incorrectly adds only selected degree durations.

**Architecture:** Add a default-off answer resolver limited to questions explicitly asking for formal education from high school through a Bachelor's completion. It must find one user-authored high-school year range and one user-authored Bachelor's completion year, reject conflicting or missing endpoints, and compute only the calendar span between those years. The output retains both exact source citations and a structured count item so the normal validator remains authoritative.

**Tech Stack:** Python 3.10+, Pydantic answer models, existing citation and numeric validators, pytest, configured LongMemEval judge.

---

### Task 1: Add failing resolver and runtime tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** A source with a user quote stating high school 2010–2014 and a user quote stating Bachelor's graduation in 2020 resolves to an approximately 10-year span; assistant-only dates, conflicting endpoints, missing endpoints, and unrelated duration questions stay unchanged. Verify the feature is default-off and the answerer audit passes the full validator.

### Task 2: Implement and wire the opt-in resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** Resolve only when both endpoint years are unique, exact, user-authored, and linked to the requested education milestones. Preserve citations, emit a validated count item with the elapsed year span, and leave ambiguous cases untouched.

### Task 3: Verify and replay frozen answers

Run focused tests and the complete adapter suite, compileall, and `git diff --check`. Replay Stage264 saved answers, judge each changed answer with seed 42, and report the paired denominator.

### Task 4: Run a 500-question candidate

Regenerate all 500 answers over Stage264 frozen retrieval with only this resolver newly enabled. Record paired outcomes, per-type scores, citations, and artifact hashes. Commit the adapter module with an English message, then record the result in Civilization.
