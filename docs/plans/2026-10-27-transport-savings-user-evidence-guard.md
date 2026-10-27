# Transport Savings User-Evidence Guard Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's 97% LongMemEval release gate and factual precision target.

**Goal:** Prevent a transport-savings answer from treating assistant-generated fare suggestions as user-provided prices.

**Architecture:** Add a default-off guard for questions that ask how much the user would save by taking one transport mode instead of another. The guard recognizes the two modes in the question, scans only user-authored messages in the supplied Top-20 context for explicit monetary amounts tied to each mode, and checks that the answer cites user-authored amounts for both options. If either option is missing, it replaces the numeric claim with a validated abstention and records the missing modes.

**Tech Stack:** Python 3.10+, Pydantic answer models, existing LongMemEval evidence validation, pytest, configured OpenAI-compatible judge.

---

### Task 1: Add focused guard tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** Tests cover an unsupported bus-vs-taxi estimate, one-sided user price evidence, complete cited user prices for both modes, assistant-only suggestions, unrelated questions, and default-off CLI/pipeline behavior.

### Task 2: Implement the evidence guard and answerer wiring

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** A rewrite occurs only when at least one queried transport option lacks a directly cited user-authored monetary amount. It clears unsupported numeric audit state and passes the existing abstention validator. Assistant quotes remain usable for questions that explicitly ask what the assistant recommended.

### Task 3: Verify and run a frozen paired pilot

Run the focused tests and full adapter suite, compile the adapter, and run `git diff --check`. Replay every Stage262 transport-savings comparison over the same saved Top-20 evidence, judge changed answers with seed 42, and report paired outcomes.

### Task 4: Run the full 500-question candidate

Regenerate all 500 answers over the Stage262 frozen retrieval order with the guard enabled and all other settings fixed. Record paired outcomes, per-type scores, citation/audit results, and artifact provenance. Commit the adapter module with an English message; record the result in Civilization with a separate documentation commit.

## Stage263 outcome

- Adapter module commit: `8adcfa2 fix: require user fares for transport savings`.
- Adapter verification: 160 tests and 25 subtests passed; `compileall` and `git diff --check` passed.
- On the frozen Stage262 answers, only the bus-vs-taxi abstention question changed from FAIL to PASS; the train-vs-taxi question remained FAIL. Both revised outputs passed the answer validator. This is a saved-answer replay with two directly relevant questions.
- The 500-question regeneration over frozen Stage262 retrieval scored 433/500 (86.6%), 52 short of 485/500. Paired outcomes against Stage262 were 28 candidate-only and 26 baseline-only, so the 0.4 pp net change is not attributable to the narrow guard. Stage243 remains best at 438/500 (87.6%). Keep the guard opt-in.
- Scorecard: `docs/operations/data/longmemeval-stage263-transport-savings-user-evidence-guard-top20-2026-10-28.json`.
