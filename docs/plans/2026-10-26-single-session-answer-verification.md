# Question-Type Answer Verification Implementation Plan

> **For Codex:** Execute each task in order in the authorized `memory-benchmarks` checkout.

**Goal:** Add an opt-in answer-verification allowlist so independent citation-backed review can cover question types beyond the existing multi-session, knowledge-update, temporal, and preference paths.

**Architecture:** Preserve current verification defaults. Add `--answer-verification-question-types` as an additive allowlist, thread the selected types through the answer pipeline, and run the existing verifier with its current evidence validation and rollback behavior. Invalid verifier output or unsupported citations must leave the original answer in place.

**Tech Stack:** Python 3.10+, argparse, Pydantic response models, existing LLM client, pytest/unittest.

---

## Evidence and scope

- Stage230 scored 427/500 after the relative temporal-order module; single-session user facts were 64/70 and single-session assistant facts 51/56.
- Stage231 froze all 126 single-session user/assistant answers and their Top-20 evidence. An additional verifier changed 23 answers; 4 changed from wrong to correct, none changed from correct to wrong. This is a targeted pilot, not a full-run score.
- The API key remains a runtime environment value; only the endpoint and model alias are recorded in scorecards.

## Task 1: Add CLI and unit-test coverage

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Steps:**
1. Add `--answer-verification-question-types` with `QUESTION_TYPES` choices, default `None`, and stable duplicate removal.
2. Add a CLI test for parse, deduplication, default-off behavior, and invalid values.
3. Run the new CLI test and confirm it fails before implementation.

## Task 2: Thread the opt-in types through the answer pipeline

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Steps:**
1. Extend `_generate_evidence_grounded_answer` with an additive type set; keep current cross-session and preference behavior intact.
2. Pass the set through normal and evaluate-only flows and record it in result metadata.
3. Add a fake-answerer integration test proving a selected single-session type invokes verification and only accepts a citation-valid revision.
4. Add README usage for the opt-in allowlist.
5. Run focused tests, then the full adapter suite, compileall, and `git diff --check`.

## Task 3: Verify on the full benchmark

Use the Stage221 fixed retrieval snapshot, the selected gateway/model, seed 42, Top-20 cutoff, and the four-type reranker. Enable independent verification only for `single-session-user` and `single-session-assistant` in this arm. Save all 500 answers and paired outcomes against the Stage230 candidate. Report seed acceptance and provider selection variance; do not attribute a score delta to verification unless the input hashes and protocol match.

## Task 4: Commit

Commit code/tests/README with an English message: `feat: scope answer verification by question type`. Record the pilot and full scorecard in Civilization with a separate English documentation commit.

**Observed Stage231 frozen pilot:** 23 of 126 single-session answers changed; 4 were corrected and none regressed. The slice moved 115/126→119/126.

**Observed Stage232 full run:** 432/500 (86.4%); Stage230 was 427/500. Paired outcomes were 27 candidate-only and 22 baseline-only. Single-session user moved 64/70→68/70; assistant moved 51/56→52/56. Reranker orders matched in only 3/374 rows, so the aggregate delta is confounded. The isolated frozen pilot supports keeping this feature opt-in; the best complete score remains 433/500.

## Safety boundary

The allowlist is opt-in. A verifier can revise an answer only with citations that pass existing exact-quote validation and the full answer validator; otherwise retain the generated answer.
