# Multi-Session Count Adjudication Implementation Plan

> **For Codex:** Execute each task in order in the authorized `memory-benchmarks` checkout.

**Goal:** Add an opt-in, citation-validated adjudication pass that reconciles a direct answer draft and an independent count inventory for selected numeric question types.

**Architecture:** Preserve the current answer path by default. For an allowlisted count question, generate a direct answer draft and a separate inventory draft, then ask a third structured call to reconcile both against the same retrieved memories. Accept the reconciled count only when its item citations, units, count value, and answer all validate; otherwise fall back to the direct answer and existing completeness review.

**Tech Stack:** Python 3.10+, Pydantic response models, argparse, existing OpenAI-compatible client, pytest/unittest.

---

## Evidence and scope

- Stage228 froze the same 94 multi-session count questions and Top-20 order under two policies: `skip_multi_session` scored 71/94, `all` scored 69/94.
- A frozen Stage236 adjudication pilot compared both drafts plus the same Top-20 evidence. It scored 73/94: 5 wins and 3 losses versus skip. It is a pilot, not a full benchmark result.
- Do not change the default count policy. The new adjudicator is opt-in and restricted to `is_count_inventory_question` rows in the selected question types.

## Task 1: Add prompt and unit tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`

**Steps:**
1. Add `get_count_adjudication_prompt` that includes the original question, both candidate inventories as untrusted drafts, and all numbered memories.
2. Require the adjudicator to independently enumerate distinct user-supported items, respect time windows and role/status constraints, deduplicate repeated event mentions, and abstain when required evidence is missing.
3. Add tests for candidate-draft inclusion, data-not-instruction handling, exact citation requirements, and the expected Pydantic response shape.
4. Run the new prompt tests and confirm they fail before implementation.

## Task 2: Add the opt-in runtime path

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Steps:**
1. Add `--count-adjudication-question-types`; default to an empty list and deduplicate selected types.
2. For selected count-inventory questions, build direct and inventory drafts, run adjudication, normalize and validate inventory citations/units, convert it to the final answer, and run the existing full answer validator.
3. If the adjudicator abstains, returns invalid evidence, or fails count/answer validation, retain the direct draft and run the normal completeness-review path.
4. For an accepted complete adjudication, record the result and skip a redundant third completeness audit; preserve the existing count inventory and completeness behavior when the option is unset.
5. Record allowlisted types and per-question adjudication status in run metadata/audits.
6. Add tests for default-off behavior, successful adjudication, and invalid-response fallback. Run focused tests, full suite, compileall, and diff-check.

## Task 3: Verify on frozen and full inputs

1. Build the same 94 multi-session count questions with the Stage227 selected Top-20 order frozen and run `--count-adjudication-question-types multi-session` against the selected gateway and seed 42.
2. Compare paired results with Stage227 skip and Stage228 all; require valid citations for every accepted adjudication and report all fallback counts.
3. If the implementation pilot improves the matched 94-question slice, run a fresh 500-question Top-20 candidate with the option enabled for `multi-session` and compare against Stage232. Record provider seed acceptance and reranker-order variance; do not attribute a full-run delta to adjudication unless the input protocol matches.

## Task 4: Commit and document

Commit adapter code/tests/README with an English message: `feat: adjudicate multi-session count candidates`. Record the frozen pilot, full scorecard, and decision in Civilization with a separate English documentation commit.

## Safety and precision boundary

Never select between draft totals by majority alone. The adjudicator must cite exact supplied memory text, honor user-authored evidence for personal counts, and refuse unsupported partial totals. Invalid or ambiguous adjudications fall back to the existing validated path.

## Stage237 outcome and decision

- Frozen 94-question runtime result: 68/94, compared with 71/94 for Stage227 `skip_multi_session` and 69/94 for Stage228 `all`.
- Paired outcomes against skip: 4 adjudicator-only, 7 skip-only, 64 both correct, 19 both wrong. Against all: 5 adjudicator-only, 6 all-only, 63 both correct, 20 both wrong.
- The selected Top-20 memories matched Stage227 in the same order for all 94 questions, isolating the result from retrieval selection changes.
- The implementation accepted 43 adjudications, fell back for invalid evidence on 34, and fell back for unanswerable adjudications on 17.
- The prior Stage236 offline replay of pre-generated drafts scored 73/94 but did not reproduce in the runtime path.
- Decision: reject the runtime implementation and retain `skip_multi_session`. The uncommitted adapter changes were removed after this pilot, so no feature-code commit was made. The compact scorecard is `docs/operations/data/longmemeval-stage237-count-adjudication-frozen-implementation-pilot-2026-10-26.json`.
- Stage237 raw metadata omitted the provider base URL. It recorded `openai`, the answerer/judge aliases, and seed acceptance; do not infer the exact endpoint from this run.
