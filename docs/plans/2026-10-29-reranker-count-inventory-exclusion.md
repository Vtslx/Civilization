# Reranker Count-Inventory Exclusion Plan

> **For Codex:** Implement this plan task by task. Keep the new behavior opt-in and commit the adapter module in English only after the frozen and full-run gates are reviewed.

**Goal:** Let the LLM reranker remain active for selected weak question types while preserving the original Top-20 order for questions that require a complete count inventory.

**Architecture:** Add an optional `--llm-reranker-skip-count-inventory` policy to the existing question-type reranker. When enabled, `should_use_llm_reranker` will apply the existing type allowlist and also reject numeric inventory questions before any reranker call. The default behavior and non-count questions remain unchanged. Audit metadata will show when the reranker was skipped for this reason.

**Tech Stack:** Python 3.10+, the existing LongMemEval runner and reranker audit, pytest, Stage268 frozen Top-200 result files.

---

## Evidence and alternatives

Stage297 selectively reranked 266 `multi-session` and `temporal-reasoning` questions and scored 456/500, a new best by 4 over Stage291. It also changed Top-20 order on 228/266 reranked questions. On target `88432d0a`, the reranked answer counted only 3 baking events and failed; the boundary resolver removed one event at the 14-day cutoff, but the selected evidence omitted another event. Stage295 without reranking had the same target pass at 4. This is direct evidence that reranking an inventory question can hide a qualifying item.

Approaches considered:

1. **Disable reranking globally:** avoids count omissions but discards Stage297's gains on weaker non-count multi-session/temporal questions.
2. **Skip count-inventory questions when opted in (recommended):** preserves the Stage297 configuration for narrative/time questions and the source order for full-inventory counts. This protects the observed failure with a narrow condition.
3. **Increase reranker selection size beyond 20:** rejected because it would violate the fixed Top-20 evaluation gate and still would not guarantee complete count evidence.

## Task 1: Add selection-policy tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test that the new flag defaults off, leaves ordinary reranking behavior unchanged when off, skips numeric inventory questions only when on, and preserves type allowlist behavior for non-count questions.

## Task 2: Implement and plumb the opt-in exclusion

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Pass the question text and the new option into both evaluate-only and normal reranker selection paths. Record the setting in full-run metadata and per-question reranker audit. Skip only when `is_count_inventory_question(question)` is true; do not change retrieval data or count-completeness behavior.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay Stage297 target `88432d0a` using frozen Top-200 candidates and require the reranker audit to show skipped for count inventory while preserving the original Top-20 candidate order. If valid, run a fresh 500-question candidate over Stage268 frozen Top-200 with Stage297's two-type reranker configuration and the new exclusion. Compare with Stages291, 295, and 297; report paired results, selection-order changes, audit validity, and retry/API errors.

## Task 4: Commit and archive

Keep the guard only if the target is protected without cancelling non-count reranking and the full run provides a useful paired outcome. Commit with an English message. Record the score and unresolved 97%, memory-precision, disaster-recovery, and Stage166 efficacy gates in the bilingual scorecards and separate Civilization project archive.
