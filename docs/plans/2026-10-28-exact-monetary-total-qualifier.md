# Exact Monetary Total Qualifier Resolution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove an unsupported lower-bound phrase from a monetary answer when the audited user-cited item amounts establish an exact total.

**Architecture:** Extend the existing opt-in monetary-total consistency resolver. It will accept an exact amount only when a user-authored quote states that amount without a lower-bound or approximation qualifier. If every selected item has such an exact citation and the selected sum matches `count_value`, a leading “more than”, “over”, or “at least” claim is normalized to the exact total and revalidated. If any item is supported only by a lower bound, the answer remains unchanged.

**Tech Stack:** Python, Pydantic answer audits, existing LongMemEval monetary resolver, unittest/pytest.

---

## Evidence and scope

Stage275 row `d851d5ba` has a validated audit totaling $3,750 across four user-cited fundraiser items. Each item has an exact user quote matching its amount, though some of the same facts also appear with “over” wording. The generated answer says “More than $3,750” and is judged FAIL against `$3,750`. The current monetary resolver skips it because `_leading_monetary_total` does not recognize lower-bound prefixes.

## Task 1: Exact quote verification and qualifier normalization

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`
- Test `tests/test_longmemeval_prompts.py`

**Acceptance:**
- Recognize a lower-bound qualifier only when it prefixes the leading currency amount.
- Lower-bound and approximate currency mentions do not count as exact item-value evidence.
- Reconcile only when every selected item has an exact user-authored amount citation, currencies match, and item sum equals `count_value`.
- Normalize a matching lower-bound total to the exact audited total, then pass the shared answer validator.
- Lower-bound-only, approximate-only, missing, conflicting, or mismatched audits remain unchanged.

## Task 2: Frozen replay and full candidate

Replay Stage275 row `d851d5ba` and rejudge only the corrected answer using the configured judge and seed 42. If it passes with zero replay regressions, run a full 500-answer candidate over frozen Stage268 retrieval with the established Stage275 resolver configuration.

## Task 3: Commit and record

Commit adapter code and tests with an English message. Add a Civilization scorecard separating the isolated replay from the fresh full candidate and preserving the current best full score unless the full run verifies an improvement.
