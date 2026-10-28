# Unverified Master's Education Span Guard Implementation Plan

> **For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.**

**Goal:** Prevent an answer from claiming a total through Master's completion when retrieved user evidence only establishes lower-degree education and says the Master's is merely being considered.

**Architecture:** Add a deterministic, default-off guard in the LongMemEval adapter. It applies only to questions explicitly asking for formal-education duration from high school through completion of a Master's degree, only when the answer asserts a numeric year total, and only when no user-authored source confirms completion of that degree. When the evidence shows intent to pursue the degree but no completion, replace the unsupported total with a cited abstention and validate it. Assistant-authored or hypothetical degree claims do not count as completion evidence.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and constraints

Stage277 row `gpt4_372c3eed_abs` asks for formal education from high school through completion of a Master's degree. Stage268 correctly abstains: the user evidence supports high school, an Associate's degree, and a Bachelor's degree, while the only Master's reference says the user is considering one. Stage277 answered `10 years` and failed. The current education-year resolver was enabled but returned `not_applicable`; the regression came from answer generation, not that resolver.

The cleaned 500-question dataset contains one question whose endpoint is completion of a Master's degree. A separate temporal question mentions a Master's thesis but asks for the interval from an undergraduate degree to thesis submission; it must not trigger this guard.

## Task 1: Add resolver-level tests

**Files:**
- Modify `tests/test_longmemeval_prompts.py`

**Acceptance:** Test that an unsupported numeric high-school-to-Master's total becomes a cited abstention; explicit user-authored Master's completion leaves the candidate unchanged; assistant-only completion does not count; unrelated degree questions remain unchanged.

## Task 2: Implement the narrow guard

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`

**Acceptance:** Match only high-school-to-Master's-completion questions with a numeric year answer. Inspect exact user-authored source sentences. Preserve the answer if one explicitly confirms the user's Master's completion; otherwise abstain only when relevant user-authored education evidence can be cited. Validate the rewritten answer and keep the original response unchanged if validation fails.

## Task 3: Wire the opt-in pipeline

**Files:**
- Modify `benchmarks/longmemeval/run.py`
- Modify `tests/test_longmemeval_run.py`
- Modify `README.md`

**Acceptance:** Add default-off `--unverified-masters-education-span-guard`, a per-question audit, and run metadata. Test default-off behavior and the enabled abstention on the Stage277 row. Ensure the temporal Master's-thesis interval question is unaffected.

## Task 4: Verify and run the candidate

Run the full adapter suite, `compileall`, and `git diff --check`. Replay the Stage277 saved target with the guard enabled and rejudge it; then run all 500 answers over frozen Stage268 Top-200 retrieval using the established Stage277 model, seed, cutoffs, and feature flags. Record paired results and whether the guard fires in fresh generation.

## Task 5: Commit and archive

Commit the adapter change with an English message. Add a separate Stage278 scorecard and update the current best score and Civilization project archive only after all 500 rows and artifact hashes are verified.
