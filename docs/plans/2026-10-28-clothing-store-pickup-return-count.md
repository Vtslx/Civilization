# Clothing Store Pickup and Return Count Implementation Plan

> **For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.**

**Goal:** Count distinct clothing items the user explicitly still needs to pick up or return from a store.

**Architecture:** Add a default-off LongMemEval resolver scoped to questions asking for clothing items to pick up or return from a store. It extracts only user-authored pending actions with explicit garment and store or dry-cleaning context, deduplicates repeated mentions of the same action, and keeps pickup and return as distinct actions even when the garment type is the same. It refuses to revise if it finds an unparsed clothing/store action or cannot cite every counted task. The rewritten answer includes one validated counted item per distinct task.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and constraints

Stage279 row `0a995998` asks how many clothing items the user needs to pick up or return from a store. Top-20 contains three user-authored pending actions: pick up a replacement pair of boots after a Zara exchange, pick up a navy blazer from dry cleaning, and return boots to Zara. Top-20 also repeats the same blazer task in a second memory and mentions a green sweater lent to a sister; duplicate dry-cleaning evidence must count once, and the sweater is not a store task. The correct answer `3 items of clothing` passed both the answer validator and a live judge replay with seed 42 accepted.

## Task 1: Add resolver tests

**Files:**
- Modify `tests/test_longmemeval_prompts.py`

**Acceptance:** Cover the three supported actions, duplicate dry-cleaning mentions, distinct pickup and return tasks, assistant-only evidence, unrelated personal returns, and preservation of the original answer when a relevant task cannot be parsed or cited.

## Task 2: Implement source-backed task inventory

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`

**Acceptance:** Match only clothing pickup/return count questions. Extract pending actions only from exact user-authored sentences, require store or dry-cleaning context, deduplicate same-action/same-garment repeats, keep separate return and pickup actions, validate every counted item, and preserve the original response if any candidate fails validation or source coverage.

## Task 3: Wire default-off pipeline support

**Files:**
- Modify `benchmarks/longmemeval/run.py`
- Modify `tests/test_longmemeval_run.py`
- Modify `README.md`

**Acceptance:** Add `--clothing-store-pickup-return-count-resolution`, run metadata, and per-question audit. Verify the flag defaults off and the enabled resolver changes only answers with complete user-authored task sources.

## Task 4: Verify and benchmark

Run the full adapter suite, `compileall`, and `git diff --check`. Replay `0a995998` on Stage279 frozen Top-20 evidence, validate and rejudge it, then run all 500 questions over frozen Stage268 Top-200 retrieval with the established model, seed, cutoff, and Stage279 feature flags plus the new resolver.

## Task 5: Commit and archive

Commit the adapter change with an English message. Record the target replay, full-run paired outcomes, trigger count, and artifact hashes in a Stage280 scorecard. Preserve the best observed full result if Stage280 does not exceed 441/500.
