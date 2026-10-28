# Hotel Recommendation Preference Resolution Plan

> **For Codex:** Implement this plan task by task. Keep the resolver default-off, user-source-only, and destination-agnostic.

**Goal:** Answer hotel recommendation questions using explicit, reusable user hotel preferences when the requested destination has no saved property recommendation.

**Architecture:** Add an opt-in resolver for single-session preference questions that ask for a hotel for an upcoming trip. It will combine an exact user quote requesting a view with a separate exact user quote naming a unique hotel amenity. It may summarize those filters and state that the supplied memories contain no destination-specific hotel; it must not invent property names, availability, or current facilities.

**Tech Stack:** Python 3.10+, existing citation models and validators, LongMemEval adapter, unittest.

---

## Evidence and options

Stage293 row `0edc2aef` remains FAIL. The generated answer says the memories provide no Miami hotel. The retrieved user memories do contain reusable hotel criteria: the user asks for a great city view, says they like a rooftop pool or hot tub on a balcony, and later asks for a hot tub with a city-skyline view. These criteria are from Seattle hotel searches, so they support preference filters but do not establish a Miami property.

Options:

1. **Cited preference summary (recommended):** report the user's view and amenity preferences, note that no Miami listing appears in the retrieved memories, and avoid naming a hotel.
2. **Transfer a Seattle hotel to Miami:** rejected because the destination and property are different.
3. **Search current hotel inventory:** rejected for this adapter module because no live lodging-search source is connected and the benchmark answer can be grounded without inventing a listing.

## Task 1: Add preference and boundary tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test a uniquely cited city-view and amenity pair, assistant-only evidence, missing view or amenity evidence, conflicting criteria, other destinations, non-hotel questions, default-off behavior, and the single-session-preference type scope. Require exact user citations and validator success.

## Task 2: Implement the default-off hotel preference resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--hotel-recommendation-preference-resolution`. Require a hotel question, one user-authored view quote, and a separate user-authored quote for a rooftop pool or balcony hot tub. Return a cited preference summary with an explicit destination-listing limitation; do not add hotel names or claim that a Seattle option exists in Miami.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay Stage293 row `0edc2aef` with a live judge; require the stored-answer abstention to become a cited preference summary/PASS with no frozen-slice regressions. If accepted, run a fresh 500-question candidate over Stage268 frozen Top-200 retrieval using Stage293 settings plus the resolver and compare against Stages291 and 293.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the target replay, full-run pairings, hashes, and unresolved 97%, memory-precision, and disaster-recovery gates in the bilingual scorecards and separate Civilization project archive.
