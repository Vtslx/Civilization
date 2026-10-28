# Elapsed Days Between Immediate User Events Plan

> **For Codex:** Implement this plan task by task. Keep the resolver default-off and require exact user-authored event evidence.

**Goal:** Compute a days interval between a recently completed user event and a later event explicitly anchored to “today”.

**Architecture:** Add an opt-in resolver for questions of the form “how many days ago did I [event A] when I [event B]?” It must find one exact user-authored quote for each distinct event, anchor event A only from a “just” statement and its observation date, anchor event B from an explicit “today” statement and its observation date, and prove source chronology. It will write a cited interval only when the dates are unique, ordered, and within a bounded window; otherwise it leaves the answer unchanged.

**Tech Stack:** Python 3.10+, existing temporal evidence and citation models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage289 row `eac54adc` asks how many days had passed since launching the user's website when signing the first-client contract. A user quote dated 2023-02-10 says “I just launched my website”; another user quote dated 2023-03-01 says “I just signed a contract with my first client today.” The answerer returned zero days, merging the two events even though the source observations establish a 19-day separation. The same error appears in Stages286–288.

Options:

1. **Narrow two-event resolver (recommended):** map “just” and “today” to their respective user-message observation dates only for the explicitly paired events, after matching both event phrases and source ordering.
2. **Generic prompt reminder:** rejected because the error persisted through three full runs and temporal answer variation obscures the date arithmetic.
3. **Treat every “just” event as exactly its observation date:** rejected because that would overstate ambiguous recency across unrelated temporal questions.

## Task 1: Add resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the 19-day target, exact event matching, both user-authored source requirements, the “just” and “today” anchors, source order, date bounds, conflicting candidate quotes, assistant-only quotes, missing relative markers, and the default-off flag. Require both exact citations and a final validator pass.

## Task 2: Implement the default-off resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--elapsed-days-between-immediate-user-events`. Match the target and anchor clauses in a days-ago/when question to user-authored quotes. Resolve the target only from a matching “just” quote and the anchor only from a matching “today” quote. Require source dates no later than the question date, strict source order, a positive interval no greater than 60 days, and unique dates for both events. Revalidate the rewritten answer and record the effective flag and resolver audit.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay frozen Stage289 row `eac54adc` with the option and a live judge; require 0 days/FAIL to change to 19 days/PASS with valid citations. If accepted, run a fresh 500-question answerer-and-judge candidate over Stage268 frozen Top-200 retrieval using Stage288 settings plus the option; compare every question with Stages288 and 289.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the one-row replay, full candidate, hashes, and unresolved accuracy, memory-precision, and disaster-recovery gates in the bilingual scorecards and separate Civilization project archive.
