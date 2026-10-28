# Assistant Recommendation Clue Resolution Plan

> **For Codex:** Implement this plan task by task. Keep the resolver opt-in and limited to assistant-output questions with a unique clue-matched assistant list item.

**Goal:** Select the exact prior assistant recommendation that matches all distinguishing clues in a `single-session-assistant` question.

**Architecture:** Add a default-off resolver that runs only when `--assistant-authored-evidence-for-assistant-questions` is also enabled and the question type is `single-session-assistant`. It will parse numbered or bulleted items from explicitly assistant-authored messages, match the question's named food anchor and fruit clue, and revise the answer only when exactly one distinct item matches. It cites the exact assistant list line and leaves user facts and other question types unchanged.

**Tech Stack:** Python 3.10+, existing evidence models and validators, LongMemEval adapter, unittest.

---

## Evidence and options

Stage288's independent 56-question slice corrected `778164c6` to “Grilled snapper with mango salsa,” but the fresh 500-question run and Stage290 returned “Escovitch Fish.” The Top-20 contains an assistant-authored list with both options: one line describes Escovitch Fish as a Jamaican dish, while another names Grilled Snapper with Mango Salsa and describes a fruity salsa. The question asks for the prior recommendation with snapper and fruit; the answerer selects the later alternative instead of the unique item matching both clues.

Options:

1. **Unique exact assistant-list match (recommended):** match the named food anchor plus explicit fruit clue inside one assistant-authored list item; abstain from rewriting when zero or multiple items match.
2. **Prompt-only reminder:** rejected because the tuned assistant-output prompt passed the independent 56-question slice but failed to reproduce in the full run.
3. **Another model call:** rejected because the source list already has a unique literal match and a second generation call adds cost and variance.

## Task 1: Add source and ambiguity tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the mango-snapper item, no matching assistant item, multiple matching fruit dishes, user-only text, the default-off flag, and refusal to run outside `single-session-assistant` or without the parent assistant-evidence flag. Require exact assistant citation validation.

## Task 2: Implement the bounded resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--assistant-recommendation-clue-resolution`. Parse list items only from explicit assistant message blocks. For questions matching “name of … recommended … with [anchor] … fruit”, require the unique item line to contain the exact anchor and a concrete fruit term. Preserve the source role, clear unrelated structured fields, validate the exact citation, and record the resolver status. Do not select from user echoes, prose outside a list item, or ambiguous matches.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay Stage290 row `778164c6` with both assistant flags and a live judge; require Escovitch/FAIL to change to the exact mango-snapper item/PASS with valid citation. If accepted, run a fresh 500-question answerer-and-judge candidate over Stage268 frozen Top-200 retrieval using Stage290 settings plus both flags; compare against Stages288 and 290.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the target replay, full-run pairing, hashes, and remaining 97%, memory-precision, and disaster-recovery gaps in the bilingual scorecards and separate Civilization project archive.
