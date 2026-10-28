# Judge Verdict Format Retry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent malformed yes/no judge responses from being silently counted as incorrect while preserving strict verdict parsing.

**Architecture:** Parse only a final yes/no verdict outside reasoning tags. Represent a missing or malformed verdict as invalid, retry that judge request once with a format-only reminder, and fail closed if the retry is still malformed. Do not infer correctness from chain-of-thought text or modify the answer, retrieval, or memory state.

**Tech Stack:** Python, async `LLMClient`, unittest/pytest, existing OpenAI-compatible judge provider.

---

## Evidence and scope

Stage268 has one malformed judge response among 500 Top-20 rows: `2ebe6c92` answers “The Nightingale by Kristin Hannah,” matching the expected title and author, while the judge response says that it is the same book inside `<judge_thinking>` and omits the required final `yes` or `no`. `LLMClient._parse_yes_no_judgment` reads only the text after the closing reasoning tag and currently maps the missing verdict to `False`. The response format contract in `get_judge_prompt` explicitly requires a separate verdict after that tag.

The change is in `memory-benchmarks`, not answer generation. Valid judge responses must make exactly one request and preserve existing parsing. Only malformed responses receive one additional request. Persist the raw response sequence for audit. This does not assert a benchmark gain until the affected saved answer is rejudged.

## Approaches considered

1. Read `yes` or `no` from inside `<judge_thinking>`: rejected because reasoning text is not the requested final verdict and can disagree with it.
2. Retry only when the final verdict is absent or malformed: selected because it preserves the explicit output contract and leaves valid runs untouched.
3. Keep mapping malformed output directly to `FAIL`: preserves the current fail-closed behavior but incorrectly conflates a protocol error with a negative correctness judgment.

## Task 1: Strict verdict parsing and one format retry

**Files:**
- Modify `benchmarks/common/llm_client.py`
- Create `tests/test_llm_client_judge.py`

**Acceptance:**
- `_parse_yes_no_judgment` returns `True` or `False` only for an explicit final verdict; otherwise it returns `None`.
- `judge_yes_no` makes no extra call for a valid verdict.
- A missing/invalid verdict receives exactly one format-only retry; a valid retry determines the result.
- A second malformed response returns `False` and preserves both raw responses for audit.
- A `yes` or `no` only inside `<judge_thinking>` is never accepted as the final verdict.

## Task 2: Focused regression and saved-row rejudge

Run the new judge-client tests and adjacent LLM client tests. Rebuild the Stage268 judge prompt for `2ebe6c92` from its saved question, expected answer, generated answer, and question date, then rejudge only that row with the existing configured model/provider and seed. Record both the first malformed response and the retry response. Keep the answer text and retrieval snapshot unchanged.

If the retry returns an explicit verdict, update only the diagnostic replay and scorecard evidence. Do not rerun or rewrite all 500 judgments from a single row. If the retry remains malformed or returns `no`, record no score gain and retain the fail-closed result.

## Task 3: Commit and record evidence

Commit the adapter implementation and tests with an English message. Add a Civilization scorecard documenting the denominator, the isolated row, retry result, and whether the score changed. Update only the existing Civilization project archive after the scorecard is verified.
