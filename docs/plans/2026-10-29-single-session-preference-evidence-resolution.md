# Single-Session Preference Evidence Resolution Plan

> **For Codex:** Implement this plan task by task. Use only exact user-authored preference evidence and keep the feature disabled by default.

**Goal:** Ground single-session preference answers in directly stated, topic-matched user interests instead of generic or unrelated recommendations.

**Architecture:** Add one default-off resolver restricted to `single-session-preference`. For a small set of explicit question topics with repeated or multi-part user evidence, it will produce a concise preference summary with exact citations. It will not invent current event schedules, treat assistant suggestions as preferences, or override questions outside the recognized topic patterns.

**Tech Stack:** Python 3.10+, existing LongMemEval evidence validators, Pydantic models, unittest.

---

## Evidence and options

Stage292's four wrong preference rows have direct user evidence in Top-20:

- `35a27287`: several user requests ask for cultural/language exchange events to practice Spanish and French, but the answer only abstains because no weekend schedule is stored.
- `38146c39`: several user messages discuss the richer flavor of turbinado sugar, while the answer gives generic cookie additions without anchoring them to that ingredient.
- `57f827a0`: user messages express a plan to replace the bedroom dresser and a mid-century-modern preference, while the answer uses unrelated router advice and assistant-authored style text.
- `fca70973`: a user message explicitly requests a mix of thrill rides, special events, unique food, and nighttime shows, while the answer lists Halloween events without those decision criteria.

Options:

1. **Question-topic-gated user evidence resolver (recommended):** recognize these bounded preference topics, require the necessary exact user quotes, and answer with a cited preference summary. For current events, explicitly say the stored evidence does not contain the schedule.
2. **Expand the general prompt:** rejected because the same preferences recur in saved answers despite existing preference verification.
3. **Use assistant suggestions as preference evidence:** rejected because assistant recommendations do not prove the user's preference.

## Task 1: Add source and boundary tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Cover the four evidence patterns, exact citations, missing required dimensions, assistant-only suggestions, ambiguity, unrelated questions, default-off behavior, and strict `single-session-preference` scope.

## Task 2: Implement the opt-in preference resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--single-session-preference-evidence-resolution`. Match only the four question-topic patterns above and require the specific user-authored evidence for that topic before revising. Cite the exact source quotes, preserve uncertainty about live event schedules, and record per-row application status. Do not call another model or write durable memory.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay the four frozen Stage292 preference answers with the option and live judging; require valid citations and no saved-answer regressions. If accepted, run a fresh 500-question answerer-and-judge candidate over the Stage268 frozen Top-200 retrieval using Stage292 settings plus the resolver, then compare against Stages288 and 291.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record target replays, full-run pairings, artifact hashes, and remaining 97%, memory-precision, and disaster-recovery gaps in the bilingual scorecards and separate Civilization project archive.
