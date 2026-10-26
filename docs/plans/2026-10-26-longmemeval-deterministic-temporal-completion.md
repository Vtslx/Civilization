# LongMemEval Deterministic Temporal Completion Implementation Plan

> **For Codex:** Implement this plan task by task while preserving the fixed 500-question Top-20 release gate.

**Goal:** Resolve two citation-backed temporal patterns that the answerer misses: explicit event anchors introduced by “when I …”, and trip duration when user-authored start and return events are both present.

**Architecture:** Keep the resolver pure and inside the benchmark answer adapter. Use only exact user-authored quotes and dates supported by explicit calendar text or a relative phrase tied to the cited memory’s observation date. If evidence is missing, assistant-authored, contradictory, or ambiguous, leave the generated answer unchanged. Never write synthetic facts into memory.

**Tech Stack:** Python 3.10+, existing Pydantic answer/evidence models, standard library date parsing, pytest.

---

## Evidence and decision

- Stage225 scored 430/500 (86.0%); 56 of 70 wrong answers already had all answer-marked pairs in Top-20.
- A temporal failure had a cited user quote that the baking-class event happened “yesterday” on March 21 and a separate user quote that the birthday cake was baked “today” on April 10. The model resolved the event date but used the question date instead of the cake-making anchor, producing 26 days instead of the benchmark’s 21-day interval.
- A second failure had user-authored Yosemite trip start and return quotes dated May 15 and May 17. The answerer abstained because the question phrasing (“how many days did I spend on my … trip?”) did not enter the existing event-interval resolver.
- Decision: add strict deterministic resolution from those user quotes and trusted observation dates. Do not add another model call; defer when citation or date checks are ambiguous.

## Task 1: Trust only directly supported event dates

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`

**Acceptance:** Extract an event date from an exact cited user quote only when it has a full calendar date or a supported relative term (`today`, `yesterday`, `tomorrow`, or `N days ago`) and the relative form can be anchored to the retrieved item’s observation date. Reject assistant-only, missing-date, conflicting, or ambiguous evidence.

## Task 2: Resolve an explicit “when I …” anchor

**Files:**
- Modify: `benchmarks/longmemeval/prompts.py`
- Test: `tests/test_longmemeval_prompts.py`

**Acceptance:** For “how many units ago … when I [event]” questions, prefer a uniquely matching cited anchor event derived from user-authored source text. If the question has an explicit anchor clause and no trustworthy anchor can be found, do not substitute the question date; leave the answer unresolved.

## Task 3: Resolve trip duration from start and return

**Files:**
- Modify: `benchmarks/longmemeval/prompts.py` and `benchmarks/longmemeval/run.py`
- Test: `tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`

**Acceptance:** Resolve a trip duration only when one matching user quote states the trip started and another states the user returned, both dates are supported, the destination/trip identity matches, and the order is unambiguous. Add both citations to the final answer audit. Keep the current generation unchanged for incomplete or assistant-only evidence.

## Verification and rollout

Run the targeted prompt/runtime tests and adjacent temporal tests. Replay the saved Stage225 temporal failures in a paired pilot before another full benchmark. If the deterministic pilot fixes the targeted rows without changing unrelated rows, run a fresh 500-question Top-20 scorecard. No precision or 97% claim is allowed before that scorecard passes 485/500.

**Commit:** `fix: resolve citation-backed temporal event intervals`
