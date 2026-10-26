# Cited Multi-Event Order Resolver Implementation Plan

> Execute this plan in the authorized Civilization and memory-benchmarks checkouts; preserve the default answer path until a positive frozen pilot.

**Goal:** Correct temporal list-order answers only when every event has a user-cited date interval that establishes an unambiguous total order.

**Architecture:** Add a pure, model-agnostic resolver over `TemporalEventEvidence` and retrieved memories. Derive dates only from exact user-authored citations and observation metadata, validate the event count and unique descriptions, then rewrite the ordered list only when adjacent intervals are strictly disjoint. Expose it through an opt-in benchmark flag and record per-question resolver status.

**Tech Stack:** Python, Pydantic temporal evidence, existing citation/date helpers, LongMemEval frozen scorecards.

---

## Invariants and gates

- No second generation or reranker call in the isolated pilot.
- Every event date used for sorting must be supported by an exact supplied user quote and unambiguous observation metadata.
- Overlapping, missing, duplicate, assistant-only, or count-mismatched event evidence leaves the original answer untouched.
- A rewritten answer includes every expected event exactly once and retains evidence citations.
- Keep the resolver opt-in unless the frozen pilot has positive paired net gain and zero regressions; a full 500-question Top-20 candidate is still required before any release claim.

## Task 1: Add pure resolver tests

**Files:** `benchmarks/longmemeval/prompts.py`, `tests/test_longmemeval_prompts.py`

1. Test sorting three user-authored events with explicit dates.
2. Test a bounded “before observation date” event sorting ahead of later exact dates.
3. Test rejection for overlapping intervals, missing dates, duplicate descriptions, wrong event count, and assistant-only citations.
4. Verify every output event retains an exact citation and non-order questions are unchanged.

## Task 2: Add opt-in runtime plumbing

**Files:** `benchmarks/longmemeval/run.py`, `tests/test_longmemeval_run.py`, `README.md`

1. Add `--temporal-event-list-order-resolution`, disabled by default.
2. Apply the pure resolver after answer verification and before final validation/judging.
3. Record a per-question audit status (`resolved`, `unchanged_ambiguous`, or `not_applicable`) and run-level flag.
4. Test default-off behavior, accepted sorted output, and conservative fallback.

## Task 3: Run a frozen-output pilot

Use Stage239's 133 temporal answers and frozen Top-20 provenance. Replay only the resolver, rejudge only changed answers, and compare paired outcomes. Reject the code change if any correct answer regresses or if the pilot has no net gain; otherwise commit the adapter feature and run a full 500-question Top-20 candidate.

**Stage240 result:** Two of 133 temporal answers were resolvable. One already-correct three-trip answer stayed correct; the airline-order answer was corrected. Temporal accuracy moved 109/133→110/133, and the frozen full-score diagnostic moved 422/500→423/500. Paired resolver outcomes were 1 candidate-only, 0 baseline-only, and 1 both-correct, with no regressions. The adapter implementation is committed as `d1dc923 fix: order cited multi-event temporal answers`. Artifact: `docs/operations/data/longmemeval-stage240-temporal-list-order-frozen-pilot-2026-10-27.json`.

## Task 4: Commit and document

The isolated pilot passed its correction/no-regression gate. Stage241 then reran answer generation, reranking, and judging over saved Stage221 retrieval results; the frozen raw retrieval order matched in 500/500 questions. The candidate scored 433/500 (86.6%), tying Stage227 and not improving the best full score. Against Stage227, paired outcomes were 25 candidate-only and 25 reference-only. The reranker selected exactly the same Top-20 order as Stage239 in 130/500 rows, with mean ID overlap 16.658/20, so the +11 paired delta versus Stage239 is not attributable to the resolver. In Stage241 the resolver fired twice: one wrong order was corrected and one already-correct order remained correct. Keep the feature opt-in. Full scorecard: `docs/operations/data/longmemeval-stage241-temporal-list-order-full-top20-2026-10-27.json`.

## Scope boundary

This resolver only orders events already represented in the answer audit. It does not retrieve omitted events, infer dates from vague language, or establish 97% accuracy or 100% factual precision by itself.
