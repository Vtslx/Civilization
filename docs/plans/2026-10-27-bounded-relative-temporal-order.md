# Bounded Relative-Date Ordering Implementation Plan

> **For Codex:** Implement in the existing `memory-benchmarks` checkout. Keep the additional resolver opt-in and discard it if the frozen temporal pilot fails.

> **Status:** Stage249 implementation and frozen replay completed; code was committed opt-in. The full 500-question run did not invoke the new fallback and did not improve the verified score.

**Goal:** Correct two-option temporal ordering when two exact user-authored relative-date quotes share the same observation anchor and their conservative `about` intervals overlap by at most one day.

**Architecture:** Preserve the current conservative date-interval resolver. Add an opt-in fallback in `resolve_temporal_choice_order` that applies only after ordinary interval comparison cannot decide. It must verify that both event options map to distinct citation-backed events; each event has a user-authored numeric relative-duration quote; both quotes use the same observation date and unit; the approximate durations differ by exactly one unit; and their computed intervals overlap by at most one elapsed day (two inclusive date values are possible because calendar-date rounding is discrete). In that narrow boundary case, the larger “ago” duration is earlier. Every other ambiguous or unsupported case remains unchanged. No new model call, output field, or memory write is added.

**Tech Stack:** Python, existing citation and temporal interval helpers, LongMemEval frozen Top-20 retrieval.

---

## Evidence

Stage243 question `gpt4_213fd887` is a concrete error. The user asked whether a volleyball league or charity 5K came first. Both user quotes were observed on 2023-05-24: “about two months ago” for the volleyball league and “about a month ago” for the 5K. The answer reversed them. Existing bounds reduce to March 8–April 9 and April 8–May 9, which overlap by one elapsed day; the current strict disjoint-interval check declines to resolve it. Keep the proposed fallback limited to this exact evidence pattern.

## Acceptance gate

- Unit tests prove the one-day-overlap case resolves to the longer “ago” duration as earlier.
- Negative tests prove different anchors, different units, non-user quotes, ambiguous event matching, broad/unequal intervals, and overlap wider than one day remain unresolved.
- CLI default-off and answer-pipeline propagation tests pass.
- Focused tests, the complete adapter suite, `compileall`, and `git diff --check` pass.
- A frozen answer replay covers all 133 Stage243 temporal questions and keeps the exact saved Top-20 order. Keep only if a saved wrong answer is corrected, no saved correct answer regresses, and the candidate passes existing validation and the benchmark judge.
- Also run a fresh 500-question Top-20 candidate with the flag enabled; record it separately and do not attribute its aggregate score unless the per-question audit confirms the fallback ran.

## Implementation tasks

1. Add failing resolver and CLI/pipeline tests in `tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`.
2. Implement a strict quote-to-duration extractor and opt-in shared-boundary resolver in `benchmarks/longmemeval/prompts.py`.
3. Add `--temporal-relative-duration-order-resolution`, pass it through both run paths, and record it in metadata.
4. Run focused and full adapter tests, then the frozen 133-question answer replay. If it passes, commit with an English message, run the full 500-question candidate, and preserve the feature as opt-in until a full-score gain is measured.

## Recovery

If citation, role, anchor, unit, or boundary checks fail, keep the model's answer unchanged.

## Stage249 outcome

The resolver was committed as `019b31b feat: resolve adjacent relative temporal order`. In the frozen 133-question answer replay, only `gpt4_213fd887` changed: the answer moved from the charity 5K to the volleyball league as the earlier event, passed answer validation, and passed the benchmark judge. Replay accuracy moved 114/133→115/133, with one candidate-only correction, zero baseline-only regressions, and 132 unchanged rows.

The full 500-question candidate scored 429/500 (85.8%), below Stage243's 438/500. It reused the same Top-20 retrieval order in 500/500 rows, but the per-question audit showed the new fallback applied zero times; this full rerun is not evidence that the resolver caused the score delta. Keep the feature opt-in; do not claim a full-score gain. Stage243 remains the best verified full score. Artifact: `/Volumes/Net/Civilization/docs/operations/data/longmemeval-stage249-relative-temporal-boundary-pilot-2026-10-27.json`.
