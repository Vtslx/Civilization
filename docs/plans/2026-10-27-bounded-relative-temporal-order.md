# Bounded Relative-Date Ordering Implementation Plan

> **For Codex:** Implement in the existing `memory-benchmarks` checkout. Keep the additional resolver opt-in and discard it if the frozen temporal pilot fails.

**Goal:** Correct two-option temporal ordering when two exact user-authored relative-date quotes share the same observation anchor and their conservative `about` intervals meet at only one boundary day.

**Architecture:** Preserve the current conservative date-interval resolver. Add an opt-in fallback in `resolve_temporal_choice_order` that applies only after ordinary interval comparison cannot decide. It must verify that both event options map to distinct citation-backed events; each event has a user-authored numeric relative-duration quote; both quotes use the same observation date and unit; the approximate durations differ by exactly one unit; and their computed intervals overlap on exactly one day. In that single boundary case, the larger “ago” duration is earlier. Every other ambiguous or unsupported case remains unchanged. No new model call, output field, or memory write is added.

**Tech Stack:** Python, existing citation and temporal interval helpers, LongMemEval frozen Top-20 retrieval.

---

## Evidence

Stage243 question `gpt4_213fd887` is a concrete error. The user asked whether a volleyball league or charity 5K came first. Both user quotes were observed on 2023-05-24: “about two months ago” for the volleyball league and “about a month ago” for the 5K. The answer reversed them. Existing bounds reduce to March 10–April 9 and April 9–May 9, so the current strict disjoint-interval check declines to resolve the shared boundary day. Keep the proposed fallback limited to this exact evidence pattern.

## Acceptance gate

- Unit tests prove the shared-boundary case resolves to the longer “ago” duration as earlier.
- Negative tests prove different anchors, different units, non-user quotes, ambiguous event matching, broad/unequal intervals, and overlap wider than one day remain unresolved.
- CLI default-off and answer-pipeline propagation tests pass.
- Focused tests, the complete adapter suite, `compileall`, and `git diff --check` pass.
- A frozen paired pilot covers all 133 Stage243 temporal questions, keeps the exact saved Top-20 order in both arms, disables reranking, and changes only the new flag. Keep and commit only with positive net accuracy and no new validation/API failures; otherwise discard.

## Implementation tasks

1. Add failing resolver and CLI/pipeline tests in `tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`.
2. Implement a strict quote-to-duration extractor and opt-in shared-boundary resolver in `benchmarks/longmemeval/prompts.py`.
3. Add `--temporal-relative-duration-order-resolution`, pass it through both run paths, and record it in metadata.
4. Run focused and full adapter tests, then the paired 133-question temporal pilot from Stage243 retrieval files. Commit the code with an English message only if the gate passes.

## Recovery

If citation, role, anchor, unit, or boundary checks fail, keep the model's answer unchanged. If the paired pilot misses the gate, remove the implementation and retain only the experiment record.
