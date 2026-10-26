# Count-Aware Answer Verification Implementation Plan

> Execute this plan in the authorized `memory-benchmarks` checkout. Keep the new verifier opt-in until a frozen 94-question pilot proves a net gain.

**Goal:** Let the existing answer verifier correct numeric memory answers only when it returns a complete, citation-backed count audit that passes the same production validator.

**Architecture:** Extend the structured verification response with optional count and temporal audit fields. For numeric questions, ask the verifier to independently enumerate qualifying user-supported items and report the total. Factor acceptance and rollback into a reusable helper so the regular answer pipeline and frozen-input pilot use identical validation behavior. The feature remains disabled unless selected with `--answer-verification-question-types`.

**Tech Stack:** Python 3.10+, Pydantic models, existing OpenAI-compatible `LLMClient`, pytest, LongMemEval Top-20 scorecards.

---

## Invariants and acceptance gates

- Non-numeric verification and existing defaults retain their current behavior.
- A numeric correction is accepted only with exact supplied-evidence citations, a valid item inventory or dated temporal events, compatible units, and `count_value` equal to the audited items and the numeric answer.
- Missing or invalid audit fields retain the complete original `LongMemEvalAnswer`, including citations and count metadata.
- The 94-question pilot uses Stage221's fixed retrieval snapshot and freezes the first generated answer before verifier calls. Baseline and candidate are judged against the same Top-20 evidence; unchanged answers reuse the baseline judgment.
- Promote only if the candidate has a positive paired net gain over the frozen baseline, every accepted correction passes validation, and the pilot records all fallback/rejection counts. Otherwise remove the code change and retain only the negative result.
- A 94-question pilot is not a full 500-question score. Any full-run claim requires a fresh complete Top-20 candidate and paired protocol metadata.

## Task 1: Extend the verifier response contract

**Files:** `benchmarks/longmemeval/prompts.py`, `tests/test_longmemeval_prompts.py`

1. Add optional `counted_items`, `count_value`, `count_unit`, and `temporal_events` fields to `AnswerVerificationReview`.
2. For numeric questions, require a separate inventory of distinct qualifying events/items, exact user-authored citations, time-window and status filtering, and a total whose unit matches the question.
3. Keep the existing concise answer-and-citation behavior for non-numeric questions.
4. Add prompt/schema tests for numeric audits, abstention, temporal intervals, and backward-compatible non-numeric JSON.

## Task 2: Implement strict acceptance and fallback

**Files:** `benchmarks/longmemeval/run.py`, `tests/test_longmemeval_run.py`, `README.md`

1. Factor the verification decision into a helper that accepts an existing `LongMemEvalAnswer` and frozen evidence.
2. For numeric questions, apply the verifier's complete audit fields to a candidate copy, then run `validate_longmemeval_answer`; replace the original only on success.
3. Missing fields, invalid citations, mismatched totals, and incompatible units leave the original answer untouched and set a diagnostic status.
4. Add tests for a valid corrected count, incomplete inventory, invalid citations, mismatched count totals, and unchanged non-numeric answers.
5. Document that `multi-session` verification is opt-in and adds a verifier call for numeric aggregates, which are not covered by the default cross-session verifier.

## Task 3: Run the frozen pilot

1. Select the 94 multi-session count questions from the Stage228 frozen pilot.
2. Use the Stage221 retrieval snapshot and exact Top-20 prefix; generate each baseline answer once with the current prompt and save its structured answer and audit.
3. Apply only the count-aware verifier to that frozen answer; do not rerun retrieval, reranker, or answer generation.
4. Judge each baseline once and rejudge only validated changed answers. Compare paired outcomes and report seed acceptance, accepted revisions, rejected revisions, and fallback counts.
5. If positive, run a fresh full 500-question Top-20 candidate before enabling the allowlist in any recommended configuration. If negative, restore adapter code and document the result.

## Task 4: Commit and record

Commit adapter implementation/tests/README only after a positive pilot, using the English message `feat: verify numeric answers with cited item audits`. Record the pilot and decision in Civilization with a separate English documentation commit.

## Scope boundary

This experiment targets answer synthesis for count questions. It does not change retrieval, reranker selection, stored memory, or the default verification policy. It does not by itself prove 97% accuracy or 100% factual precision.

## Stage238 frozen pilot outcome

- Input: 94 multi-session count questions from Stage228, with the Stage221 retrieval Top-20 prefix fixed and reranking disabled. The first answer and judge result were saved once; the verifier saw that exact answer and evidence, and only valid changed answers were rejudged.
- Baseline: 57/94 (60.6%). Count-aware candidate: 59/94 (62.8%). Paired outcomes: 3 candidate-only, 1 baseline-only, 56 both correct, and 34 both wrong; net +2.
- The verifier changed 32 answers; status counts were 32 `revised`, 5 `verified_count_audit`, 6 `rejected_invalid_evidence`, and 51 `rejected_unverified_revision`. Provider accepted seed 42 for answerer and judge; this does not imply general determinism.
- One regression remains: `b5ef892d` changed 8 days to 3 by incorrectly excluding a Yellowstone trip described as “last month” from “this year.” The cited answer passed the current structural validator but failed the temporal meaning of the question.
- Compact artifact: `docs/operations/data/longmemeval-stage238-count-aware-verification-frozen-pilot-2026-10-27.json`.
- Decision: commit the feature as opt-in, keep the default unchanged, and require a fresh full 500-question Top-20 candidate before recommending it. Add a calendar-year window regression guard before broad enablement. The best full score remains 433/500; Stage238 is a targeted slice, not a release score.
