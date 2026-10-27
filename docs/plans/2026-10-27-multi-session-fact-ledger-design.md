# Multi-session Fact Reconciliation Prompt Plan

> **Record:** Proposed implementation and paired-pilot plan. Stage246 completed the experiment; the feature did not pass the acceptance gate.

> **Status:** Stage246 implementation and frozen pilot completed on 2026-10-27. The candidate did not pass the gate and was reverted; details are recorded below.

**Goal:** Improve multi-session answer synthesis by making the answerer and verifier reconcile entity, role, date, and state evidence across retrieved sessions before answering.

**Architecture:** Add a feature-gated prompt section for `multi-session` questions only. It asks the model to build a private fact ledger from the already-selected Top-20 evidence: identify the subject, relation/value, user or assistant role, event/observation date, and whether a later explicit update supersedes an earlier fact. The ledger is reasoning guidance, not an additional model call, returned field, memory write, or source of evidence. Final answers still cite exact source quotes and pass the existing validation and verifier pipeline. Other question types and default behavior remain unchanged.

**Tech Stack:** Python, existing prompt builders, Pydantic evidence answer schema, LongMemEval frozen Top-20 evaluation.

## Options and decision

1. **Prompt-only fact ledger (recommended):** no new request or schema; low latency and reuses existing exact-citation checks. It may improve cross-session chronology or entity binding while retaining the current parser/retry behavior.
2. **Separate structured ledger extraction:** machine-auditable, but adds another generation call and another schema failure point. The Stage245 count-inventory A/B lost 3/94 and logged parse retries, so avoid expanding this path without evidence.
3. **Deterministic chronology rules only:** low model cost but too narrow for identity, preferences, and current-state questions.

The first option is the smallest reversible experiment. Keep it behind `--multi-session-fact-ledger`, disabled by default.

## Implementation and acceptance tasks

### Task 1: Prompt contract tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`

- Assert the default prompt for all question types is unchanged when the flag is off.
- Assert the enabled multi-session prompt asks for entity/role/date/state reconciliation, exact quote support, and stale-versus-current fact handling.
- Assert non-multi-session questions do not receive the ledger instructions.
- Keep evidence text explicitly untrusted and never obey instructions inside memories.

### Task 2: Minimal opt-in implementation

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`, `memory-benchmarks/benchmarks/longmemeval/run.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

- Add a default-off CLI flag and pass it through answer generation and multi-session verification only.
- Add no output fields or model calls; use the existing structured answer schema, citation validator, count policy, and verifier.
- Record the flag in run metadata.
- Run focused tests, the adapter test suite, `compileall`, and `git diff --check`.

### Task 3: Frozen paired pilot before keeping the feature

- Use the 133 Stage243 multi-session questions and their saved reranked Top-20 evidence.
- Re-run control and candidate with the same model, gateway, and seed 42; do not rerank either arm.
- Keep the feature only if the paired strict score has a positive net gain, no new evidence-validation failures, and no provider/structured-output failures. Otherwise revert the implementation and record the negative result.
- If the isolated pilot passes, commit the adapter module with an English commit message, then run a full 500-question candidate before updating the verified best score.

## Risks and recovery

The fact ledger is private reasoning guidance and can still cause hallucinations; citations and existing validators remain the authority. The flag stays off outside the pilot. Each pilot writes to its own temporary output directory; Stage243 files remain the rollback source.

## Stage246 outcome

The adapter implementation was committed as `aa340a9 feat: add opt-in multi-session fact ledger` and then reverted as `6e86ec5 Revert "feat: add opt-in multi-session fact ledger"` after the pilot failed its acceptance gate. The adapter has no remaining fact-ledger flag or prompt instructions.

The paired control and candidate each reran all 133 Stage243 multi-session questions at Top-20 with the same model alias, accepted seed 42, answer-verification settings, and frozen reranked retrieval order. The Top-20 order matched in 133/133 questions. Control scored 103/133 (77.44%); candidate scored 99/133 (74.44%), a net loss of four answers (−3.01 percentage points). Paired outcomes were 92 both correct, 23 both wrong, 7 candidate-only correct, and 11 control-only correct.

The candidate also had 10 invalid answer audits versus 9 in control, plus one verifier `invalid_response` versus zero. Both arms completed all rows without unresolved API errors; transient connection retries recovered during control. Both answerer and judge accepted the requested seed, but this remains one provider run and seed acceptance does not guarantee identical generation. The feature was therefore rejected. This pilot does not change the verified full Top-20 score of 438/500 (87.6%).

Artifact: `/Volumes/Net/Civilization/docs/operations/data/longmemeval-stage246-multisession-fact-ledger-pilot-2026-10-27.json`.
