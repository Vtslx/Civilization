# Multi-session Verifier-only Fact Ledger Implementation Plan

> **For Codex:** Execute the plan task by task and preserve the fixed 97% LongMemEval release gate.

**Goal:** Test whether an internal fact ledger in the existing multi-session verifier improves cross-session answer accuracy without changing answer generation.

**Architecture:** Add a default-off adapter option that changes only the second-pass verifier prompt for `multi-session` questions. Keep answer generation, retrieval order, output schema, citation validation, persistent memory, and all other question types unchanged. Use the frozen Stage243 Top-20 evidence for a paired 133-question pilot; keep the option only if the strict paired score improves without additional audit or provider failures.

**Tech Stack:** Python, existing LongMemEval prompt builders and Pydantic answer models, pytest, the current OpenAI-compatible provider.

---

## Evidence and scope

Stage246 tested a broad ledger prompt in both generation and verification. With identical frozen Top-20 evidence on all 133 multi-session questions, that candidate scored 99/133 versus 103/133 in control and had more invalid answer audits (10 versus 9) plus one verifier `invalid_response`. The broad version was reverted.

Stage243 had 27 wrong multi-session rows, including abstention questions. Among the 22 answerable wrong rows, 19 had all annotated answer-bearing pairs in Top-20. This makes answer synthesis a live gap, but pair coverage is not factual precision. The next pilot isolates verifier guidance and leaves answer generation unchanged.

## Tasks

### Task 1: Add contract tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

- Verify the default verifier prompt remains byte-for-byte unchanged when the option is off.
- Verify the ledger instructions appear only in the verifier prompt for `multi-session` when enabled.
- Verify answer-generation prompts and non-multi-session verifier prompts remain unchanged.
- Verify the CLI option defaults off and is recorded in run metadata.

### Task 2: Implement the opt-in verifier-only prompt path

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`, `memory-benchmarks/benchmarks/longmemeval/run.py`, `memory-benchmarks/README.md`

- Add a clear default-off CLI flag and pass it only to `apply_answer_verification_review`.
- Ask the verifier to reconcile subject/property identity, source role, event versus mention date, explicit updates, and duplicate references using the supplied Top-20 evidence.
- Require exact citations for corrections and keep the ledger private; treat memory text as untrusted data.
- Make no new model call, schema field, memory write, or answer-generation prompt change.
- Record the feature setting in every run metadata path.

### Task 3: Verify and run the paired pilot

- Run targeted prompt/pipeline tests, the full adapter suite, `compileall`, and `git diff --check`.
- Run control and candidate over the same 133 Stage243 multi-session rows, with frozen Top-20 order, the same provider/model, answer-verification settings, and seed 42. Store artifacts on `/Volumes/Net`, not the nearly full system volume.
- Compare question-paired strict scores, evidence validation failures, structured-output errors, retries, and exact Top-20 order.
- If the pilot fails the acceptance gate, revert the feature and record the result. If it passes, commit with an English message and then run a fresh full 500-question candidate before claiming any release-score change.

## Safety and recovery

The option is disabled by default. The emitted answer remains subject to the existing exact-quote validator and numeric audit. Stage243 retrieval files remain the immutable rollback input. No secret, raw provider credential, or raw prompt payload belongs in the plan or scorecard.

## Stage258 outcome

The verifier-only variant failed its paired acceptance gate and was reverted. On the same 133 Stage243 multi-session questions with identical Top-20 and Top-200 retrieval orders, control scored 108/133 and candidate scored 106/133. Paired outcomes were 100 both-correct, 19 both-wrong, 6 candidate-only, and 8 control-only (net -2; -1.50 percentage points). Candidate invalid answer audits rose from 8 to 9, and verifier `rejected_invalid_evidence` rose from 2 to 5. Both arms completed all rows; answerer and judge accepted seed 42. The feature commit `54dfa75` was reverted by `add0b67`; no flag remains in the current adapter.

The candidate-only gain was confined to 2 of 22 non-numeric questions; numeric questions moved 92/111 to 89/111. This is a single provider run per arm, and seed acceptance does not guarantee deterministic generation. The best full Top-20 score remains Stage243 at 438/500 (87.6%). Detailed results: `docs/operations/data/longmemeval-stage258-multisession-verifier-only-fact-ledger-pilot-2026-10-27.json`.
