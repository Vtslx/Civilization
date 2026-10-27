# Nonnumeric Multi-session Verifier Ledger Implementation Plan

> **For Codex:** Execute the plan in small verified steps; keep the 485/500 full Top-20 release gate unchanged.

**Goal:** Determine whether a verifier-only cross-session fact ledger helps nonnumeric multi-session questions without changing numeric answers.

**Architecture:** Scope the existing opt-in ledger prompt to `multi-session` questions for which `is_numeric_aggregate_question` is false. Keep the answer-generation prompt, numeric verifier path, output schema, retrieval, and persistence unchanged. Evaluate the 22 eligible Stage243 nonnumeric multi-session rows over three paired seeds so the result is less sensitive to one provider draw.

**Tech Stack:** Python, existing LongMemEval prompt and verifier pipeline, pytest, the configured OpenAI-compatible gateway.

---

## Evidence

Stage258's verifier-only ledger A/B scored 106/133 versus 108/133 in control. Its diagnostic split showed 2 candidate-only and 1 control-only among 22 nonnumeric questions, while numeric questions lost 3 net answers (4 candidate-only versus 7 control-only). Overall evidence rejected the broad verifier prompt. The nonnumeric signal is small and needs a dedicated paired pilot before any code is kept.

## Tasks

### Task 1: Add prompt-scope tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

- Verify the flag remains default-off and still requires the multi-session verification allowlist.
- Verify the ledger appears only for nonnumeric `multi-session` verifier prompts.
- Verify numeric multi-session prompts, answer-generation prompts, and all other question types remain unchanged.

### Task 2: Implement and verify narrow scope

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`, `memory-benchmarks/benchmarks/longmemeval/run.py`, `memory-benchmarks/README.md`

- Reuse `--multi-session-verifier-fact-ledger`, but activate the prompt only for nonnumeric multi-session questions.
- Record the per-question applied flag accurately; keep the run-level metadata flag for configuration provenance.
- Run targeted tests, the full adapter suite, `compileall`, and `git diff --check`.

### Task 3: Run a repeated frozen pilot

- Select the 22 nonnumeric multi-session questions by question type and numeric predicate, not by benchmark correctness labels.
- Copy their Stage243 Top-200 retrieval files into independent output directories on `/Volumes/Net`.
- Run paired control and candidate arms at seeds 42, 43, and 44. Keep model, judge, retrieval, Top-20 cutoff, verification allowlist, and all other options identical.
- Compare 66 question-seed pairs, audit/validation failures, structured-output retries, and exact Top-20/Top-200 order.
- Keep the option only if candidate net accuracy is positive across paired seeds with no increase in invalid answer audits or unresolved provider/schema failures. Otherwise revert and record the negative result.
- If it passes, commit with an English message, then run a fresh 500-question candidate before claiming full-score progress.

## Recovery

Every seed and arm has a separate immutable output directory. Stage243 retrieval files remain the rollback source. No credentials or raw provider secrets may be written to scorecards, memory, or commits.
