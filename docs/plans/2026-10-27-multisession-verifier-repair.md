# Cross-Session Verifier Repair Implementation Plan

> **For Codex:** Implement in the existing `memory-benchmarks` checkout. Use this plan task by task; keep the feature opt-in and discard it if the frozen pilot fails.

**Goal:** Recover supported cross-session answer revisions that the current citation or answer validators reject, without changing valid verifier results.

**Architecture:** Add one opt-in retry only for `multi-session` answer verification. Trigger it only after the first verifier response fails parsing, citation checks, or full answer validation. Include the exact validation errors and the previous response as data, request the existing schema, and rerun the same validators. Accept the retry only if every check passes; otherwise preserve the original answer and record the repair outcome. Do not add a memory write, answer field, reranker call, or global behavior change.

**Tech Stack:** Python, existing LongMemEval prompt builders, Pydantic answer and verifier schemas, existing citation and numeric validators, Stage243 frozen Top-20 results.

---

## Evidence and decision

Stage243 had 27 wrong multi-session answers. Of these, 15 had a verifier revision rejected by answer validation, one had a verifier revision rejected for an invalid quote, and one verifier returned an invalid response. The rejected numeric revisions most often had a missing `count_value`, inconsistent item sum, or incompatible units. Stage246's broad fact-ledger prompt scored 99/133 versus 103/133 control and was reverted; do not repeat broad prompt wording as the next experiment.

The repair retry is narrower than a new verifier or prompt strategy: it runs only after a failed verifier result and uses the existing validators as the acceptance boundary. Keep it disabled by default behind `--multi-session-verifier-repair`.

## Acceptance gate

- Prompt and CLI tests prove default-off behavior and multi-session-only scope.
- Unit tests prove valid initial reviews do not trigger another call; an invalid review can be repaired once; invalid repair output preserves the original answer; other question types do not use this feature.
- Focused tests, the full adapter test suite, compile check, and `git diff --check` pass.
- A frozen paired pilot covers all 133 Stage243 multi-session questions using the same saved retrieval order, model alias, and accepted seed 42. Reranking stays disabled in both arms.
- Keep and commit the code only if the candidate has a positive paired net gain and no increase in invalid answer audits, verifier invalid responses, or unresolved provider/API failures. Otherwise discard the implementation and record the negative result.

## Implementation tasks

1. Add failing tests in `memory-benchmarks/tests/test_longmemeval_run.py` for one-shot repair behavior, prompt grounding, and CLI default-off behavior.
2. Add the CLI flag, propagate it to `apply_answer_verification_review`, record it in run metadata, and add the one-retry validation path in `benchmarks/longmemeval/run.py`.
3. Run focused tests, then the complete adapter suite and compile check. Do not change benchmark retrieval or Stage243 source files.
4. Run control and candidate on the same 133 frozen Top-20 rows. Save scorecards in separate `/tmp` directories, compute pairwise outcomes and validation/error counts, then commit with an English message only if the gate passes.

## Recovery

If the retry is malformed or still fails validation, discard it and keep the original answer. If the paired pilot fails its gate, restore the adapter to its pre-experiment state; retain only the experiment record in Civilization documentation.
