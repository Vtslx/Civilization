# Multi-Session Vague-Count Reconciliation Implementation Plan

> **For Codex:** Implement in the existing `memory-benchmarks` checkout. Keep the rule opt-in and remove it if the frozen pilot fails.

**Goal:** Prevent a vague repeat such as “several goals” from being added as an unknown extra to an exact user count such as “3 goals” when both refer to the same countable property.

**Architecture:** Add a narrow evidence detector and a prompt instruction gated by `--multi-session-explicit-count-priority`. It activates only for multi-session count-inventory questions when separate user-authored memory turns contain both an explicit number and a vague quantity for the same normalized noun, and that noun appears in the question. It does not infer a total or alter counted items itself. The answerer and existing completeness-review/verifier prompts are told to treat repeated vague counts as non-additive unless the user explicitly signals an additional item, new event, or updated total. Exact quote citations and existing validators remain the acceptance boundary. No new model call, schema, memory write, or default behavior change is allowed.

**Tech Stack:** Python, existing LongMemEval prompt builders, exact-citation validators, Pydantic count schemas, Stage243 frozen Top-20 retrieval.

---

## Evidence and scope

Stage243 had 22 wrong multi-session numeric-aggregate answers. A user-only scan of their saved Top-20 evidence found seven count questions with both an explicit number and a vague count phrase for the same noun across separate user-authored memories; four were wrong and three were correct. In question `4adc0475`, one user turn reported 3 soccer goals and a later turn vaguely said “several goals”; the answerer treated the vague repetition as additional unknown goals and abstained, even though the benchmark answer totals those 3 goals with 2 assists.

The prior broad fact-ledger prompt lost 4/133 and was reverted. This experiment limits wording changes to the rare exact/vague numeric-repeat pattern and leaves other memory reconciliation and counting behavior untouched.

## Acceptance gate

- Unit tests prove the detector only uses user/human turns, the same normalized count noun, and separate memory records; assistant text, unrelated nouns, and non-count question types do not trigger it.
- Prompt tests prove default-off behavior and correct propagation to answer generation, count completeness review, and verification.
- Focused tests, the full adapter suite, `compileall`, and `git diff --check` pass.
- Run control and candidate on all 133 Stage243 multi-session questions with exact saved Top-20 order, same model/gateway/seed 42, reranking disabled, and all other settings held constant.
- Keep and commit only if paired Top-20 accuracy has positive net gain with no increase in invalid answer audits, verifier invalid responses, or unresolved provider errors. Otherwise remove the implementation and record the pilot.

## Implementation tasks

1. Add failing detector, prompt-scope, and CLI tests in `tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`.
2. Implement the user-only exact/vague count detector and gated prompt instructions in `benchmarks/longmemeval/prompts.py`.
3. Add the default-off CLI flag, propagate it to answer generation/count completeness/verification, and record it in run metadata in `benchmarks/longmemeval/run.py`.
4. Run focused and full adapter tests, then the 133-question frozen A/B. Commit the code with an English message only if the gate passes.

## Recovery

If the detector cannot establish exact/vague overlap on user-authored evidence, it does not inject instructions. If the pilot fails, restore the adapter and retain only the experiment record.

## Stage250 outcome

The candidate triggered the rule on 8 of 133 multi-session questions. Both control and candidate scored 106/133 (79.70%); paired outcomes were 100 both-correct, 21 both-wrong, 6 candidate-only, and 6 control-only. Within the eight triggered questions, both arms were correct on 7 and wrong on 1, with no correctness changes. The answer-audit and verifier-response counts shifted slightly, but the accuracy gate did not pass. The adapter code was discarded without a feature commit. Artifact: `/Volumes/Net/Civilization/docs/operations/data/longmemeval-stage250-multisession-explicit-count-priority-pilot-2026-10-27.json`.
