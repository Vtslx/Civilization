# Composite Count Unit Compatibility Implementation Plan

> **For Codex:** Implement in the existing `memory-benchmarks` checkout. Keep the compatibility change narrow and remove it if the frozen pilot fails.

**Goal:** Allow complete totals over different discrete item categories when the question explicitly asks for one composite count, without merging measurement units such as days, hours, or currency.

**Architecture:** Extend `count_unit_compatibility_key` so labels such as `goal`, `assist`, and `goal or assist` compare as one dimensionless count only when the question is a count-inventory question that explicitly combines categories with “and” or “or”. Preserve the original display unit and existing physical-unit/currency keys. The shared key already governs answer validation, count-inventory validation, count-completeness merges, and verifier numeric audits; no new model call, schema, or memory write is added.

**Tech Stack:** Python, existing count unit normalizer/validators, LongMemEval frozen Top-20 count questions.

---

## Evidence and scope

Stage243 wrong multi-session question `4adc0475` asks for the total number of goals and assists. The answer audit contains user-cited counts of 3 goals and 2 assists, but the count-completeness path rejected a missing-item set as `incompatible_units` and the answer was judged wrong. Similar unit mismatches appear in composite item counts such as albums or EPs. Stage245 showed that broad multi-session inventory-policy changes lost accuracy, so this change is limited to compatibility checking for dimensionless item counts and keeps the existing `skip_multi_session` inventory policy.

## Acceptance gate

- Unit tests prove `goal` and `assist` compare as one count only for composite count questions, and cover other countable category pairs.
- Unit tests prove durations, distance, currency, and non-composite unit comparisons retain their existing compatibility behavior.
- Answer and inventory validation tests prove a sum of different countable categories can be validated while physical-unit mismatches still fail.
- Focused tests, full adapter suite, `compileall`, and `git diff --check` pass.
- A frozen paired pilot uses the Stage243 multi-session count-inventory question set and exact saved Top-20 order, same model/gateway/seed 42, reranking disabled, and unchanged count-inventory policy. Keep and commit only with positive paired net accuracy and no increase in invalid audits or unresolved provider failures.

## Implementation tasks

1. Add failing tests to `tests/test_longmemeval_prompts.py` for composite counts, physical units, currency, and non-composite questions.
2. Update `count_unit_compatibility_key` in `benchmarks/longmemeval/prompts.py` with a unit-family mapping guarded by the question type/structure; retain display labels.
3. Run focused and full adapter tests, then the frozen count-question A/B. Commit the code with an English message only if the gate passes.

## Recovery

If the question does not clearly ask for a composite count, or a unit has a physical dimension/currency, use the existing key. If the pilot fails, restore the adapter and keep only the result record.
