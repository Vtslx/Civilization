# Assistant-Authored Evidence Scope Implementation Plan

> **For Codex:** Implement this plan task by task. Keep assistant-authored memory eligible only for the explicit assistant-output benchmark category and leave the feature disabled by default.

**Goal:** Let `single-session-assistant` answers cite exact prior assistant text without treating assistant claims as user facts.

**Architecture:** Add a default-off answerer flag and activate its relaxed count-source validation only when the dataset question type is exactly `single-session-assistant`. Add category-specific prompt guidance that treats assistant messages as evidence of what the assistant previously said, while keeping them as untrusted data and preserving user-authored evidence rules for every other category. Record the flag and its audit outcome in saved results.

**Tech Stack:** Python 3.10+, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage287 has four wrong `single-session-assistant` answers that also failed in Stages283 and 286. In `0e5e2d1a`, an exact assistant memory says the Music and Medicine study involved 38 subjects, but count inventory validation rejects its citation because it is not user-authored. In `18dcd5a5`, the assistant's retrieved one-shot contains four mummies; count inventory rejects the assistant source. In `778164c6`, the retrieved assistant recommendation includes “Grilled Snapper with Mango Salsa,” but the answer selected a later Escovitch recommendation. These questions ask about prior assistant output, not verified user facts.

Options:

1. **Question-type-gated opt-in (recommended):** enable assistant-authored evidence only when the feature flag is on and question type is exactly `single-session-assistant`. Keep all other question types on the user-authored count rule.
2. **Relax authorship globally:** rejected because assistant-generated claims could become unsupported user facts in ordinary memory questions.
3. **Infer scope from question wording:** rejected because broad phrases like “previous conversation” can refer to user facts as well as assistant output; the benchmark category is an explicit scope signal.

## Task 1: Add coverage for source-role scope

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test assistant-authored count citations rejected by default, accepted only with the flag and `single-session-assistant` type, and still rejected for user-fact questions. Verify that the default is off, the prompt explicitly distinguishes prior assistant output from user fact, the pipeline retains exact assistant citations, and the audit records the active scope.

## Task 2: Implement scoped prompt and validator plumbing

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--assistant-authored-evidence-for-assistant-questions`. Pass its enabled state through normal and evaluate-only answerer flows. When both the flag and exact question type match, include assistant-output instructions in answer and count-inventory prompts, allow exact assistant-message citations in count validation, and skip the user-only count-item filter. Do not relax general citation matching or any other question type. Record flag state and source-scope audit per result and in run metadata.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Re-run all 56 `single-session-assistant` questions over the Stage268 frozen Top-200 retrieval with Stage287 settings and the flag. Inspect target corrections, citation validity, API errors, and paired category regressions. If accepted, run a fresh 500-question answerer-and-judge candidate on the same frozen retrieval and record paired outcomes versus Stage287, trigger counts, and exact retrieval equality.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record category and full-run evidence, hashes, and the still-open 97%, memory-precision, and disaster-recovery targets in the bilingual scorecards and separate Civilization project archive.
