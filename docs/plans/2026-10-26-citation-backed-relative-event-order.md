# Citation-Backed Relative Event-Order Implementation Plan

> **For Codex:** Execute the tasks below in order in the authorized `memory-benchmarks` checkout.

**Goal:** Correct two-option chronology answers when user-authored source quotes support separated relative-date intervals, while preventing an unsupported relative phrase from collapsing to its embedded anchor date.

**Architecture:** Preserve exact-date ordering. When a generated event date is an unsupported relative phrase, do not parse the calendar anchor as the event date. Derive a bounded event-date interval only from a validated user-authored citation and its conversation date. Rewrite the answer only when the two event intervals do not overlap; otherwise keep the current answer unchanged.

**Tech Stack:** Python 3.10+, Pydantic answer models, existing temporal helpers, pytest/unittest.

---

## Evidence and scope

- Stage227 QID `gpt4_88806d6e` asks who was met first: Mark and Sarah or Tom.
- The cited user quotes say “a few months ago” for Tom and “about a month ago” for Mark and Sarah, both anchored to 2023-05-28.
- The answer audit marked temporal order as resolved but chose Mark and Sarah. `_parse_temporal_date("A few months before 2023-05-28; exact date not given")` falls through to the absolute-date parser and returns the anchor date, which can invert chronology.
- The current best full score is 433/500. This change is a targeted temporal correction, not a 97% claim.

## Task 1: Add regression tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Test implementation: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`

**Steps:**
1. Add a test reproducing the QID ordering from exact user quotes and `created_at=2023-05-28`; expect “Tom happened first.”
2. Add tests that overlapping intervals, missing conversation dates, invalid citations, and assistant-only quotes leave the answer unchanged.
3. Add a parser regression asserting unsupported “few months before <date>” text does not resolve to `<date>`.
4. Run `./.venv/bin/python -m pytest tests/test_longmemeval_prompts.py -k temporal_choice_order -q` and confirm the new regression fails before implementation.

## Task 2: Implement bounded interval comparison

**File:** `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`

**Steps:**
1. Make `_parse_temporal_date` return `None` when a relative marker is present but the amount/unit grammar is unsupported; never treat the embedded anchor date as the event date.
2. Add a helper that extracts a conservative interval from a validated user-authored quote and the cited memory’s `created_at` or explicit conversation-date metadata.
3. Support “a few” as a broad interval and “about/around/roughly” quantities as approximate intervals. Convert each interval to calendar-date bounds; reject missing anchors and malformed dates.
4. Require all usable citations for one event to agree on a non-empty interval. Compare the two intervals only when one ends before the other starts.
5. Update the answer and append event citations only for a disjoint, citation-supported order. Preserve the current answer for overlap, ambiguous chronology, or unsupported source roles.
6. Run the focused tests and `./.venv/bin/python -m pytest tests/test_longmemeval_prompts.py -q`.

## Task 3: Run a frozen-output pilot

**Inputs:** Stage227 four-type candidate scorecard and its matched Stage221 retrieval snapshot.

**Steps:**
1. Replay the updated resolver against all 133 frozen temporal-reasoning answers using the exact reranked Top-20 ordering from each answer audit.
2. Rejudge only answers changed by the resolver with the configured gateway and `seed=42`; retain original judgments for unchanged rows.
3. Require the target case to change to the supported earlier event, no regressions on the 133 temporal rows, and exact citation validation.
4. Record changed QIDs, score delta, verifier results, provider seed acceptance, and the fixed retrieval/prompt hashes.
5. If the pilot is non-regressing, run a fresh full 500-question Top-20 evaluation before describing any release-score improvement.

**Observed frozen-output pilot:** Two of 133 temporal answers changed. One previously wrong answer was corrected and one already-correct answer remained correct; no regression. Frozen overall diagnostic increased from 433/500 to 434/500. The full run is separate and required for release claims.

**Observed fresh full run:** Stage230 scored 427/500 (85.4%) with the new resolver; Stage227 previously scored 433/500. Across 374 reranked questions, only 2 selected orders matched, so provider variability prevents attributing the aggregate difference to this resolver. The target chronology correction persisted. The 97% gate remains unmet.

## Task 4: Verify and commit

**Steps:**
1. Run the adapter full test suite, `compileall`, and `git diff --check`.
2. Commit the adapter code/tests with an English message: `fix: resolve citation-backed relative event order`.
3. Commit the plan, ADR, pilot record, full scorecard, and updated master plan in Civilization with an English documentation commit.

## Safety boundary

Relative-language intervals are approximate evidence bounds, not invented exact dates. Any overlap or unsupported user quote leaves the answer unchanged. Do not write the provider credential to files or memory.
