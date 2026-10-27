# Temporal Event Alias Deduplication Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's 97% LongMemEval release gate.

**Goal:** Deduplicate repeated descriptions of the same source-backed event before ordering a temporal event list.

**Architecture:** Extend the opt-in temporal event-list resolver with a conservative source identity check. It may merge two event records only when validated user quotes come from adjacent pairs in the same source session, refer to the same typed event and named location, and use an anaphoric definite article for one mention. Preserve all citations and only mutate the answer after every remaining event has a validated, disjoint time interval.

**Tech Stack:** Python 3.10+, Pydantic models, pytest, existing LongMemEval replay runner.

---

### Task 1: Add focused failing tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`

**Acceptance:** A same-session adjacent `a music festival in Brooklyn` / `the music festival in Brooklyn` pair is treated as one event; separate sessions, separate locations, or non-anaphoric mentions are not merged; unresolved inputs leave the answer unchanged.

### Task 2: Implement conservative event identity consolidation

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py` only if audit metadata needs a new status field.

**Acceptance:** The resolver keeps its existing opt-in boundary, uses exact user-authored citations and source session/pair metadata, retains all merged provenance, and applies no rewrite if interval ordering remains ambiguous or conflicting.

### Task 3: Verify the adapter and run frozen evidence replay

Run the focused prompt tests and complete adapter suite. Replay all 133 Stage261 temporal answers with frozen Top-20 evidence, enabling both event-list ordering and the existing recent-event interval option. Judge changed answers against the configured provider; preserve all unchanged rows and report the paired denominator.

### Task 4: Run the full 500-question candidate and record evidence

Run all 500 questions over the same frozen Stage243 retrieval order and existing Stage261 feature set, changing only the new event consolidation behavior plus its required recent-event interval flag. Compare per-question outcomes against Stage261 and Stage243, retain artifacts, and commit the adapter module with an English message.
