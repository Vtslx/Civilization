# Property Offer Anchor Count Resolver Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user's 97% LongMemEval release gate and factual-precision target.

**Goal:** Avoid counting the named target property as one of the other properties viewed before making an offer on it, while stating the count convention clearly.

**Architecture:** Add a default-off resolver for property-count questions phrased as viewing properties before an offer on a named target property. It may remove one item from the other-property count only when exactly one counted item matches the target's distinctive name/location terms, has a user-authored citation that mentions both the target and the offer, and its audit says it was viewed before that offer. The answer will state the number of other properties and clarify the total if the target itself is included. Ambiguous or incomplete item audits remain untouched.

**Tech Stack:** Python 3.10+, Pydantic count audits, existing exact-quote evidence validator, pytest, configured LongMemEval judge.

---

### Task 1: Add failing resolver and runtime tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

**Acceptance:** A unique cited Brookside townhouse target is excluded from the four-other-property count and the answer clarifies that including the target makes five. Missing, duplicate, or weak target matches leave the original answer unchanged; the default remains off.

### Task 2: Implement the opt-in property anchor resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

**Acceptance:** Rewrite only after validating the target's unique identity, user-authored offer/view evidence, item units, audit sum, and final answer. Preserve citations for the four alternatives and the target property; keep the answer auditable.

### Task 3: Replay frozen evidence and run a full candidate

Apply the resolver to all Stage268 saved answers, live-judge every changed answer with seed 42, and report the paired denominator. Then regenerate all 500 answers over frozen Stage268 retrieval with the resolver enabled and other settings unchanged. Record retrieval-order equality, per-type accuracy, pairwise outcomes, and artifact hashes. Commit the adapter module with an English message and record the outcome in Civilization.
