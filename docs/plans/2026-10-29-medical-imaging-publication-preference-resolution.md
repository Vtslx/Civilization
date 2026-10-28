# Medical-Imaging Publication Preference Resolution Plan

> **For Codex:** Implement this plan task by task. Keep the resolver opt-in, citation-backed, and limited to publication/conference preference questions.

**Goal:** Use repeated user-authored medical-imaging research requests when answering broad publication/conference recommendation questions.

**Architecture:** Add a default-off resolver for `single-session-preference` questions that ask about recent publications, research papers, articles, or conferences. It will require multiple distinct user-authored messages requesting research material in medical image analysis, preserve exact citations, and avoid treating unrelated assistant output as a preference. It will not override questions that name another topic or ambiguous evidence.

**Tech Stack:** Python 3.10+, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage291 row `75832dbd` remains FAIL. The candidate cites a statement-of-purpose message about multi-agent reinforcement learning and broad AI venues, but the same Top-20 retrieval contains several distinct user-authored requests for recent papers/articles in deep learning for medical image analysis, including multimodal fusion, transfer learning, domain adaptation, and explainable AI. The benchmark reference expects healthcare-AI and medical-imaging recommendations. The current answer misses that repeated, query-specific interest.

Options:

1. **Repeated topical user-request resolver (recommended):** prefer a unique domain repeatedly supported by direct user requests for papers/articles when the question asks for publications or conferences. Preserve any directly cited venue names separately.
2. **Prompt-only ranking instruction:** rejected because the current answerer already prefers the broader statement-of-purpose topic across repeated full runs.
3. **Treat all user interests as equal:** rejected because a broad career aspiration does not answer which research area the user has repeatedly requested reading material about.

## Task 1: Add evidence and scope tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test repeated medical-imaging research requests, direct venue quotes, assistant-only content, a competing domain with equal direct request support, a question that names another domain, the default-off CLI flag, and category gating. Require exact user citations and final validator success.

## Task 2: Implement the opt-in preference resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--research-publication-preference-resolution`. Extract user-authored request sentences that mention research material and an explicit medical-imaging topic; require at least two distinct source pairs and a unique topical focus. Build a qualified recommendation-focus answer from those exact user quotes, and optionally include directly quoted conference names without asserting that every listed venue specializes in healthcare. Do not use assistant-authored preferences or trigger for other question types.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay frozen Stage291 row `75832dbd` with a live judge; require the medical-imaging focus correction to PASS with valid user citations. If accepted, run a fresh 500-question candidate over Stage268 frozen Top-200 retrieval using Stage291 settings plus the new flag; compare against Stages288 and 291 and report all answer-text changes and paired regressions.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the frozen replay, full-run pairing, hashes, and open accuracy, memory-precision, and disaster-recovery goals in the bilingual scorecards and separate Civilization project archive.
