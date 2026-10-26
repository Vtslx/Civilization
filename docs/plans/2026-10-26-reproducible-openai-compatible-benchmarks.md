# Reproducible OpenAI-Compatible Benchmark Calls Implementation Plan

> **For Codex:** Implement this plan task by task while preserving the existing default benchmark behavior.

**Goal:** Allow benchmark answerer, verifier, reranker, and judge calls to use an explicit provider-supported seed, and record whether each OpenAI-compatible client accepted it.

**Architecture:** Keep seeding opt-in. Pass the seed to text and structured Chat Completions calls when `--openai-seed` is provided. If the provider explicitly rejects the seed field, retry without it and record that fallback; do not retry without seed for unrelated errors. Preserve temperature and prompt settings. A seed is best-effort: provider acceptance alone does not prove deterministic outputs.

Use the OpenAI-compatible gateway selected by the operator through `OPENAI_BASE_URL`; read its credential from `OPENAI_API_KEY` at runtime and never write the key to source files or result artifacts.

**Tech Stack:** Python 3.10+, existing OpenAI-compatible client, argparse, unittest/pytest.

---

## Context and decision

- Stage221/223/224 repeated full runs with `group/auto-gpt-6-luna` differed even though the same model alias, provider, prompt, dataset, and cutoff were used.
- The requested gateway `http://192.168.3.13:3425/v1` accepted `seed=42` on two identical smoke requests; both resolved to `gpt-6-luna` and returned the same text. The credential was supplied only to the process and is intentionally absent from this record.
- Add a seed option to make matched answerer/judge runs more repeatable. Keep the option unset by default; this feature does not itself improve answer quality.

## Task 1: Add seed support to the shared LLM client

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/common/llm_client.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_llm_client_seed.py`

**Acceptance:** Pass the seed for text and structured OpenAI-compatible requests; do not add the parameter when unset; if a 400/422 explicitly rejects the seed field, retry without it and expose the accepted/unsupported state. Reject invalid seed values and non-OpenAI providers.

## Task 2: Expose the seed in LongMemEval CLI and result metadata

**Files:**
- Modify: `benchmarks/longmemeval/run.py` and `README.md`
- Test: `tests/test_longmemeval_run.py`

**Acceptance:** Validate the seed range/provider, pass it to applicable answerer and judge clients, and record the requested seed plus per-client acceptance state in result metadata. Existing runs without the flag remain unchanged.

## Task 3: Verify reproducibility

Run the client unit tests and a same-prompt provider preflight twice with a fixed seed. Then run baseline and selective-reranker 500-question Top-20 evaluations with the same seed and protocol. Check per-question reranker selections as well as seed-acceptance metadata before describing the result as repeatable.

**Observed matched evaluation:** With the Stage221 retrieval snapshot held fixed, the new gateway scored 385/500 (77.0%) without reranking and 433/500 (86.6%) with the four-type reranker: 62 candidate-only and 14 baseline-only outcomes, a net gain of 48. The candidate is 52 answers below the 485/500 gate. Both arms used the same retrieval hash and prompt hash; answerer and judge metadata reported seed acceptance.

**Determinism caveat:** In a separate 94-question count-policy rerun, the gateway accepted seed 42 but selected a different Top-20 reranker ordering in all 94 questions compared with the first run. The first all-policy run is exploratory; the frozen-selection follow-up is the valid policy comparison. Keep the seed feature as a request control, not a determinism guarantee.

**Verification:** Adapter full suite: 88 tests and 20 subtests passed; Python compileall and `git diff --check` passed. Adapter commit: `cef267b feat: add reproducible seeds for OpenAI-compatible benchmarks`.

**Decision:** Keep credentials in runtime environment variables. No provider key is saved in code, benchmark JSON, or memory.
