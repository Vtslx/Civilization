# Orion Memory Accuracy and Recovery Plan

**Goal:** Improve evidence retrieval and memory recovery while measuring every claim against a reproducible acceptance set and the full LongMemEval result.

**Architecture:** Keep the in-memory Orion API and durable journal as the production path. Add a replaceable, dependency-free retrieval strategy with observable score components; preserve the existing Helix approved-weight signal. Keep session isolation and global-memory opt-in as policy boundaries. Extend persistence with verifiable backup and restore artifacts rather than changing journal replay semantics.

**Tech Stack:** Python 3.10+, standard library, pytest, existing Orion persistence and benchmark interfaces.

---

## Baseline and acceptance

- Reference LongMemEval baseline: 349/500 (69.8%) at Top-20. The reproducible scorecard measures source-session chunk recall separately from answer accuracy; do not describe session presence as proof that the exact answer fact was retrieved.
- Keep the same dataset, model, answerer, judge, write adapter, and Top-20 cutoff when comparing a new full run. Record retrieval coverage, MRR, answer accuracy, question type, API errors, and run parameters separately.
- Report the 97% target only if the same 500-question protocol measures at least 485 correct answers. Synthetic retrieval fixtures are regression gates, not substitutes for that result.
- Run targeted unit and service tests for each module, then the full repository suite when its engine dependencies are available. Validate backup restore in a separate temporary state directory.
- Preserve source data and credentials outside the repository. Never write private endpoint configuration or API keys into plans, fixtures, logs, or commits.

## Module 1: Fielded relevance ranking

**Files:**
- Create: `src/civilization/engine/stages/stage161_orion_relevance_ranking.py`
- Modify: `src/civilization/engine/stages/stage73_orion_memory_kernel.py`
- Test: `tests/engine/test_stage161_orion_relevance_ranking.py`
- Document: this plan and the active Orion retrieval design note

**Work:** Replace set-overlap scoring with deterministic fielded BM25 ranking over summary, content, and source. Add an explicit recency tie-break only when the query asks for current/latest information. Keep importance, confidence, and approved Helix link weights bounded so they cannot outweigh strong query evidence. Keep existing `read()` parameters and trace fields compatible, and expose enough score diagnostics to explain why a cell ranked.

**Acceptance:** Verify exact and paraphrased query fixtures, long irrelevant assistant text, duplicate/common terms, empty queries, filters, explicit recency, stable ordering, and learned-weight compatibility. Measure precision@20, recall@20, and MRR on a fixed set; do not label it LongMemEval.

**Commit:** `feat: add fielded relevance ranking for Orion memory`

## Module 2: Cross-tier evidence ranking

**Files:**
- Modify: `src/civilization/engine/stages/stage79_orion_global_retrieval.py`
- Test: `tests/engine/test_stage79_orion_global_retrieval.py`
- Document: `docs/operations/orion-relevance-ranking.md` and its Chinese counterpart

**Work:** Compare session and opted-in global results on the same relevance scale, with tier priority only as a deterministic tie-break. Keep global retrieval opt-in, session IDs isolated, resolved-global policy enforced, and all retrieval choices auditable.

**Acceptance:** Include strong-global/weak-session and strong-session/weak-global fixtures, disabled-global isolation, conflict-policy exclusions, and deterministic tie cases. Re-run the Orion service integration tests.

**Commit:** `fix: rank opted-in global evidence by relevance`

## Module 3: Disaster recovery

**Files:**
- Create: a versioned backup bundle and verifier under `src/civilization/engine/persistence/`
- Modify: persistence CLI/service integration and `docs/operations/session-memory-durability.md` plus the Chinese translation
- Test: persistence export, corruption detection, atomic restore, and interrupted-restore cases

**Work:** Export a consistent snapshot of session and global stores with a versioned manifest and cryptographic checksums. Verify the complete bundle before restore; stage restore into a temporary location and atomically promote it. Retain the existing local journal as the fast crash-recovery path and document that disaster recovery still requires an independent backup destination.

**Acceptance:** Verify round-trip cell/link/metadata equality, corrupted and truncated artifact rejection, safe handling of unknown format versions, and recovery after an interrupted promotion. Do not claim disk-loss protection until an independent copy has been configured and tested.

**Commit:** `feat: add verified Orion memory backup and restore`

## Module 4: Full-score evaluation and release gate

**Files:**
- Create: `src/civilization/engine/stages/stage162_orion_accuracy_scorecard.py`
- Modify: `src/civilization/cli.py`
- Test: `tests/engine/test_stage162_orion_accuracy_scorecard.py`
- Document: `docs/operations/longmemeval-scorecard.md` and its Chinese counterpart

**Work:** Add an auditable scorecard for full-data runs that separates answer accuracy from evidence recall, ranking quality, abstention, latency, and service errors. Compare each release candidate to the fixed 500-question baseline without changing prompt, model, data, or cutoff mid-run.

**Acceptance:** A release may claim 97% only at 485/500 or better on the same full benchmark; otherwise publish the measured score and error breakdown and keep the target open.

**Commit:** `test: add reproducible Orion memory accuracy scorecard`

## Module 5: Speaker-aware evidence ranking

**Files:**
- Create: `src/civilization/engine/stages/stage163_orion_role_aware_ranking.py`
- Modify: `src/civilization/engine/stages/stage161_orion_relevance_ranking.py`, `stage73_orion_memory_kernel.py`, and `stage79_orion_global_retrieval.py`
- Test: `tests/engine/test_stage163_orion_role_aware_ranking.py`
- Document: `docs/operations/orion-relevance-ranking.md` and its Chinese counterpart

**Work:** When conversation cells preserve speaker labels or typed speaker metadata, index user and assistant text as separate fields. First-person personal questions prefer user statements; explicit assistant-recommendation questions prefer assistant turns. Unstructured cells keep neutral BM25F ranking. Preserve the raw memory and source metadata; ranking only changes candidate order.

**Acceptance:** Cover user fact vs. assistant advice, assistant recommendation queries, metadata-only speaker labels, unstructured fallback, Helix weights, and the full native memory regression slice. Run a fresh full Top-20 scorecard before retaining any claimed accuracy gain.

**Commit:** `feat: rank conversational memory by speaker provenance`

## Module 6: Temporal date coverage

**Files:**
- Create: `src/civilization/engine/stages/stage164_orion_temporal_coverage.py`
- Modify: `src/civilization/engine/stages/stage73_orion_memory_kernel.py` and `stage79_orion_global_retrieval.py`
- Test: `tests/engine/test_stage164_orion_temporal_coverage.py` and the Stage79 integration test
- Document: `docs/operations/orion-relevance-ranking.md` and its Chinese counterpart

**Work:** For sequence, temporal, or multi-event queries, preserve the BM25-selected candidate set but place the highest-relevance cell from each observation date before same-date repeats. Non-temporal and latest/current queries keep their existing order. Apply the same coverage rule after session/global fusion when global memory is enabled.

**Acceptance:** Verify date coverage, relevance fallback, latest-first behavior, cross-tier handling, resolved-global policy, and full service regression. Run a full Top-20 scorecard before retaining any accuracy claim.

**Commit:** `feat: diversify temporal memory evidence by date`

## Module 7: Bounded source-session coverage through the benchmark adapter

**Files:**
- Create: `src/civilization/engine/stages/stage165_orion_source_session_coverage.py`
- Modify: `stage164_orion_temporal_coverage.py`, `stage73_orion_memory_kernel.py`, and `stage79_orion_global_retrieval.py`
- Test: `tests/engine/test_stage165_orion_source_session_coverage.py` and the Stage79 integration test
- Benchmark adapter: persist source-session/pair provenance and preserve Civilization's server order through top-k selection

**Work:** When benchmark cells include a source-session ID, advance a new source conversation by at most two ranks ahead of a nearby repeat for temporal or multi-event queries. Scope session IDs by source user when available. Keep relevance candidate selection and latest/current query behavior intact; retain Stage164 date coverage when source-session metadata is absent.

**Acceptance:** Verify same-date sessions remain represented within the bounded lookahead, rank displacement never exceeds two places, latest-first ordering is unchanged, missing provenance preserves the Stage164 fallback, and session/global fusion respects source-session diversity. Run the fixed 500-question Top-20 scorecard before recording any accuracy gain.

**Commit:** `feat: diversify temporal evidence by source session`
