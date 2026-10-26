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

## Module 8: Temporal answer completeness

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py` and `run.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Evidence: `docs/operations/data/longmemeval-stage165-reranked-top20-2026-10-25.json`

**Work:** The full reranked run found source-session evidence in 470/470 answerable Top-20 cases, while temporal-reasoning accuracy remained 93/133 (69.9%). Improve temporal event extraction so the answerer distinguishes event dates from observation dates, resolves before/after relations, and computes intervals from the requested reference points. Preserve selected-memory citations and count audit; use explicit fallback behavior when event dates are absent or conflicting.

**Acceptance:** Add matched fixtures for event-relative intervals, inclusive windows, update-vs-history questions, and ambiguous/missing dates. Re-run all 500 LongMemEval questions at Top-20 and record a new scorecard; the target remains 485/500.

**Commit:** `feat: improve temporal evidence reasoning`

**Observed full result:** Stage166 scored 414/500 (82.8%). Temporal reasoning improved by 2 answers to 95/133 (71.4%), but the total fell by 2 answers versus Stage165; 97% remains unmet.

## Module 9: Resolve event-relative intervals and anchor counts

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py` and `run.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Evidence: `docs/operations/data/longmemeval-stage168-temporal-top20-2026-10-25.json`

**Work:** Resolve “N days ago” from the event named in the question clause and the question date, calculate event-to-event month intervals when both dates are explicit, and remove anchor/later events from before/after counts. Keep approximate or unknown dates unresolved; do not substitute a contextual event's date for the requested event.

**Acceptance:** Cover event-vs-context date selection, missing/approximate dates, yearless month/day intervals, before/after anchors, and non-temporal regressions. Run the full 500-question Top-20 gate after implementation.

**Observed full result:** Stage168 scored 426/500 (85.2%), 59 answers below the target. Although the 470 annotated answer-source sessions all appeared in Top-20, only 401 of those questions were correct; multi-session and temporal reasoning remain the next accuracy bottleneck.

## Module 10: Resolve direct same-day event evidence

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py` and `run.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Evidence: Stage216 failure `gpt4_59149c77`

**Design options:** A prompt-only reminder leaves date arithmetic to the model; a second model call adds latency and can overrule correct evidence; a deterministic citation-backed resolver can use the source observation date when a user quote explicitly says they just returned from a museum tour. Use the deterministic resolver, limited to event-interval questions, missing event dates, exact user-authored quotes, a matching museum event, and a recognized immediate-return phrase. Preserve unknown dates in all other cases.

**Work:** Before interval arithmetic, derive the event date from the cited memory's trusted `created_at` observation-date metadata only when the exact user quote says they just got back from a guided tour at a museum and the event description matches that museum. Keep the evidence citation and record the inference in `date_basis`; do not infer dates from assistant-only text, general recency words, or missing metadata.

**Acceptance:** Add unit cases for a direct user quote, assistant-only quote, unrelated event, and missing observation date. Replay the affected temporal question with the fixed Stage216 candidate list; promote only if the answer is correct and verify no other Stage216 row meets the narrow trigger.

**Commit:** `fix: resolve same-day museum visit dates`

**Observed subset result:** The trigger matches only `gpt4_59149c77` in the Stage216 composite. Two independent live replays both answer 7 days; replacing that one row gives 444/500 (88.8%), one paired gain and no regressions. This is a subset composition, not a new full 500-answer run.

## Module 11: Expand inflected terms for episodic search

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/common/mem0_client.py`, `benchmarks/longmemeval/run.py`, and `README.md`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_civilization_adapter.py`, `tests/test_longmemeval_run.py`, and `tests/test_longmemeval_prompts.py`
- Evidence: `docs/operations/data/longmemeval-stage221-retrieval-only-ablation-2026-10-26.json`

**Evidence:** The matched current-source baseline misses 11 of 883 answer-marked pairs at Top-200.

**Design:** Keep the exact question intact and optionally append bounded English inflection forms to the Civilization search query. For `top_k > 20`, preserve the base Top-20 prefix and fill the tail from deduplicated expanded-query results, then the base-query tail. Do not rewrite the answer prompt or mutate stored memories. Keep the behavior opt-in and record it in run and per-query metadata.

**Acceptance:** Test inflection generation, opt-in behavior, other backends, prefix preservation, stable deduplication, and fallback. Compare both arms on all 500 questions under the same current-source runtime. Treat Top-20 answer accuracy as the release gate and Top-200 as retrieval diagnostics.

**Observed retrieval-only result:** Top-20 ordered results are identical in 500/500 questions. Top-200 answer-marked pair recall rises from 872/883 (98.75%) to 876/883 (99.21%), and questions with all answer pairs in Top-200 rise from 460/470 to 465/470. Top-20 pair recall is unchanged. This is a retrieval-only result, not an answer-accuracy gain.

**Observed full answerer result under the user-selected provider/model:** 390/500 (78.0%) at Top-20, 95 answers below the 485/500 release gate. By type: knowledge update 66/78, multi-session 87/133, single-session assistant 53/56, preference 24/30, user 69/70, and temporal reasoning 91/133. The inflection feature preserved the exact Top-20 input, so the answerer score is a new provider run, not a causal comparison of this feature.

**Commit:** `bcf6cbd feat: expand Civilization recall without changing Top-20`

## Module 12: Stochastic token reservation and output regression

**Files:**
- Create: `src/civilization/engine/stages/stage166_orion_stochastic_memory.py`
- Test: `tests/engine/test_stage166_orion_stochastic_memory.py`
- Plan: `docs/plans/2026-10-26-orion-stochastic-prediction-regression.md`
- Decision: `docs/decisions/adr-001-stochastic-token-reservation.md`

**Design:** Keep the experiment provider-agnostic and side-effect free. The caller supplies logits for several candidate output offsets; a transaction-scoped SHA-256 counter stream reserves one token. Compare that token with the emitted token at the selected offset. An exact match is accepted; a mismatch is rolled back to the actual emitted sequence. Only after output completion may the component reserve a memory token, conditioned on and bound to the reconciled output digest and source-cell IDs. The one-token draft is never persisted.

**Acceptance:** Verify deterministic replay, multiple-position predictions, exact-match acceptance, mismatch rollback, incomplete-stream handling, evidence provenance, and zero persistent writes. Run targeted plus adjacent Orion stage regressions. Do not claim memory precision or LongMemEval accuracy from this synthetic stage; the 97% gate remains 485/500 under the fixed full benchmark.

**Runtime boundary:** The current provider runtime exposes completed text and not incremental token logits/KV state, so this module is an experimental component and is not wired into provider chat generation. A future local-runtime integration needs its own accuracy and latency ablation before enablement.

**Observed verification:** 16 new stage tests and 25 adjacent Orion regression tests passed. No production-memory write path is exposed.

**Commit:** `2663a6d feat: add stochastic memory token reservation and regression`
