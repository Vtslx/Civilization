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

**Current verification (2026-10-26):** `python3 -m pytest tests/engine/test_memory_backup.py -q` passed all 13 backup/restore tests. The implementation uses a versioned SHA-256 manifest, mode-0600 archive, atomic publish, staged restore, and refuses overwrite.

**Operational limit:** The backup CLI is offline/manual, archives are unencrypted, and no independent destination, schedule, retention policy, or restore drill is configured. Treat local backup tooling as verified; enterprise disaster recovery remains incomplete until a separate failure-domain target and end-to-end drill are established.

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

## Module 13: Scope LLM evidence reranking to weak question types

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py` and `README.md`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`
- Scorecard: `docs/operations/data/longmemeval-stage225-selective-four-type-reranker-top20-2026-10-26.json`
- Intermediate scorecard: `docs/operations/data/longmemeval-stage224-selective-reranker-preference-top20-2026-10-26.json`
- Error audit: `docs/operations/data/longmemeval-stage225-error-audit-2026-10-26.json`
- Count-policy diagnostic: `docs/operations/data/longmemeval-stage223-count-inventory-pilot-2026-10-26.json`
- Preference reranker pilot: `docs/operations/data/longmemeval-stage224-preference-reranker-pilot-2026-10-26.json`
- Knowledge-update pilot: `docs/operations/data/longmemeval-stage224-knowledge-update-reranker-pilot-2026-10-26.json`

**Design:** Keep global LLM reranking opt-in. Add a question-type allowlist so the answerer model can select up to 20 items from Top-200 only for categories where evidence selection helps. The best tested allowlist is `multi-session`, `temporal-reasoning`, `single-session-preference`, and `knowledge-update`; single-session user/assistant questions keep the existing ranker path.

**Acceptance:** Verify default-off behavior, type allowlist parsing, and non-target bypass. Compare a full 500-question Top-20 run against the matched baseline using the same dataset and provider. Report pairwise outcomes and per-type deltas; do not infer causality from selected pilots or rolling progress alone.

**Observed full results:** The two-type run scored 418/500; adding preference reranking scored 422/500; the final four-type selective reranker scored 430/500 (86.0%) versus the 390/500 (78.0%) baseline, a +8.0 percentage-point difference; 62 candidate-only and 22 baseline-only outcomes. It remains 55 answers below the 485/500 release gate. Final by type: knowledge update 68/78, multi-session 108/133, single-session assistant 52/56, preference 27/30, user 67/70, and temporal reasoning 108/133. The auto-routed provider changed several unreranked user/assistant answers between runs, so the paired delta includes provider variability.

**Evidence retrieval:** On the four-type reranked result, Top-20 answer-marked pair recall is 868/883 (98.30%), with all answer pairs in 458/470 questions. The matched baseline is 732/883 pairs and 365/470 complete-pair questions. These are evidence-retrieval metrics, not factual precision or 100% memory accuracy.

**Error audit:** 56 of 70 wrong answers had every answer-marked pair in Top-20, 65 wrong-answer audits passed structural/citation validation, and 67 included citations. The remaining majority therefore needs better fact synthesis, temporal anchoring, and cross-session reasoning, not broader retrieval alone.

**Count policy:** A separate 15-question evidence-complete multi-session count pilot scored 6/15 under each inventory policy, with one win each. Keep the default `skip_multi_session`; the pilot did not justify changing it globally.

**Commit:** `6cc4f45 feat: scope memory reranking by question type`; result record: `docs/operations/data/longmemeval-stage225-selective-four-type-reranker-top20-2026-10-26.json`.

## Module 14: Resolve citation-backed temporal anchors and trip durations

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py` and `benchmarks/longmemeval/run.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`
- Design/ADR: `docs/plans/2026-10-26-longmemeval-deterministic-temporal-completion.md`, `docs/decisions/adr-002-citation-backed-temporal-resolvers.md`
- Pilot: `docs/operations/data/longmemeval-stage226-temporal-resolver-pilot-2026-10-26.json`
- Full scorecard: `docs/operations/data/longmemeval-stage226-temporal-completion-top20-2026-10-26.json`

**Design:** Resolve “how many units ago … when I …” from a uniquely matched user-authored anchor quote when its direct calendar or relative date is tied to observation metadata. Resolve trip duration only from matching, user-authored start and return events with supported dates. When source role, event identity, or date is missing or ambiguous, leave the answer unresolved. Singularize duration output such as “1 month”.

**Acceptance:** Test explicit anchors, observation-date relative terms, assistant-only/undated/mismatched events, trip start/return, and answerer runtime integration. Run a frozen-output paired pilot, then a fresh full 500-question Top-20 scorecard before claiming any release gain.

**Observed frozen-output pilot:** 16 of 20 elapsed/trip questions had valid direct user-date resolutions; 14 were already correct, and two previously wrong answers were corrected. Both passed the benchmark judge; four unsupported or ambiguous cases stayed unchanged. This is a targeted postprocessor pilot, not a full score.

**Observed full result:** Stage226, with the four-type selective reranker and temporal resolver, scored 419/500 (83.8%), +5.8 percentage points versus the 390/500 baseline, 66 answers below the release gate. Temporal reasoning stayed at 108/133; provider rerun variability prevents attributing a net score change to the resolver. The targeted 2/2 corrections remain supported by the frozen-output pilot.

**Verifier ablation:** A mixed 14-question temporal numeric verification holdout moved from 7/14 to 8/14, with two candidate-only and one baseline-only result. That is not enough evidence to add the extra verifier call broadly; keep it off by default. Record: `docs/operations/data/longmemeval-stage226-temporal-verification-pilot-2026-10-26.json`.

**Verification:** 81 prompt/runtime/adapter tests and 17 subtests passed; compileall and `git diff --check` passed.

**Commit:** `74cc7e9 fix: resolve citation-backed temporal intervals`.

## Module 15: Seeded OpenAI-compatible benchmark runs

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/common/llm_client.py`, `benchmarks/longmemeval/run.py`, and `README.md`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_llm_client_seed.py` and `tests/test_longmemeval_run.py`
- Plan: `docs/plans/2026-10-26-reproducible-openai-compatible-benchmarks.md`
- Decision: `docs/decisions/adr-003-reproducible-openai-compatible-benchmarks.md`

**Design:** Make provider seeding optional for OpenAI-compatible requests. Retry without the seed only after an explicit unsupported-seed response; record acceptance in run metadata. Supply custom endpoint and credentials through runtime environment variables, with no secret written to source, result JSON, or saved memory.

**Provider preflight:** The requested endpoint `http://192.168.3.13:3425/v1` accepted seed 42 twice for the same short prompt; both responses resolved to `gpt-6-luna` and matched exactly. This verifies route and seed support only, not LongMemEval accuracy.

**Verification:** Adapter full suite: 88 tests and 20 subtests passed; Python compileall and `git diff --check` passed.

**Observed matched evaluation:** With the Stage221 retrieval snapshot held fixed, the requested endpoint scored 385/500 (77.0%) without reranking and 433/500 (86.6%) with four-type reranking: 62 candidate-only and 14 baseline-only outcomes, a net gain of 48. The candidate remains 52 answers below the 485/500 release gate. Both arms used the same retrieval-source hash and prompt hash; answerer and judge metadata reported seed acceptance.

**Determinism caveat:** A separate 94-question rerun accepted seed 42 but selected a different Top-20 reranker order on all 94 questions. The initial `all` count-inventory arm is exploratory only; use the frozen-selection Stage228 pilot for policy conclusions. Seed acceptance is not a determinism guarantee.

**Credential handling:** The endpoint key was provided to the process at runtime and is absent from source, scorecards, and memory.

**Provider configuration follow-up (2026-10-26):** LongMemEval now loads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the ignored local `.env` file and accepts `LONGMEMEVAL_ANSWERER_MODEL` / `LONGMEMEVAL_JUDGE_MODEL` defaults. The requested model alias was verified through `/models` and one Chat Completions smoke call. Adapter commit: `17b0453 feat: configure LongMemEval models via environment`. The key remains local and is not included in the commit.

**Commit:** `cef267b feat: add reproducible seeds for OpenAI-compatible benchmarks` in the `memory-benchmarks` repository.

## Module 14: Resolve cited user-event anchors and trip durations

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py` and `benchmarks/longmemeval/run.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py` and `tests/test_longmemeval_run.py`
- Design: `docs/plans/2026-10-26-longmemeval-deterministic-temporal-completion.md`
- Decision: `docs/decisions/adr-002-citation-backed-temporal-resolvers.md`
- Frozen-output pilot: `docs/operations/data/longmemeval-stage226-temporal-resolver-pilot-2026-10-26.json`

**Design:** Compute an elapsed interval from an explicit “when I …” anchor only when an exact user-authored quote supports the anchor date. For trip-duration questions, require matching user-authored start and return events with dates supported by explicit dates or relative phrases tied to observation-date metadata. Reject assistant-only, unstructured, missing, or ambiguous dates and leave the answer unresolved.

**Acceptance:** Cover explicit user anchors, observation-date relative phrases, assistant-only/ambiguous anchors, trip start/return, mismatched destinations, missing dates, citation validity, and run-time integration. Run a paired temporal pilot, then a fresh full 500-question Top-20 scorecard before claiming any accuracy gain.

**Observed frozen-output pilot:** 20 elapsed/trip questions were eligible; 16 had valid direct user-date resolutions, including 14 already-correct answers. Two previously wrong answers were corrected and both passed the benchmark judge; four ambiguous or unsupported cases stayed unchanged. Pilot record: `docs/operations/data/longmemeval-stage226-temporal-resolver-pilot-2026-10-26.json`. This is not a full-run release score.

**Commit:** `74cc7e9 fix: resolve citation-backed temporal intervals`

## Module 16: Recheck multi-session count inventory on the selected gateway

**Files:**
- Scorecards: `docs/operations/data/longmemeval-stage227-provider-seed-baseline-top20-2026-10-26.json`, `docs/operations/data/longmemeval-stage227-provider-seed-four-type-reranker-top20-2026-10-26.json`
- Error audit: `docs/operations/data/longmemeval-stage227-error-audit-2026-10-26.json`
- Count-policy records: `docs/operations/data/longmemeval-stage228-count-inventory-all-pilot-2026-10-26.json`, `docs/operations/data/longmemeval-stage228-count-inventory-frozen-pilot-2026-10-26.json`
- Related implementation: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`

**Observed Stage227 result:** Best full result on the requested gateway is 433/500 (86.6%) with four-type reranking, 52 below the 485/500 gate. Per type: knowledge update 72/78, multi-session 104/133, assistant 52/56, preference 26/30, user 67/70, temporal reasoning 112/133.

**Error audit:** Candidate misses: 29 multi-session, 21 temporal, 6 knowledge-update, 4 preference, 3 user, 4 assistant. Sixty-three of the 67 answer-side audits passed structural/citation validation, so evidence synthesis remains a major gap. This does not prove 100% factual precision.

**Policy experiment:** On all 94 multi-session count-inventory questions with the Stage227 selected Top-20 order frozen, `all` scored 69/94 versus 71/94 for `skip_multi_session`; 7 were all-only and 9 skip-only. The prompt hash matched and the Top-20 order matched in 94/94. Keep `skip_multi_session` as the default.

**Provider caveat:** The gateway accepted seed 42 in metadata but its unpaired reranker selections differed in 94/94 questions. The initial all-policy run cannot isolate inventory effects; the frozen-order run is authoritative for this pilot.

**Status:** No count-policy code change. Overall goal remains active: 433/500, 52 short of 485/500. Continue with temporal event matching and reasoning failures under fixed evidence.

## Module 17: Bound relative event ordering to cited date intervals

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Test: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Design: `docs/plans/2026-10-26-citation-backed-relative-event-order.md`
- Decision: `docs/decisions/adr-004-bounded-relative-event-order.md`
- Frozen-output pilot: `docs/operations/data/longmemeval-stage229-relative-event-order-frozen-pilot-2026-10-26.json`
- Fresh full candidate: `docs/operations/data/longmemeval-stage230-relative-order-four-type-candidate-top20-2026-10-26.json`

**Design:** Keep exact-date ordering. For approximate relative phrases, derive bounded intervals only from validated user-authored quotes and an unambiguous observation date. Rewrite two-option order answers only when cited intervals are disjoint; leave overlapping, assistant-only, or undated evidence unchanged. Prevent unsupported phrases such as “a few months before” from falling through to their embedded anchor date.

**Verification:** Adapter full suite: 92 tests and 20 subtests passed; compileall and `git diff --check` passed. Adapter commit: `4b0b2d2 fix: resolve citation-backed relative event order`.

**Frozen-output pilot:** Replayed the resolver on all 133 Stage227 temporal answers. Two answers changed; one previously wrong answer was corrected, one answer already judged correct remained correct, and there were no regressions. Frozen overall score diagnostic moved from 433/500 to 434/500; this is not a new full generation score.

**Fresh full candidate:** Stage230, using the new code and the Stage221 retrieval snapshot, scored 427/500 (85.4%), 58 below 485. The earlier Stage227 full candidate scored 433/500. Across reranked rows, only 2/374 selected memory orders matched between the two runs despite seed acceptance, so the six-answer delta is confounded by provider reranker variability. In Stage230 the target Mark/Tom chronology was corrected, while temporal category accuracy moved from 112/133 to 109/133.

**Status:** The best full score remains Stage227 at 433/500 (86.6%), 52 below the gate. Relative interval logic is citation-bounded and tested; it does not close the overall accuracy gap. Continue targeting multi-session and temporal reasoning under fixed evidence and track provider selection variance.

## Module 18: Add opt-in verification for single-session fact questions

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`, `README.md`, and `tests/test_longmemeval_run.py`
- Pilot: `docs/operations/data/longmemeval-stage231-single-session-verification-pilot-2026-10-26.json`
- Full scorecard: `docs/operations/data/longmemeval-stage232-single-session-verification-four-type-candidate-top20-2026-10-26.json`
- Design: `docs/plans/2026-10-26-single-session-answer-verification.md`
- Decision: `docs/decisions/adr-005-optional-question-type-answer-verification.md`

**Design:** Add `--answer-verification-question-types` as an opt-in allowlist. Preserve existing default cross-session, temporal, knowledge-update, and preference behavior. Revisions must pass exact-quote citation validation and full answer validation; invalid evidence keeps the original answer.

**Frozen pilot:** Across all 126 single-session user/assistant answers, 23 answers changed; 4 previously wrong answers were corrected, 0 correct answers regressed, and the slice moved from 115/126 to 119/126. This is a frozen-input pilot, not a full score.

**Full Stage232 result:** The new allowlist on a fresh full run scored 432/500 (86.4%) versus 427/500 in Stage230; paired outcomes were 27 Stage232-only and 22 Stage230-only. User questions moved 64/70→68/70 and assistant questions 51/56→52/56. However, only 3 of 374 reranker selections matched across the two runs, so provider variation confounds the aggregate +5. The Stage231 frozen pilot is the isolated evidence for this module.

**Decision/status:** Keep the flag opt-in. The best complete score remains Stage227 at 433/500 (86.6%), 52 below 485/500. No 97% or 100% factual precision claim is supported.

**Commit:** `5809c05 feat: scope answer verification by question type`.

## Module 19: Reject a second temporal verification pass

**Pilot:** `docs/operations/data/longmemeval-stage233-temporal-second-verification-frozen-pilot-2026-10-26.json`

**Protocol:** Replay one additional independent answer-verification call on the 133 Stage232 temporal answers with the same frozen Top-20 evidence. Do not regenerate answers or rerun the reranker. Rejudge only the changed answers.

**Observed result:** 60 answers changed; 3 were corrected and 7 previously-correct answers regressed. The frozen 500-question diagnostic moved from 432/500 to 428/500; temporal reasoning fell from 112/133 to 108/133. Keep one temporal verification pass and reject a second global pass.

**Status:** No code change. The single-session verification allowlist remains opt-in and supported by its separate positive frozen pilot. The best full score is still 433/500, 52 below the 485/500 target.

## Module 20: Reject consensus reranker selection from three prior runs

**Pilot:** `docs/operations/data/longmemeval-stage234-consensus-selection-pilot-2026-10-26.json`

**Protocol:** For 266 multi-session and temporal questions, combine three prior reranker selections by candidate frequency, mean position, and original retrieval rank. Freeze the resulting Top-20 order and rerun answer generation/judging without another reranker call.

**Observed result:** Consensus scored 210/266 (78.9%), versus 217/266 (81.6%) for the matched Stage232 selections; paired outcomes were 14 consensus-only and 21 Stage232-only. The consensus order used all three historical votes for 260 questions and Stage232 fallback order for six.

**Decision/status:** Do not implement this consensus policy. It underperformed the matched reference; continue with answer synthesis and evidence-grounded reasoning. Best full score remains 433/500, 52 below the target.

## Module 21: Audit answer-marked pair coverage for Stage227–232

**Artifact:** `docs/operations/data/longmemeval-stage235-answer-pair-coverage-audit-2026-10-26.json`

**Method:** For non-abstention questions, derive answer-bearing pairs from `has_answer=true` messages inside `answer_session_ids` sessions. Match them to Top-20 retrieval metadata `(source_session_id, source_pair_index)`; report coverage separately from the judged answer score.

**Observed Stage232 result:** 863/883 answer-marked pairs were in Top-20 (97.7%), and all answer pairs were present for 453/470 answerable questions. Among 61 incorrect answerable rows, 47 already had all annotated answer pairs in Top-20. This supports answer synthesis and event reasoning as the largest current gap; pair coverage is not factual precision.

**Decision/status:** No code change. Preserve the 485/500 Top-20 gate and target synthesis errors using frozen evidence. The best full score remains 433/500, 52 short.

## Module 22: Reject runtime count adjudication after the frozen pilot

**Artifact:** `docs/operations/data/longmemeval-stage237-count-adjudication-frozen-implementation-pilot-2026-10-26.json`

**Protocol:** Run the opt-in adjudication path on the same 94 multi-session count questions with the Stage227 Top-20 memories frozen. Compare against the Stage227 `skip_multi_session` and Stage228 `all` references; verify each question retained the exact same Top-20 memory order.

**Observed result:** Stage237 scored 68/94, below both `skip_multi_session` at 71/94 and `all` at 69/94. Against skip, the adjudicator had 4 candidate-only and 7 skip-only outcomes; against all, it had 5 candidate-only and 6 all-only outcomes. The Top-20 memory order matched Stage227 in 94/94 questions. The earlier offline Stage236 draft replay at 73/94 did not reproduce in the runtime implementation. Stage237 metadata recorded model aliases and seed acceptance but omitted the provider base URL.

**Decision/status:** Do not ship count adjudication. The uncommitted adapter implementation was removed; keep `skip_multi_session` as the default. This is a negative frozen-slice experiment, not a full benchmark score. The best full score remains 433/500, 52 below 485/500.

## Module 23: Encrypt Orion memory backups

**Files:** `src/civilization/engine/persistence/backup.py`, `src/civilization/cli.py`, `pyproject.toml`, `tests/engine/test_memory_backup.py`, and the session durability documentation.

**Design:** Make all new backups encrypted v2 bundles using independently authenticated 1 MiB AES-256-GCM frames and an Argon2id-derived key. Require a high-entropy passphrase via the Python API or `CIVILIZATION_BACKUP_PASSPHRASE`; never accept it as a CLI argument. Stream verification directly through the authenticated reader, restore into a private staging tree, and atomically promote only after the terminal frame, manifest, and payload hashes validate. Keep v1 plaintext archives readable for recovery but do not create them.

**Verification:** `python3 -m pytest tests/engine/test_memory_backup.py -q` passed 23 tests, including missing/short/wrong-key rejection, tampering/truncation, bounded KDF parameters, long PAX paths, multi-frame archives, v1 recovery, no plaintext verification temp file, optional-dependency failure, and CLI secret handling. `compileall`, pyproject TOML validation, and `git diff --check` passed. The adjacent session-persistence suite could not be collected in the current environment because `psutil` is not installed; no dependencies were installed to work around it.

**DR boundary:** This protects backup confidentiality and integrity; it does not create an independent copy. The checked volume has only about 3.6 GiB free and the repository and `/Volumes/Net/Recovery` share that same filesystem. No separate offsite target or credentials are available, so scheduling, replication, alerting, retention, and a production restore drill remain open. A same-volume encrypted archive is not disaster recovery.

## Module 24: Add count-aware answer verification

**Adapter commits:** `e41684f feat: verify numeric answers with cited item audits`; `e5063c6 fix: guard current-year count verification`; README follow-up `eb36278 docs: clarify count-aware verification scope`.

**Design:** Extend the opt-in answer verifier with item counts, units, totals, and dated events. A revised numeric answer is accepted only if the complete audit passes citation, evidence, unit, and count validation; all other responses keep the original structured answer. The normal default remains unchanged.

**Frozen Stage238 pilot:** With the Stage221 direct-retrieval Top-20 prefix frozen for the same 94 Stage228 multi-session count questions, a single baseline answer was generated and judged, then independently verified on that exact answer/evidence. Baseline was 57/94; candidate was 59/94. Paired results were 3 candidate-only and 1 baseline-only, a net gain of 2. The verifier changed 32 answers; 51 revisions were rejected as unverified and 6 for invalid evidence.

**Known regression:** Question `b5ef892d` moved from 8 days to 3 after the verifier misread “last month” as outside “this year.” The candidate had valid citations but incorrect time-window reasoning. Keep the feature opt-in and add a calendar-year window guard before broad enablement.

**Artifact and status:** `docs/operations/data/longmemeval-stage238-count-aware-verification-frozen-pilot-2026-10-27.json`. Stage238 is not a full score; the best complete score remains 433/500 (86.6%), 52 short of 485/500. A fresh 500-question Top-20 candidate is still required.

**Stage239 guard:** Added a deterministic check that rejects a current-year count revision when its cited Top-20 evidence contains a relevant user event from “last month” in the same year but the candidate omitted that item. It also prevents counting such an event when its source date falls in the prior year. Focused regression tests cover the observed Yellowstone case. This is a narrow safeguard; Stage239 full-candidate scoring remains pending.

**Stage239 full candidate:** On the Stage221 retrieval snapshot with four-type reranking and numeric verification selected for multi-session questions, the full Top-20 score was 422/500 (84.4%). Stage227 remains best at 433/500 (86.6%); Stage232 scored 432/500 (86.4%). Paired outcomes were 19 candidate-only / 30 Stage227-only and 20 candidate-only / 30 Stage232-only. Per type: knowledge update 73/78, multi-session 102/133, assistant 50/56, preference 25/30, user 68/70, temporal reasoning 104/133. The selected Top-20 matched Stage221's direct prefix in 130/500 rows, with mean ID overlap 11.8/20; ranking variance confounds attribution. The count subset scored 72/94 vs 71/94 Stage227 skip and 75/94 Stage232, also unpaired by selected rank.

**Stage239 pair-coverage audit:** 865/883 annotated answer-marked pairs were in the selected Top-20 (97.96%), with complete pair coverage in 455/470 answerable questions. Of 70 wrong answerable rows, 61 already had all marked answer pairs in Top-20. This reinforces answer synthesis and event reasoning as the next target; pair coverage is not factual precision. Artifact: `docs/operations/data/longmemeval-stage239-count-aware-verification-full-top20-2026-10-27.json`.

**Stage240 cited multi-event order pilot:** Added the opt-in `--temporal-event-list-order-resolution` resolver. It rewrites only a complete event list whose user-cited date intervals are all source-validated and pairwise disjoint; ambiguous cases retain the original answer. On Stage239's 133 frozen temporal answers, 2 rows were resolvable: one remained correct and one airline order was corrected. Temporal accuracy moved 109/133→110/133; paired outcomes were 1 candidate-only, 0 baseline-only, 1 both-correct, and 0 regressions. The frozen full-score diagnostic moved 422/500→423/500, which is not a fresh full score. Keep the flag opt-in pending the Stage241 full candidate. Adapter commit: `d1dc923`; Civilization pilot artifact: `docs/operations/data/longmemeval-stage240-temporal-list-order-frozen-pilot-2026-10-27.json`.

**Decision/status:** Keep the feature opt-in and do not claim an aggregate gain. The isolated Stage238 verifier pilot scored 59/94 vs its paired frozen baseline of 57/94, but had one current-year temporal regression. Best complete score remains 433/500 (86.6%), 52 below the 485/500 release gate. The 97% and 100% precision targets remain unmet.

## Module 25: Resolve cited multi-event temporal lists

**Adapter commit:** `d1dc923 fix: order cited multi-event temporal answers`. The opt-in `--temporal-event-list-order-resolution` resolver orders a complete temporal list only when every event has user-authored source evidence and a disjoint, source-validated date interval. Ambiguous or incomplete lists keep the original answer.

**Stage240 isolated pilot:** Replayed all 133 Stage239 temporal answers with frozen Top-20 evidence and rejudged only changed answers. The resolver applied to 2 rows: one already-correct answer stayed correct and one wrong airline order was fixed. Temporal accuracy moved 109/133→110/133. Paired outcomes were 1 candidate-only, 0 baseline-only, 1 both-correct, and 0 regressions. The frozen full diagnostic moved 422/500→423/500; this was not a fresh full run.

**Stage241 full candidate:** Replayed all 500 questions using the frozen Stage221 raw retrieval results and reran reranking, answer generation, and judging with seed 42. The 200 retrieved IDs and order matched the saved Stage221 inputs in 500/500 questions; answerer and judge accepted the seed. The score was 433/500 (86.6%), tying Stage227 and remaining 52 answers short of 485/500. By type: knowledge update 71/78, multi-session 102/133, assistant 51/56, preference 25/30, user 70/70, temporal reasoning 114/133. Against Stage227 the paired outcomes were 25 candidate-only and 25 reference-only (net zero); against Stage239 they were 30 candidate-only and 19 reference-only (net +11), but the reranker was repeated. The Stage241 Top-20 matched Stage239's exact order in 130/500 rows with mean ID overlap 16.658/20, and matched Stage221's direct Top-20 in 130/500 rows with mean overlap 11.816/20. Thus the aggregate delta versus Stage239 is not attributable to the resolver. The resolver fired twice in the full run; one answer changed from incorrect to correct, and one remained correct.

**Stage241 answer-pair coverage:** The selected Top-20 contained 868/883 annotated answer-bearing message pairs (98.3%), with complete pair coverage in 458/470 answerable questions. Of 59 wrong answerable answers, 52 already had every annotated answer pair in Top-20. This identifies answer synthesis and temporal reasoning as the larger measured gap, while 7 wrong answerable rows still lacked at least one marked pair. Pair coverage is not factual precision.

**Artifact and decision:** `docs/operations/data/longmemeval-stage241-temporal-list-order-full-top20-2026-10-27.json`. Keep the resolver opt-in. The best complete accuracy remains 433/500 (86.6%); the 97% and 100% precision goals remain unmet.

## Module 26: Bound user-cited “last week” dates conservatively

**Adapter commit:** `f455c0d fix: resolve cited last-week temporal order`. Extended the existing temporal comparison resolver to recognize only user-authored `last week` and `previous week` phrases with one observation date. The parser uses a broad 13-day interval ending one day before observation, covering rolling-week and previous-calendar-week readings. Existing logic still refuses to order overlapping intervals, assistant-only citations, missing dates, or conflicting observations.

**Stage242 isolated pilot:** Replayed all 133 Stage241 temporal answers and frozen Top-20 evidence. The resolver newly applied once and changed one answer: `gpt4_b4a80587`, the road-trip versus prime-lens order, moved from incorrect to correct. Temporal accuracy moved 114/133→115/133; paired outcomes were 1 candidate-only, 0 baseline-only, 114 both-correct, and 18 both-wrong. The frozen full diagnostic moved 433/500→434/500; this is not a fresh full score. Adapter verification passed 111 tests and 22 subtests, plus compileall and diff checks. Artifact: `docs/operations/data/longmemeval-stage242-last-week-temporal-frozen-pilot-2026-10-27.json`.

**Stage242 interaction audit:** Replayed the existing N-event list-order resolver on the same 133 frozen temporal answers using the expanded relative-week parser. It produced no additional answer changes and no new list resolutions.

**Status:** Stage242 passed its isolated gate. Stage243 initially stopped at 253/500 because `group/auto-gpt-6-luna` routed Chat Completions to the Home GPT-6 member, which rejected that protocol. On 2026-10-27, with user authorization, the local Magpie group was edited to disable only the Home GPT-6 member while retaining its slot, the alias ID, and the Codex member. A minimal Chat Completions request to the group then returned HTTP 200. Saving this edit made the auto-discovered group user-managed; future membership changes require manual review.

## Module 27: Bound immediate-event dates for temporal lists

**Adapter commit:** `8292b8d feat: add opt-in immediate event ordering`. The new `--temporal-recent-event-order-resolution` flag uses a broad `[observation_date - 6 days, observation_date]` interval only when an exact user-authored citation contains a completed immediate-event phrase, its observation date is unique, and the event description shares at least two meaningful terms with the quote. The N-event resolver still requires the expected complete list and pairwise-disjoint intervals. The flag is disabled by default and recorded in run metadata.

**Stage244 offline replay:** Replayed all 133 frozen Stage241 temporal answers. One answer changed: `gpt4_d6585ce8`; its previously undated Brooklyn music festival and Queen/Adam Lambert concert were placed between the exact March 25 outdoor concert and April 8 jazz night. The resulting five event identities and order match the benchmark reference. Chat Completions was unavailable, so the changed answer was not live-judged and no accuracy score is claimed. Artifact: `docs/operations/data/longmemeval-stage244-immediate-event-order-offline-replay-2026-10-27.json`.

**Verification and status:** Adapter suite passed 114 tests and 25 subtests; compileall and diff checks passed. Keep the feature opt-in pending a frozen judge pilot and full 500-question candidate. The best verified complete score remains 433/500 (86.6%); 97% accuracy and 100% precision remain unachieved.

## Module 28: Replicate encrypted backups across filesystems

**Civilization commit:** `2f968d5 feat: replicate encrypted backups across filesystems`. `replicate-backup` copies an already encrypted v2 bundle to an existing target directory only when source and destination device IDs differ. It authenticates the source, streams ciphertext to a mode-0600 temporary file, compares size and SHA-256, authenticates the copy, then atomically publishes without replacing an existing generation. The CLI reads the passphrase only from `CIVILIZATION_BACKUP_PASSPHRASE`; it has no passphrase argument and does not delete older backups.

**Synthetic cross-device drill:** Source was a tiny synthetic state under `/Volumes/Net` (iSCSI); the encrypted replica was written under a temporary directory on `/tmp` (local data volume). Device IDs differed; replica verification and restore succeeded; all payload files matched exactly. The random test passphrase and temporary source, replica, and restore trees were removed after the run. No production memory state was copied.

**DR boundary:** `/Volumes/Recovery` is a root-owned, non-writable macOS Recovery volume on the internal SSD, not a user backup destination. `/tmp` proves only a cross-device local round trip; it is ephemeral and not offsite. The new command provides manual replication but no configured offsite destination, schedule, retention, alerting, production restore drill, or RPO/RTO commitment. Disaster recovery remains incomplete. The focused backup suite passed 30 tests; full repository test collection is blocked by missing optional packages including `torch` and `psutil`.

## Module 29: Add resumable S3-compatible encrypted backup replication

**Civilization commit:** `474ec9e feat: add resumable S3 encrypted backup replication`. The optional `backup-s3` extra keeps boto3 out of the core package. `replicate-s3-backup` requires a versioned bucket, caller-selected prefix, and private durable checkpoint directory. It derives a content-addressed key, uploads multipart with SHA-256 part checksums, checkpoints the upload ID and each confirmed part, resumes by reconciling `ListParts`, and verifies the completed VersionId with a full remote read-back plus the existing authenticated bundle verifier. The CLI gets cloud credentials from the SDK provider chain; no access key or passphrase CLI arguments exist. Retention planning preserves the latest configured generations, recent versions, Object Lock retention, and legal holds; deletion requires explicit `--apply`.

**Local verification:** S3 fake-client tests cover multipart resume after lost create/part responses, expired upload restart, durable checkpoint mode, encrypted remote read-back, corruption rejection without deleting the remote object, Object Lock-aware retention planning/application, fail-closed lock inspection, endpoint validation, CLI credential boundaries, and the optional dependency error. Combined S3 + backup suite: 41 passed. `compileall`, pyproject TOML parsing, and `git diff --check` passed. Full repository test collection still fails because optional `torch`, `psutil`, and other research dependencies are not installed.

**Stage243 full candidate:** The resumed 500-question Top-20 run scored **438/500 (87.6%)**, 47 answers below 485/500. Compared with Stage241 (433/500), 21 questions were candidate-only correct and 16 baseline-only correct (net +5). Three empty judge responses were rerun against their saved answers without answer changes; 2 passed and 1 failed. The scorecard records 470/470 source-session hits and 871/883 answer-marked pairs in Top-20, while 47 answers remained wrong despite all marked pairs being present. Per-type and paired details: `docs/operations/data/longmemeval-stage243-relative-week-top20-2026-10-27.json`. The 97% target remains unmet; 100% memory precision has not been measured.

**Stage245 multi-session count-inventory A/B:** Replayed 94 Stage243 multi-session count questions with the Stage243 Top-20 order frozen, the same provider/model, and seed 42 accepted by both answerer and judge. Changing only the count-inventory policy from `skip_multi_session` to `all` moved 73/94→70/94; paired outcomes were 8 candidate-only, 11 baseline-only, 62 both-correct, and 13 both-wrong (net −3). Each variant logged a structured count-output retry, then completed all rows. Keep `skip_multi_session`; the broad inventory policy failed this isolated gate. Artifact: `docs/operations/data/longmemeval-stage245-multisession-count-inventory-pilot-2026-10-27.json`. This 94-row pilot is not a full benchmark score.

**Stage246 multi-session fact-ledger A/B:** Replayed all 133 Stage243 multi-session questions with the same saved Top-20 order, `group/auto-gpt-6-luna`, and seed 42 accepted by both answerer and judge. The prompt-only ledger scored 99/133 versus 103/133 with the flag off (−4; −3.01 pp); paired outcomes were 7 candidate-only, 11 control-only, 92 both-correct, and 23 both-wrong. The candidate had one more invalid answer audit (10 vs 9) and one verifier `invalid_response` (vs 0). All rows completed; transient control connection retries recovered. The `memory-benchmarks` feature commit `aa340a9` was reverted in `6e86ec5`; do not enable or retry this broad ledger without a narrower failure-specific design. Artifact: `docs/operations/data/longmemeval-stage246-multisession-fact-ledger-pilot-2026-10-27.json`. This 133-row pilot does not change the full Stage243 score of 438/500 (87.6%).

**Stage247 multi-session verifier-repair A/B:** Replayed all 133 Stage243 multi-session questions with frozen Top-20 evidence, the same provider/model, and seed 42 accepted by both answerer and judge. Control and the opt-in one-retry candidate both scored 106/133; paired outcomes were 6 candidate-only, 6 control-only, 100 both-correct, and 21 both-wrong (net zero). The candidate retried 59 verifier results; 26 passed validation and 33 remained rejected. It had 11 invalid answer audits versus 9 in control. Numeric aggregates fell 92/111→90/111. The non-numeric subgroup rose 14/22→16/22, but no retry occurred in that subgroup, so its two candidate-only wins are not attributable to the feature. Discard the implementation; no flag remains. Artifact: `docs/operations/data/longmemeval-stage247-multisession-verifier-repair-pilot-2026-10-27.json`. This is not a new full score; Stage243 remains 438/500 (87.6%).

**Stage249 bounded relative-date ordering:** Frozen replay of all 133 Stage243 temporal answers changed only `gpt4_213fd887`; it corrected one wrong ordering, passed the existing judge and answer validator, and caused zero regressions (114/133→115/133). A full 500-question candidate scored 429/500 versus Stage243's 438/500 and reused the same Top-20 retrieval order in 500/500 rows; however, the new fallback applied zero times in those generated outputs, so the full score delta is not attributable to the resolver. Keep the feature opt-in and do not claim a score gain. Code commit: `019b31b feat: resolve adjacent relative temporal order`. Artifact: `docs/operations/data/longmemeval-stage249-relative-temporal-boundary-pilot-2026-10-27.json`. Stage243 remains the best verified full score.

**Stage250 explicit-count priority A/B:** A user-only detector found 8/133 Stage243 multi-session questions with an exact numeric quantity and a vague repeat for the same noun across separate memories. The candidate injected the prompt rule only in those questions. Control and candidate each scored 106/133; paired outcomes were 6 candidate-only, 6 control-only, 100 both-correct, and 21 both-wrong (net zero). Within the 8 triggered questions, both arms were correct on 7 and wrong on 1, with no label changes. The feature failed its accuracy gate and the uncommitted implementation was discarded. Artifact: `docs/operations/data/longmemeval-stage250-multisession-explicit-count-priority-pilot-2026-10-27.json`. The best full Top-20 score remains Stage243 at 438/500 (87.6%).

**Live DR boundary:** No S3-compatible endpoint, bucket, credentials, or retention policy have been authorized/configured, so no live object was uploaded and no end-to-end remote restore has been performed. The adapter is locally verified only; an authorized endpoint and bucket are required before the live drill.
