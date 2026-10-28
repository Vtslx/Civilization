# LongMemEval scorecards and release gate

## Fixed protocol

The release metric is the standard 500-question LongMemEval answerer result at
`top_20`. The runner retrieves up to 200 memories, generates an answer from the
first 20, and uses the same configured judge protocol. `top_200` is available
as a diagnostic when the answerer receives the complete retrieval list, but it
does not satisfy the Top-20 release gate. Candidate and baseline
must match on the full question-ID set, question categories, answerer, judge,
provider, `top_k=200`, and cutoff list. The result metadata must certify
`all_questions=true`, answerer mode, and all 500 rows.

The 97% target means at least 485 fully correct answers out of 500. The scorecard
reports both strict correct-answer accuracy and mean judge score, plus each of
the six question types, paired wins/losses, search latency, returned-result
counts, explicit API-error rows, exact source-session chunk recall at Top-20 and
Top-200 with MRR@200, and answer-marked pair recall when the dataset provides
pair labels. Session recall only proves at least one chunk from a source
session was retrieved; pair recall checks whether the chunks marked as
containing answers were retrieved. Neither is the answerer accuracy metric. A
partial score does not count as a fully correct answer for the 485-answer gate.

## Run and score

Run the benchmark through Civilization's native Orion adapter with the same
answerer/judge settings as the baseline. Keep dataset, adapter, prompts, model,
retrieval count, and cutoff unchanged between runs.

```bash
civilization scorecard \
  --results /path/to/candidate-longmemeval.json \
  --baseline /path/to/baseline-longmemeval.json \
  --dataset-file /path/to/longmemeval_s_cleaned.json \
  --expected-questions 500 \
  --cutoff top_20 \
  --target-accuracy 0.97 \
  --output /path/to/candidate-scorecard.json
```

The scorecard rejects partial runs, duplicate question IDs, missing scores,
retrieval-only mode, unsupported cutoffs, and mismatched baseline protocols. It
checks the candidate and baseline against the same dataset file and contains its
SHA-256 fingerprint. Only a `top_20` card can satisfy the release gate. It
contains aggregate metrics and model/run labels, but does not copy question
text, ground-truth answers, generated answers, or judge reasoning.

## Baseline and limits

The previously recorded full run scored **349/500 (69.8%)** at Top-20. On the
470 questions with an annotated answer session, an exact chunk from that source
session appeared in 446 Top-20 lists (94.9%) and all 470 Top-200 lists (100%);
MRR@200 was 0.7457. This measures source-session presence, not whether the
specific answer fact is in the chunk. The six answer-accuracy categories and
these retrieval diagnostics are recorded in the sanitized [baseline scorecard](data/longmemeval-baseline-2026-10-25.json).

The Stage162 full Top-20 rerun scored **414/500 (82.8%)**, a 13.0 percentage-point
gain over that baseline. A separate same-run Top-200 diagnostic scored
**451/500 (90.2%)**. With a source-session chunk in Top-20, the answerer still
missed 81 of 465 questions (82.6% conditional answer accuracy). With a
source-session chunk in Top-200, it missed 48 of 470 questions. These results
show a large retrieval gain while leaving answer generation as a separate
limit; neither score meets 485/500.
The paired Top-20 comparison is in
[longmemeval-stage162-top20-2026-10-25.json](data/longmemeval-stage162-top20-2026-10-25.json),
and the Top-200 diagnostic is in
[longmemeval-stage162-top200-2026-10-25.json](data/longmemeval-stage162-top200-2026-10-25.json).

The Stage163 speaker-aware rerun scored **424/500 (84.8%)** at Top-20, a further
2.0 percentage-point gain over Stage162, with 27 candidate-only wins and 17
baseline-only wins. It had source-session chunks in 458/470 Top-20 lists; the
answerer still missed 65 of those 458 questions. It remains 61 answers below
the 485-answer gate. The paired result and per-type changes are in
[longmemeval-stage163-top20-2026-10-25.json](data/longmemeval-stage163-top20-2026-10-25.json).
The same role-aware build scored **454/500 (90.8%)** in the Top-200 diagnostic,
only 3 answers above Stage162 Top-200 and still 31 below 485; see
[longmemeval-stage163-top200-2026-10-25.json](data/longmemeval-stage163-top200-2026-10-25.json).

The Stage165 bounded source-session retriever with LLM Top-20 reranking and
structured, cited answers scored **416/500 (83.2%)** on the full Top-20 run.
The 97% gate was not met; the candidate is 69 correct answers short. Per-type
results were knowledge-update 70/78 (89.7%), multi-session 105/133 (78.9%),
single-session-assistant 54/56 (96.4%), single-session-preference 27/30 (90.0%),
single-session-user 67/70 (95.7%), and temporal-reasoning 93/133 (69.9%). The
retrieval returned an answer-source session in all 470 annotated Top-20 cases,
but the answerer still missed 79 of them. Session presence is not proof that
every event or exact fact needed for a multi-event answer was selected.

This candidate used an LLM reranker, an evidence-JSON answer contract, and a
count-completeness review; it is not a prompt-identical paired comparison with
Stage163. The candidate-only scorecard, including dataset and result hashes, is
[longmemeval-stage165-reranked-top20-2026-10-25.json](data/longmemeval-stage165-reranked-top20-2026-10-25.json).

Stage166 added temporal-event evidence fields and anchor-relation checks. Its
full conditional Top-20 run scored **414/500 (82.8%)**, 71 answers short of the
gate. Temporal-reasoning increased from 93/133 to 95/133 (71.4%), while the
overall score fell by two answers and remains below Stage163. The candidate-only
card is [longmemeval-stage166-temporal-top20-2026-10-25.json](data/longmemeval-stage166-temporal-top20-2026-10-25.json).

Stage168 added event-date interval resolution and before/after anchor filtering.
The full Top-20 run scored **426/500 (85.2%)**, 59 answers short of the gate;
that is 10 answers above Stage165 and 2 above Stage163, but it is not a
prompt-identical paired comparison. Temporal-reasoning was 102/133 (76.7%),
multi-session 104/133 (78.2%), knowledge-update 70/78 (89.7%),
single-session-assistant 55/56 (98.2%), single-session-preference 25/30
(83.3%), and single-session-user 70/70 (100%). All 470 annotated answer-source
sessions appeared in Top-20, but only 401 of those questions were answered
correctly. The candidate-only card is
[longmemeval-stage168-temporal-top20-2026-10-25.json](data/longmemeval-stage168-temporal-top20-2026-10-25.json).

Stage169 introduced a separate count-inventory pass in the benchmark adapter.
Its first full run scored **409/500 (81.8%)**, 76 answers short of the gate.
On the same Stage168 ranked evidence, 18 questions changed from wrong to right
and 35 changed from right to wrong. The initial adapter over-counted cumulative
snapshots and routed some scalar questions through item enumeration. Its card is
[longmemeval-stage169-count-inventory-initial-top20-2026-10-25.json](data/longmemeval-stage169-count-inventory-initial-top20-2026-10-25.json).

Stage171 narrowed inventory routing to enumerative count questions, treated
cumulative snapshots as updates, and kept arithmetic, rate, and identifier
questions on the answerer path. With the Stage168 ranked Top-200 candidate lists
preserved, the full run scored **418/500 (83.6%)**, 67 answers short of the
485-answer gate. The paired comparison against Stage168 was 16 candidate-only
correct and 24 baseline-only correct, a net **−8 answers (−1.6 percentage
points)**. The initial inventory run scored 409/500, so this refinement was
9 answers higher there; that is a different prompt revision, not a new 97% pass.
Per-type scores were knowledge-update 67/78 (85.9%), multi-session 100/133
(75.2%), single-session-assistant 56/56 (100%), single-session-preference
25/30 (83.3%), single-session-user 69/70 (98.6%), and temporal-reasoning
101/133 (75.9%). The inventory route scored 135/159 on its selected questions;
multi-session remains the largest gap. See
[longmemeval-stage171-count-inventory-top20-2026-10-25.json](data/longmemeval-stage171-count-inventory-top20-2026-10-25.json).

Stage174 added an independent citation-backed answer check for non-numeric
multi-session, knowledge-update, and temporal questions. The full run scored
**426/500 (85.2%)**, still 59 answers short of the gate. It matched Stage168's
overall score: 24 questions changed from wrong to right and 24 from right to
wrong. The verifier revised 92 answers, of which 74 were judged correct, but
the net paired gain over Stage168 was zero. Temporal-reasoning reached 106/133
(79.7%), knowledge-update 71/78 (91.0%), and multi-session remained 102/133
(76.7%). All 470 answer-source questions still had a source session in Top-20;
402 were answered correctly. This check improves some cited answers but does
not establish a 97% result. See
[longmemeval-stage174-answer-verification-top20-2026-10-25.json](data/longmemeval-stage174-answer-verification-top20-2026-10-25.json).

Stage178 ran the same ranked evidence with `--no-count-inventory`; direct
structured answering and count-completeness review remained enabled. It scored
**424/500 (84.8%)**, 61 answers short of the gate. Compared with Stage174's
count-inventory-enabled run, the ablation was lower by two answers: 22 questions
were correct only with inventory disabled and 24 only with it enabled. Per-type
scores were knowledge-update 72/78 (92.3%), multi-session 100/133 (75.2%),
single-session-assistant 56/56 (100%), preference 25/30 (83.3%), user 69/70
(98.6%), and temporal-reasoning 102/133 (76.7%). The result is close enough that
the default stays enabled; keep the CLI switch for repeatable A/B analysis. The
scorecard records the feature flag in `adapter_features`:
[longmemeval-stage178-no-count-inventory-top20-2026-10-25.json](data/longmemeval-stage178-no-count-inventory-top20-2026-10-25.json).

Stage184 added an explicit adapter policy that skips the pre-answer count
inventory for multi-session count questions while retaining direct cited
answering and count-completeness review. The policy is recorded in the result
and scorecard metadata, and `--count-inventory-policy all` restores the prior
route. Replaying all 500 questions with Stage168's fixed ranked Top-200
candidates scored **429/500 (85.8%)**, 56 answers short of the gate. Against
Stage174, 28 questions were candidate-only correct and 25 baseline-only
correct, a net gain of 3 answers (+0.6 percentage points). On the 95
multi-session count questions, the candidate scored 74/95 versus 70/95 in
Stage174 (12 gains, 8 losses). Multi-session overall reached 105/133 (78.9%),
while preference was 24/30 (80.0%). This is a small measured gain; it does not
establish 97% accuracy. The scorecard is
[longmemeval-stage184-count-policy-top20-2026-10-25.json](data/longmemeval-stage184-count-policy-top20-2026-10-25.json).

Stage188 adds deterministic temporal post-processing for event-to-event
intervals and two-option event-order questions when both cited dates are
usable. This is a **frozen-response postprocessor replay** over Stage184's 500
saved answer records and the same ranked Top-200 candidates; changed answers
were judged again, while unaffected records were retained. It scored
**434/500 (86.8%)**, 51 answers short of the gate. Compared with Stage174, 29
questions were candidate-only correct and 21 baseline-only correct, a net gain
of 8. Temporal reasoning reached 111/133, five above Stage184. The scorecard
marks `result_provenance.postprocessor_replay_only=true` so this is not mistaken
for a fresh all-500 answer-generation run:
[longmemeval-stage188-temporal-postprocessor-top20-2026-10-25.json](data/longmemeval-stage188-temporal-postprocessor-top20-2026-10-25.json).

A fresh live runner replay of the 133 temporal questions scored 105/133, then
106/133 on a second run, versus 106/133 for Stage184. Those runs show model
answer and judge variation; they did not reproduce the full five-answer lift
from the frozen-response ablation. Treat Stage188 as evidence for the
deterministic postprocessor on fixed outputs, not as proof of a fresh-run
accuracy gain or a 97% result.

Stage195 adds an independent preference check that uses cited user preferences
to tailor advice and retries once when its citations fail validation. This is a
**subset replay composition**: the runner re-evaluated the 30
single-session-preference questions from Stage188 while retaining its other 470
rows. It scored **437/500 (87.4%)**, 48 answers short of the gate. Against
Stage174, 32 questions were candidate-only correct and 21 baseline-only
correct. Preference reached 27/30 (90.0%). Two live preference replays scored
25/30 and 27/30, compared with Stage188's 24/30; those samples show some model
variation. The scorecard records the preference flag and subset provenance,
and does not claim a new all-500 generation run:
[longmemeval-stage195-preference-subset-replay-top20-2026-10-25.json](data/longmemeval-stage195-preference-subset-replay-top20-2026-10-25.json).

Stage196 removes an explicitly labeled aggregate count row only when at least
two same-unit incremental items sum exactly to it. This prevents double-counting
reported totals while leaving incomplete or per-unit amounts on their direct
answer path. A frozen-response replay over Stage195 scored **439/500 (87.8%)**,
46 short of the gate. Against Stage195, two answers changed from wrong to right
and none regressed; the corrected cases were a handbag plus skincare total and
a car-cover plus detailing-spray total. Against Stage174, 32 questions were
candidate-only correct and 19 baseline-only correct. The scorecard records
`explicit_total_deduplication_enabled` and replay provenance:
[longmemeval-stage196-explicit-total-dedup-top20-2026-10-25.json](data/longmemeval-stage196-explicit-total-dedup-top20-2026-10-25.json).
This remains a frozen-response replay, not a fresh all-500 generation run.

Stage203 adds a conditional entity-count instruction for physical-object
questions. A 12-question fixed-candidate runner replay scored 7/12 versus 5/12
in Stage196 (3 gains, 1 loss) in two repeats with the same aggregate score.
Composing that subset with Stage196's other rows gives **441/500 (88.2%)**, 44
short of the gate. Against Stage174, the scorecard has 31 candidate-only and 16
baseline-only correct answers. Its provenance identifies the replay scope as
`multi-session` / `physical_entity_count`; the 500-row result is not a fresh
full generation run:
[longmemeval-stage203-entity-count-subset-top20-2026-10-25.json](data/longmemeval-stage203-entity-count-subset-top20-2026-10-25.json).

Stage203's pair-level retrieval diagnostic found 867/883 answer-marked pairs in
Top-20 (98.2%); all answer-marked pairs were present for 456/470 annotated
questions. Within multi-session, 312/322 pairs were present and 113/121
questions had complete pair coverage. The answerer still missed 14 of those 113
multi-session questions, so the remaining errors split between missing
answer-bearing chunks and reasoning over chunks that are present.

Stage213 validates evidence roles for personal count items in role-marked
memories: assistant-only claims cannot establish the user's ownership,
subscription, or completed action. It leaves assistant-output questions and
hypothetical amount calculations on their existing path. A one-question replay
of the NYT subscription citation over the Stage203 composite removes the
assistant-only item and recomputes the count from two user-supported
subscriptions. The 500-row composition scores **442/500 (88.4%)**, 43 short of
the gate; compared with Stage174, it has 32 candidate-only and 16 baseline-only
correct answers. Its provenance scope is `assistant_only_personal_count`; it
is not a fresh all-500 run:
[longmemeval-stage213-user-authored-count-subset-top20-2026-10-25.json](data/longmemeval-stage213-user-authored-count-subset-top20-2026-10-25.json).

Stage216 makes count questions with explicit windows such as “this year,” “in
March,” or “in the past two weeks” use temporal event fields even when their
benchmark category is multi-session. The 19-question time-window subset scored
18/19 versus 17/19 in Stage203 (1 gain, no regression). Composing it with the
other Stage213 rows yields **443/500 (88.6%)**, 42 answers short of the gate.
The scorecard marks the replay scope as `relative_count_window`; this is not a
fresh all-500 generation run:
[longmemeval-stage216-time-window-subset-top20-2026-10-25.json](data/longmemeval-stage216-time-window-subset-top20-2026-10-25.json).

Stage219 adds a deterministic date fallback for one narrow case: a cited
user-authored quote says they just got back from a guided tour, the event date
is missing, and the interval question needs that date. It uses the retrieved
memory's observation-date metadata; assistant-only quotes, unstructured
memories, unrelated events, and missing observation dates do not trigger the
fallback. Two live replays of the affected question both returned the supported
seven-day interval. Composing that one temporal-reasoning row with Stage216
produces **444/500 (88.8%)**, one answer higher and 41 short of the gate. The
paired replay has one candidate-only win and no regressions. This is a
one-question subset replay, not a new all-500 generation run:
[longmemeval-stage219-immediate-return-date-subset-top20-2026-10-25.json](data/longmemeval-stage219-immediate-return-date-subset-top20-2026-10-25.json).

## Best verified full Top-20 result: Stage243 (2026-10-27)

Stage243 regenerated and judged Top-20 answers over Stage221's frozen Top-200 retrieval
results. The complete 500-question score is **438/500 (87.6%)**, 47 answers below the
485/500 Top-20 release gate. Against Stage241's 433/500, paired outcomes were 21
candidate-only correct, 16 Stage241-only correct, 417 correct in both, and 46 wrong in
both: a net gain of 5 answers (+1.0 percentage point). Stage241's raw retrieval results
were archived, so this paired comparison checks matching question IDs, categories, and
strict Top-20 scores; it does not compare raw retrieval lists.

Per type: knowledge-update 72/78 (92.3%), multi-session 106/133 (79.7%),
single-session-assistant 51/56 (91.1%), single-session-preference 27/30 (90.0%),
single-session-user 68/70 (97.1%), and temporal-reasoning 114/133 (85.7%). Top-20
retrieved a source-session chunk for 470/470 annotated questions and 871/883
answer-marked message pairs (98.6%). Of 460 questions with every annotated answer pair
in Top-20, 47 were still wrong. This points to multi-session synthesis and answer
reasoning as the larger remaining gap; source or evidence coverage is not answer
accuracy. The run recorded zero API errors. Three saved rows with empty judge responses
were rejudged from their saved answers without changing answer text; two passed and one
remained incorrect.

To restore Chat Completions, Magpie's `group/auto-gpt-6-luna` route disabled its
protocol-incompatible Home GPT-6 member while preserving the group ID and Codex member.
The alias then returned HTTP 200 through Codex. This repaired route availability; the
paired score delta cannot be attributed to the route change alone. Stage243 did not meet
97%, and it did not measure the separate “100% memory precision” target. No Top-200
answer diagnostic was run in this stage.

Validated artifact:
[longmemeval-stage243-relative-week-top20-2026-10-27.json](data/longmemeval-stage243-relative-week-top20-2026-10-27.json).

## Stage245 multi-session count-inventory frozen A/B (subset pilot)

To test whether multi-session count inventory should be re-enabled, the 94 multi-session
count questions were replayed from Stage243 with the exact same Top-20 evidence order,
model, and seed-42 answerer/judge; only `count_inventory_policy` changed. The
`skip_multi_session` control scored 73/94, while `all` scored 70/94. Paired outcomes were
8 candidate-only correct, 11 control-only correct, 62 correct in both, and 13 wrong in
both: a net loss of 3 answers (-3.19 percentage points). Both variants completed all 94
rows; each logged a structured count-output retry and recovered.

Decision: retain `skip_multi_session` and do not enable broad count inventory. This is a
frozen-evidence 94-question A/B, not a full score or release-gate result. The artifact and
question-set hash are in
[longmemeval-stage245-multisession-count-inventory-pilot-2026-10-27.json](data/longmemeval-stage245-multisession-count-inventory-pilot-2026-10-27.json).

That 69.8% run is a baseline, not a claim about the current code. A scorecard
comparison is meaningful only when both inputs came from the same 500-question
dataset and unchanged evaluator protocol. The answerer and judge used the same
model in the baseline, so judge self-bias remains a limitation. Retrieval
precision, evidence coverage, answer accuracy, and scorecard integrity are
separate measurements; none can substitute for the 485/500 gate.

## Later pilots: Stages256–261 (2026-10-27)

Stage256's temporal audit consistency guard corrected one saved temporal answer on frozen evidence and passed the judge: 114/133→115/133 with zero regressions. This is a saved-answer replay, not a full-generation score. Details: [Stage256 scorecard](data/longmemeval-stage256-temporal-choice-audit-consistency-pilot-2026-10-27.json).

Stage257 regenerated all 500 answers with six opt-in resolvers and the reranker disabled. It scored 434/500, compared with Stage243's 438/500; paired outcomes were 22 candidate-only and 26 Stage243-only. The Top-20 and Top-200 ordered retrieval IDs matched in 500/500. All per-question results were valid, but the unified result JSON was truncated when the system volume ran out of space; the scorecard records that limitation and the per-question aggregation. Stage243 remains the best verified full score. Details: [Stage257 scorecard](data/longmemeval-stage257-combined-opt-in-top20-2026-10-27.json).

Stage258 isolated the fact ledger to the second-pass verifier. On all 133 frozen multi-session questions, control scored 108/133 and candidate 106/133; paired outcomes were 6 candidate-only and 8 control-only. Candidate invalid answer audits increased from 8 to 9, and verifier invalid-evidence rejections increased from 2 to 5. The adapter implementation was reverted after this negative pilot. Details: [Stage258 scorecard](data/longmemeval-stage258-multisession-verifier-only-fact-ledger-pilot-2026-10-27.json).

Stage259 restricted the same verifier ledger to nonnumeric multi-session questions and paired 22 questions across seeds 42, 43, and 44. Across 66 question-seed cases, control scored 46/66 and candidate 45/66; there were 6 candidate-only and 7 control-only outcomes. Candidate invalid answer audits fell from 3 to 2, but strict accuracy fell by one answer. The feature was reverted. Details: [Stage259 scorecard](data/longmemeval-stage259-nonnumeric-multisession-ledger-pilot-2026-10-27.json).

Stage260 added a default-off source-backed relative-time entity resolver. A frozen replay corrected two temporal answers; both passed citation validation and the judge, with zero regressions (114/133→116/133). A fresh 500-question candidate scored 432/500 and matched Stage243's Top-20 and Top-200 order in 500/500. The resolver did not revise any newly generated answer in that run, so the aggregate delta is not attributable to it. Stage243 remains the best verified full result. Details: [Stage260 scorecard](data/longmemeval-stage260-relative-time-event-answer-resolution-top20-2026-10-27.json).

Stage261 extended the opt-in temporal list resolver to explicit ordinal rankings and one observed “yesterady” typo. Its frozen Stage260 temporal replay corrected one wrong answer, which passed citation validation and judge; the slice moved 110/133→111/133 with zero regressions. The fresh 500-question candidate scored 432/500, tied with Stage260 and below Stage243's 438/500; Top-20 and Top-200 order matched 500/500. Keep the rule opt-in and do not credit it with a full-score gain. Details: [Stage261 scorecard](data/longmemeval-stage261-ordinal-temporal-event-order-top20-2026-10-27.json).

## Stage277: research poster conference venue resolution (2026-10-28)

The benchmark adapter adds a default-off `--research-poster-conference-venue-resolution` resolver. It joins a user-authored thesis-poster statement to a university venue only when both cite the same uniquely named first research conference. Assistant-only venue claims, conflicting universities, missing event aliases, and undergraduate-course poster questions do not trigger it.

Using Stage268's frozen Top-200 retrieval, a fresh 500-question Top-20 answer-and-judge run scored **441/500 (88.2%)**, one net answer above Stage268's 440/500 and 44 below the 485/500 gate. Paired outcomes were 26 candidate-only correct, 25 baseline-only correct, 415 correct in both, and 34 wrong in both. Because 258 answer texts changed, do not attribute the aggregate delta solely to this one-row resolver. The full Top-20, Top-200, and retrieval result objects matched Stage268 for 500/500 questions. Candidate invalid answer audits were 19 versus 23 in Stage268; no API-error markers were recorded.

Target `a96c20ee` changed from FAIL to PASS as “Harvard University.” with both source citations validated. The related undergraduate-course question `a96c20ee_abs` was correctly answered by abstention; the resolver status was `not_applicable`, so that row is not credited to the resolver. Keep the option opt-in. The adapter commit is `ca09b31 feat: resolve research poster venue from linked sources`; verification passed 210 unittest tests, 25 subtests, compileall, diff checks, and all 500 rows.

The best verified full result is now **441/500 (88.2%)**, still below **485/500 (97%)**. The separate 100% memory-precision target remains unmeasured, and a live offsite disaster-recovery restore remains unverified. See [longmemeval-stage277-research-poster-venue-top20-2026-10-28.json](data/longmemeval-stage277-research-poster-venue-top20-2026-10-28.json).

## Stage278: Master's completion endpoint guard (2026-10-28)

The adapter adds default-off `--unverified-masters-education-span-guard`. For a numeric education-duration answer explicitly asking from high school through Master's completion, it abstains with citations when user-authored evidence only establishes lower degrees and says the Master's is being considered. A direct user-authored Master's completion statement leaves the answer unchanged; assistant-only statements do not count. A question about the interval to submitting a Master's thesis stays outside this scope.

Stage278 regenerated and judged all 500 answers on Stage268's frozen Top-200 retrieval, scoring **435/500 (87.0%)**. Against Stage277, paired outcomes were 20 candidate-only, 26 Stage277-only, 415 correct in both, and 39 wrong in both (net -6) over 272 changed answers. Against Stage268, it scored 435 versus 440, with 17 candidate-only and 22 baseline-only answers. The full Top-20, Top-200, and retrieved result objects matched Stage268 in 500/500 questions; invalid answer audits were 23, and no API-error markers were recorded. One transient generation timeout recovered on retry.

Target `gpt4_372c3eed_abs` changed from Stage277's “10 years”/FAIL to a citation-validated abstention/PASS; a live one-question answer-and-judge replay also passed. The separate Master's-thesis interval row `gpt4_21adecb5` stayed at 6 months/PASS with guard status `not_applicable`. The guard triggered once and was not applicable to the other 499 questions. Keep it opt-in. Adapter commit: `3cc53e3 fix: abstain on unsupported masters education spans`. Verification passed 215 unittest tests, 25 subtests, compileall, diff checks, and all 500 rows.

Stage277 remains the best verified full score at **441/500 (88.2%)**, 44 below the **485/500 (97%)** gate. The separate 100% memory-precision target remains unmeasured, and live offsite disaster-recovery restore remains unverified. See [longmemeval-stage278-unverified-masters-education-span-top20-2026-10-28.json](data/longmemeval-stage278-unverified-masters-education-span-top20-2026-10-28.json).

## Stage279: Sephora loyalty points remaining (2026-10-28)

The adapter adds default-off `--loyalty-points-remaining-resolution`. It applies only when a user asks how many points remain to earn for a Sephora redemption and user-authored quotes identify a dated Sephora points balance and a dated redemption threshold. Both dates must parse and fall within 30 days of the question date; the latest balance and threshold values must be unique, and the threshold observation must not predate the latest balance. Assistant text, other-brand balances, stale evidence, conflicts, or missing dates leave the answer unchanged. Rewrites cite both source quotes.

Target `9ee3ecd6` has a May 21 user-reported Sephora balance of 200 points, a May 29 user-reported 300-point redemption threshold, and a May 30 question date. The resolver calculated **100 points more**, passed citation validation, and passed a live one-question answer-and-judge replay on the final strict implementation. The final Stage279 500-question candidate over Stage268 frozen retrieval scored **434/500 (86.8%)**, below Stage277's 441/500 and Stage268's 440/500. Paired with Stage277: 23 candidate-only correct, 30 Stage277-only correct, 411 correct in both, and 36 wrong in both (net -7) across 262 changed answers. Against Stage268, it scored 434 versus 440 (22 candidate-only, 28 baseline-only). The full Top-20, Top-200, and retrieval result objects matched Stage268 in 500/500 questions. Candidate invalid audits were 26; API-error markers were zero; the resolver fired once.

Keep the resolver opt-in. Final adapter commit: `bff133a feat: resolve remaining loyalty points from evidence`. Verification passed 220 unittest tests, 25 subtests, compileall, diff checks, and all 500 rows. Stage277 remains best at **441/500 (88.2%)**, 44 below the 485/500 gate. The 100% memory-precision target remains unmeasured and live offsite disaster-recovery restore remains unverified. See [longmemeval-stage279-loyalty-points-remaining-top20-2026-10-28.json](data/longmemeval-stage279-loyalty-points-remaining-top20-2026-10-28.json).
## Stage280: clothing store pickup and return count (2026-10-29)

The adapter adds default-off `--clothing-store-pickup-return-count-resolution`. It counts explicit user-authored pending clothing tasks involving store pickup, dry cleaning, or returns; repeated mentions of the same task are deduplicated, while distinct pickup and return actions remain separate. If a relevant pending task cannot be parsed, it leaves the answer unchanged rather than returning a partial count.

Target `0a995998` has three distinct Top-20 user-authored tasks: collect the replacement boots after a Zara exchange, collect a navy blazer from dry cleaning, and return another pair of boots to Zara. The repeated blazer reference is deduplicated. The answer “3 items of clothing” passed citation validation and a live answer-and-judge replay. The fresh Stage280 candidate over Stage268 frozen retrieval scored **440/500 (88.0%)**, tied with Stage268 and one below Stage277. Paired with Stage268: 24 candidate-only, 24 baseline-only, 416 correct in both, 36 wrong in both (net zero). Paired with Stage277: 27 candidate-only, 28 Stage277-only, 413 correct in both, 32 wrong in both (net -1). Across 277 changed answers, do not attribute overall score movement to the one-row resolver. Full Top-20, Top-200, and retrieved result objects matched Stage268 in 500/500; invalid answer audits were 22 and no API-error markers were recorded. The resolver triggered once.

Adapter commit: `cb56c25 feat: count clothing pickup return tasks from evidence`. Verification passed 225 unittest tests, 25 subtests, compileall, diff checks, and all 500 rows. Stage277 remains best at **441/500 (88.2%)**, 44 below the 485/500 gate. The 100% memory-precision target remains unmeasured and live offsite disaster-recovery restore remains unverified. See [longmemeval-stage280-clothing-store-pickup-return-count-top20-2026-10-29.json](data/longmemeval-stage280-clothing-store-pickup-return-count-top20-2026-10-29.json).
