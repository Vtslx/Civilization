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
counts, explicit API-error rows, and exact source-session chunk recall at Top-20
and Top-200 with MRR@200. Source-session coverage is a diagnostic derived by
matching the benchmark's answer-session IDs to the conversation chunks returned
by Civilization; it is not the answerer accuracy metric. A partial score does
not count as a fully correct answer for the 485-answer gate.

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

That 69.8% run is a baseline, not a claim about the current code. A scorecard
comparison is meaningful only when both inputs came from the same 500-question
dataset and unchanged evaluator protocol. The answerer and judge used the same
model in the baseline, so judge self-bias remains a limitation. Retrieval
precision, evidence coverage, answer accuracy, and scorecard integrity are
separate measurements; none can substitute for the 485/500 gate.
