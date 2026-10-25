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

That 69.8% run is a baseline, not a claim about the current code. A scorecard
comparison is meaningful only when both inputs came from the same 500-question
dataset and unchanged evaluator protocol. The answerer and judge used the same
model in the baseline, so judge self-bias remains a limitation. Retrieval
precision, evidence coverage, answer accuracy, and scorecard integrity are
separate measurements; none can substitute for the 485/500 gate.
