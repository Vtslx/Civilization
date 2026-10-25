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

That 69.8% run is a baseline, not a claim about the current code. A scorecard
comparison is meaningful only when both inputs came from the same 500-question
dataset and unchanged evaluator protocol. The answerer and judge used the same
model in the baseline, so judge self-bias remains a limitation. Retrieval
precision, evidence coverage, answer accuracy, and scorecard integrity are
separate measurements; none can substitute for the 485/500 gate.
