# Orion relevance ranking

## Scope

Orion reads every active cell allowed by the caller's session, source, and
memory-system filters, ranks those candidates in-process, and returns the
requested limit. The ranking strategy is replaceable at store construction;
the default strategy has no model or network dependency.

## Ranking signals

`OrionRelevanceRanker` uses a fielded BM25F score over `summary`, `content`, and
`source`. The summary has weight 2.0, the content weight 1.0, and the source
weight 0.2. Field length normalization limits the effect of long conversation
transcripts, while inverse document frequency gives more weight to terms that
distinguish one memory from the rest of the candidate set. English words and
numbers are normalized; contiguous CJK text is tokenized into character
bigrams.

The final score includes the BM25F relevance score, a small exact-phrase bonus,
bounded confidence and importance priors (at most 0.05 each), and the existing
link score. Approved Helix weights remain effective. A query that explicitly
asks for the current or latest fact uses descending `time_index` only to break
score ties; ordinary queries retain ascending chronological tie-breaking. Cell
ID is the final stable tie-break.

Every `MemoryReadResult` includes a `score_breakdown` with BM25F, phrase,
importance, confidence, and link components. The service can expose that result
for diagnostics without changing the `read()` call signature. Session/global
selection and authorization remain the responsibility of the existing router
and task policy.

## Compatibility and recovery

The ranker runs only during reads. It does not mutate cells, links, learned
weights, journals, or snapshots, so persistence replay and crash-recovery
semantics are unchanged. Injecting a custom ranker is optional; existing store
constructors continue to use the deterministic default.

## Limits and acceptance

This is a lexical ranker. It improves term weighting, field weighting, and
recency ordering, but does not create semantic candidates for vocabulary with
no overlap. The current implementation tokenizes the eligible corpus for each
read, as the previous implementation also scanned the full store; a persistent
inverted index or semantic candidate provider needs separate workload and
recovery measurements before adoption.

Unit fixtures check precision-sensitive ordering, long distractors, query
filters, English/CJK latest intent, score diagnostics, and compatibility with
approved Helix links. These fixtures are regression checks, not a public
benchmark. A LongMemEval claim requires a full run with the same data, model,
answerer, judge, adapter, and cutoff as the recorded baseline.
