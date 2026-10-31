# Stored-memory precision

The LongMemEval answerer result and stored-memory precision are separate
measurements. LongMemEval asks whether the answerer produced the right answer
from retrieved evidence. Stored-memory precision asks whether the memory cells a
session keeps are actually supported by the transcript they came from. A system
can pass one and fail the other: retrieval can surface a true answer while a
journal quietly holds an invented fact, and a store can be entirely true while
the answerer reasons poorly.

This page defines the second metric, the exact command that measures it, and
what a 100% result does and does not mean.

## Definition

For a session store `S` and a transcript `T`:

```
precision(S, T) = supported cells / all stored cells
```

A cell is **supported** when one of these holds:

1. It has provenance with a `source_id` and `quote`, the `source_id` resolves to
   a transcript span, and the normalized quote is a contiguous substring of that
   span. With `--require-user-authorship`, the cited span must be user-authored.
2. It is a consolidation/derived cell with `derived_from_cell_ids`, every parent
   exists, and every parent is supported. The summary is not re-quoted because
   it is not a verbatim claim.

A cell is **unsupported** when its quote is fabricated or its source span is
missing, **missing_provenance** when it has neither provenance nor parents,
**role_violation** when a user-authored claim cites assistant text under the
user-authorship policy, and **unresolved_derivation** when a parent is missing,
unsupported, or part of a cycle. The audit fails closed: an empty store is
reported as vacuously 1.0 and marked `empty=true`, so it cannot be presented as
production evidence.

Normalization is Unicode NFKC plus case folding plus whitespace collapsing.
Quotes must be exact after that normalization; paraphrases are not accepted.

## Run it

```bash
civilization memory-precision \
  --state-dir var/state \
  --session-id support-42 \
  --transcript transcript.jsonl \
  --target 1.0 \
  --require-user-authorship \
  --output memory-precision.json
```

`transcript.jsonl` has one JSON object per turn:

```json
{"source_id": "turn-1", "role": "user", "text": "I bought two blue chairs on 2023-03-04."}
{"source_id": "turn-2", "role": "assistant", "text": "You could consider a matching table."}
```

The command hydrates each session from its snapshot plus journal, audits every
cell, prints the report, and exits `1` when the aggregate precision is below the
target. Omit `--session-id` to audit every session under `--state-dir`.

## Provenance shape

Attach provenance when writing the cell:

```json
{
  "provenance": {
    "source_id": "turn-1",
    "quote": "two blue chairs",
    "role": "user"
  }
}
```

Multiple spans are allowed:

```json
{
  "provenance": {
    "spans": [
      {"source_id": "turn-1", "quote": "two blue chairs", "role": "user"},
      {"source_id": "turn-4", "quote": "delivered on Monday", "role": "user"}
    ]
  }
}
```

A consolidation cell cites its parents instead:

```json
{"derived_from_cell_ids": ["episodic-000001", "episodic-000002"]}
```

The API already accepts `metadata` on `write_memory` and `consolidate_memory`;
no schema change is needed.

## What 100% means

A 100% result means **every cell in the audited session was supported by an
exact quoted span, or by supported parents**. It does not mean:

- that the whole production corpus was audited;
- that the transcript itself is true (the transcript is the ground truth here);
- that retrieval returns only relevant memories;
- that LongMemEval accuracy is 97%.

Report the session ids, cell count, transcript hash, policy, and command with
the report. A 100% result over an empty or synthetic session is a smoke test,
not evidence about user data.

## Verification

`tests/engine/test_memory_precision.py` covers supported quotes, fabricated
quotes, missing sources, missing provenance, assistant-only evidence under the
user-authorship policy, supported and unsupported derivations, cycles, empty
stores, transcript loading, and the CLI exit codes. The CLI writes the same JSON
the tests inspect, so the report and the test evidence cannot drift apart.
