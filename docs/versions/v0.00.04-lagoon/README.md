# v0.00.04 Lagoon — baseline record

| | |
|---|---|
| Version | `v0.00.04` |
| Codename | Lagoon |
| Stages | 122–132 |
| Theme | Cross-episode schema consolidation, provenance, conflict review |
| Status | implemented |
| Modules | `experiments/civilization_transformer_qwen3/analysis/stage122…stage132` |
| Contract tests | 11 test files under `experiments/civilization_transformer_qwen3/tests` |

## Capabilities added

- **Consolidation clusters.** Episodes are clustered before any schema is
  proposed, so consolidation works on groups rather than single episodes
  (`stage122_lagoon_consolidation_clusters`).
- **Schema candidates.** Candidate abstractions are built from clusters and
  remain candidates until approved (`stage123_lagoon_schema_candidates`).
- **Approved consolidation.** Only approved candidates become semantic schema
  memory (`stage124_lagoon_approved_consolidation`).
- **Schema retrieval and provenance.** Semantic retrieval over approved schemas,
  plus provenance queries that trace a schema back to the episodes it came from
  (`stage125`, `stage126`).
- **Conflict handling on schemas.** A schema conflict guard, conflict-aware
  recall, conflict review, and conflict decision, so contradictory schemas are
  surfaced rather than merged away (`stage127`–`stage130`).
- **Decision-aware recall.** Recall that respects recorded conflict decisions
  (`stage131_lagoon_decision_aware_recall`).
- **Release gate.** A gate that replays the whole version contract and reports
  named boolean checks (`stage132_lagoon_release_gate`).

## Baseline contract

- A consolidated schema always has provenance: it can be traced to the episodes
  that produced it.
- Approval is required before promotion; unreviewed candidates never enter
  semantic memory.
- Conflicting schemas are preserved for review — the system does not silently
  pick a winner when the evidence is ambiguous.
- The release gate is reproducible and writes an auditable summary.

## Verify locally

```bash
python -m pip install '.[test]'
pytest experiments/civilization_transformer_qwen3/tests -q -k "stage12 or stage13"
```

## Boundary

- Schemas are consolidated from explicit episodes; automatic extraction of
  facts from arbitrary text is not claimed.
- Conflict decisions are recorded, not "discovered truths"; the version does not
  claim autonomous fact arbitration.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Lagoon are
contract-level only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
