# v0.00.06 Rosette — baseline record

| | |
|---|---|
| Version | `v0.00.06` |
| Codename | Rosette |
| Stages | 139–150 |
| Theme | Typed context packets, conflict budgets, multi-scale honeycomb graph |
| Status | implemented |
| Modules | `experiments/civilization_transformer_qwen3/analysis/stage139…stage150` |
| Contract tests | 12 test files under `experiments/civilization_transformer_qwen3/tests` |

## Capabilities added

- **Typed context packets.** Retrieved memory is assembled into a typed packet
  instead of an unstructured list (`stage139_rosette_context_packet`).
- **Atomic conflict budgets.** A budget bounds how much contradictory material a
  packet may carry, so a single conflict cannot consume the whole context
  (`stage140_rosette_atomic_budget`).
- **Validation and snapshots.** A validator for packets, packet snapshots, and
  snapshot recovery (`stage141`–`stage143`).
- **Honeycomb graph.** A multi-scale graph over memory units with traversal, a
  structural validator, snapshots, and recovery (`stage145`–`stage149`).
- **Release gates.** Two reproducible gates: one for the packet contract, one
  for the graph contract (`stage144`, `stage150`).

## Baseline contract

- A packet that violates its type contract is rejected rather than partially
  used.
- The conflict budget is enforced: exceeding it fails validation instead of
  silently dropping evidence.
- Snapshots restore exactly the validated structure; recovery cannot fabricate
  nodes or edges.
- Both release gates are reproducible and write auditable summaries.

## Verify locally

```bash
python -m pip install '.[test]'
pytest experiments/civilization_transformer_qwen3/tests -q -k "stage14 or stage150"
```

## Boundary

- The "honeycomb" naming describes a multi-scale graph structure in this code;
  it is an engineering label, not a claim about biological tissue.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Rosette are
contract-level only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
