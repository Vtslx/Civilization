# v0.00.02 Orion — baseline record

| | |
|---|---|
| Version | `v0.00.02` |
| Codename | Orion |
| Stages | 73–100 |
| Theme | Multi-system memory (working / episodic / semantic / procedural) |
| Status | implemented |
| Modules | `civilization.engine.stages` (stage73 … stage99) |
| Contract tests | 19 test files under `tests/engine` |

## Capabilities added

- **Memory kernel.** A four-system store (working, episodic, semantic,
  procedural) with cells, importance, confidence, expiry, and typed links
  (`stage73_orion_memory_kernel`).
- **Memory-aware inference.** A memory service that retrieves before a
  prediction and injects selected cells into the ordinary prediction context,
  returning an Orion trace with the retrieved and injected cell ids
  (`stage74_orion_memory_service`). The adapter context bridge feeds the same
  cells toward the structured adapter path (`stage75`).
- **Consolidation and promotion.** Replay-based consolidation and approved
  semantic promotion, so repeated episodes can become semantic memory
  (`stage76`, `stage78`).
- **Global memory.** Cross-session retrieval, global maintenance, a persistent
  global store with recovery, lifecycle management, and a store doctor
  (`stage79`, `stage82`, `stage91`–`stage96`).
- **Conflict handling.** Conflict-aware retrieval, resolution, and a resolution
  trace, so contradictory cells can be surfaced and decided instead of silently
  overwritten (`stage83`–`stage85`).
- **Policies and evaluation.** Task memory policy, multi-session evaluation, a
  global evidence benchmark, and retention with a regression check
  (`stage86`, `stage89`, `stage90`, `stage98`, `stage99`).

## Baseline contract

- Memory writes are validated and every cell belongs to exactly one memory
  system with an auditable origin.
- Retrieval is session-scoped by default; cross-session reads happen only
  through the global path, and isolated sessions never leak into each other.
- Consolidation and promotion are approval-gated: replayed knowledge is not
  silently promoted into semantic memory.
- The global store can be saved, reloaded, and recovered, and a retention pass
  cannot silently delete cells that policy marks as retained.

## Verify locally

```bash
python -m pip install '.[test]'
pytest -q -k "stage7 or stage8 or stage9"
```

## Boundary

- Memory here is explicit: cells are written through the API. Automatic
  extraction from free-form text is not claimed.
- Vector retrieval is not part of this version; ranking uses the lexical and
  policy signals that the code implements.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Orion are
contract-level only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
