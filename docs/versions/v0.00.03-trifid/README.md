# v0.00.03 Trifid — baseline record

| | |
|---|---|
| Version | `v0.00.03` |
| Codename | Trifid |
| Stages | 101–121 |
| Theme | Episodic fast binding, cue and time disambiguation, controlled replay |
| Status | implemented |
| Modules | `civilization.engine.stages` (stage101 … stage121) |
| Contract tests | 20 test files under `tests/engine` |

## Capabilities added

- **Episode binding.** Fast binding of an episode's content into addressable
  episodic cells (`stage101_trifid_episode_binding`).
- **Disambiguation.** Cue-based and temporal disambiguation, plus a temporal
  index, so similar episodes can be told apart by when they happened and by
  which cue was used (`stage102`, `stage103`).
- **Pattern completion.** Partial cues can recover a bound episode
  (`stage104_trifid_episode_pattern_completion`).
- **Controlled replay.** Episode replay with an explicit scheduler, candidate
  construction, and approval before consolidation (`stage105`–`stage108`).
- **Conflict-aware episodic retrieval.** Semantic episode retrieval, a conflict
  guard, conflict review, conflict decision, and conflict-aware retrieval
  (`stage109`–`stage113`).
- **Snapshots and checkpoints.** Episode snapshots, recovery, paired
  checkpoints, checkpoint recovery, retention, and an audit of retained
  checkpoints (`stage114`–`stage119`).
- **Service integration.** A bridge that connects the episodic machinery back
  into the Stage74 memory service (`stage121`).

## Baseline contract

- An episode is bound once and remains addressable; disambiguation never returns
  an episode from a different session.
- Replay never writes history silently: candidates require approval, and the
  decision is recorded.
- Snapshots restore exactly what was snapshotted; a recovery cannot invent or
  reorder episodes.
- Checkpoint retention honours its audit: what the audit says is retained is
  still recoverable afterwards.

## Verify locally

```bash
python -m pip install '.[test]'
pytest -q -k "stage10 or stage11 or stage12"
```

## Boundary

- Episodic structure is explicit and auditable, not an emergent claim about
  human-like memory.
- Replay is scheduled and approval-gated; autonomous self-training is not
  claimed.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Trifid are
contract-level only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
