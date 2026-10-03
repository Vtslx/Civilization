# v0.00.05 Eagle — baseline record

| | |
|---|---|
| Version | `v0.00.05` |
| Codename | Eagle |
| Stages | 133–138 |
| Theme | Task traces and approved procedural memory |
| Status | implemented |
| Modules | `experiments/civilization_transformer_qwen3/analysis/stage133…stage138` |
| Contract tests | 6 test files under `experiments/civilization_transformer_qwen3/tests` |

## Capabilities added

- **Task traces.** Execution traces are captured as first-class records
  (`stage133_eagle_task_trace`).
- **Skill candidates.** Repeated traces yield candidate procedures rather than
  directly minted skills (`stage134_eagle_skill_candidates`).
- **Approved skills.** A candidate becomes procedural memory only after approval
  (`stage135_eagle_approved_skill`).
- **Skill retrieval.** Approved procedures are retrievable and can be applied to
  later tasks (`stage136_eagle_skill_retrieval`).
- **Skill conflict guard.** Contradictory procedures are detected instead of
  both being applied (`stage137_eagle_skill_conflict_guard`).
- **Release gate.** A reproducible gate over the version contract
  (`stage138_eagle_release_gate`).

## Baseline contract

- A skill always has a traceable origin in recorded task traces.
- Unapproved candidates never reach procedural memory.
- Conflicting skills are reported, not silently resolved.
- The release gate is reproducible and writes an auditable summary.

## Verify locally

```bash
python -m pip install '.[test]'
pytest experiments/civilization_transformer_qwen3/tests -q -k "stage13"
```

## Boundary

- Procedures are derived from explicit traces in this system; the version does
  not claim autonomous skill learning from open-ended interaction.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Eagle are
contract-level only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
