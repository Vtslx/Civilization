# v0.00.07 Helix — baseline record

| | |
|---|---|
| Version | `v0.00.07` |
| Codename | Helix |
| Stages | 151–160 |
| Theme | Learned retrieval-path weights, with recovery and calibration gates |
| Status | implemented (current line) |
| Modules | `civilization.engine.stages` (stage151 … stage160) |
| Contract tests | 10 test files under `tests/engine` |
| Experiments | [native memory paired A/B](experiments/memory-on-off-ab.md) |

## Capabilities added

- **Path weight model.** Retrieval paths carry weights that influence ranking
  (`stage151_helix_path_weight_model`).
- **Feedback signals.** Outcomes and feedback are collected into signals that
  can justify a weight change (`stage152_helix_feedback_signal`).
- **Update rule.** A rule proposes weight updates from those signals rather than
  applying them directly (`stage153_helix_weight_update_rule`).
- **Approval-gated mutation.** Proposed updates require approval before they
  change retrieval behaviour (`stage154_helix_approval_gated_mutation`).
- **Weighted retrieval.** Ranking uses the approved weights
  (`stage155_helix_weighted_retrieval`).
- **Conflict guard, snapshots, recovery.** Weight conflicts are detected,
  weights can be snapshotted, and a snapshot can be restored
  (`stage156`–`stage158`).
- **Calibration gate and release gate.** A calibration check plus a
  reproducible version gate (`stage159`, `stage160`).

## Baseline contract

- Learned weights never change retrieval without an approval record.
- A weight conflict is reported and blocks silent overwriting.
- Snapshot recovery restores the exact approved weight state.
- The calibration gate and release gate are reproducible and write auditable
  summaries.

## Measured comparison

This is the only version in the line with a recorded measurement: a paired A/B
of native (Orion) memory on vs. off on a fixed 25-question synthetic set, three
rounds, same model and options, only `read_memory` toggled.

| Metric | Result |
|---|---|
| Answerable questions, memory on | 63/63 (100%) |
| Answerable questions, memory off | 17/63 (27.0%) |
| Paired result | 46 pairs won only with memory on; 0 pairs won only with memory off |
| API errors | 0 in both conditions |

Full design, limitations, supporting benchmark context, and the reproduction
command: [experiments/memory-on-off-ab.md](experiments/memory-on-off-ab.md)
(English) · [中文](experiments/memory-on-off-ab.zh-CN.md).
Recorded data: [data/memory-on-off-ab-recorded.json](experiments/data/memory-on-off-ab-recorded.json).
Reproduction script: [scripts/memory_ab.py](experiments/scripts/memory_ab.py).

**Read the scope before quoting the number.** It is 25 unique synthetic
questions repeated three times in one deployment against itself — not a public
benchmark score and not a claim of general superiority.

## Verify locally

```bash
python -m pip install '.[test]'
pytest -q -k "stage15 or stage160"
```

## Boundary

- Vector retrieval is not part of this deployment; ranking uses the lexical and
  policy signals the code implements.
- Retrieval weights are learned and approved inside the system; no external
  training signal is claimed.
- Session-store persistence across a process restart, global-store behaviour,
  and free-form conversation quality are outside the recorded experiment.

[简体中文](README.zh-CN.md) · [version index](../README.md)
