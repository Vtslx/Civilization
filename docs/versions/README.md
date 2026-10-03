# Version baselines

This directory records one baseline per implemented version of the Civilization
v1 line, so that a change can be compared against a named reference instead of
against memory or against the latest code.

| Version | Codename | Stages | Baseline record | Measured comparison |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | [v0.00.01-sun](v0.00.01-sun/README.md) | contract baseline only |
| `v0.00.02` | Orion | 73–100 | [v0.00.02-orion](v0.00.02-orion/README.md) | contract baseline only |
| `v0.00.03` | Trifid | 101–121 | [v0.00.03-trifid](v0.00.03-trifid/README.md) | contract baseline only |
| `v0.00.04` | Lagoon | 122–132 | [v0.00.04-lagoon](v0.00.04-lagoon/README.md) | contract baseline only |
| `v0.00.05` | Eagle | 133–138 | [v0.00.05-eagle](v0.00.05-eagle/README.md) | contract baseline only |
| `v0.00.06` | Rosette | 139–150 | [v0.00.06-rosette](v0.00.06-rosette/README.md) | contract baseline only |
| `v0.00.07` | Helix | 151–160 | [v0.00.07-helix](v0.00.07-helix/README.md) | **native memory paired A/B** |
| `v0.00.08` | Crab | — | not implemented | — |

Translations: [简体中文](README.zh-CN.md).

## What a baseline record contains

1. **Scope** — the codename, stage range, and the modules that belong to the
   version.
2. **Capabilities added** — what a deployment gains at this version, stated in
   terms of behaviour, not marketing.
3. **Baseline contract** — the invariants the version must hold, each one backed
   by tests that ship in this repository.
4. **Verification command** — how to reproduce the contract result locally.
5. **Boundary** — what the version deliberately does not claim.

## What is comparable, and what is not

- A **contract baseline** (tests that must pass) is directly comparable across
  revisions: run the listed tests and compare pass/fail.
- A **measured comparison** exists only where an experiment was actually run and
  recorded. In this repository that is the native-memory paired A/B under
  `v0.00.07-helix/experiments/`. It measures one deployment against itself with
  one field toggled, and it is not a benchmark leaderboard entry.
- Public-benchmark numbers (LongMemEval, BEAM) come from a separate evaluation
  campaign with its own answerers and judges. They appear as positioning context
  inside the experiment report and are **not** cross-comparable with the paired
  A/B percentages.
- Version numbers describe the **implementation line**, not accuracy. A later
  version is not by definition better on a benchmark; it is a different, tested
  capability set.

## Publication policy for this directory

- Baseline records are written for public release: no internal paths, hosts,
  credentials, or internal planning documents. Only what the code, the tests,
  and the recorded measurements support.
- Documents are published in English (canonical) and Simplified Chinese.
  Corrections and additional languages are welcome as pull requests.
- When a claim cannot be backed by something in this repository, it is not
  made. Missing measurements are reported as missing.
