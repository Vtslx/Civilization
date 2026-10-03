# Qwen3-0.6B Real-Model Experiment

This experiment loads the local Qwen3-0.6B checkpoint in offline mode. It supports both the Stage 23 read-only hidden-state baseline and the Stage 24 frozen-base Civilization Adapter experiment.

Qwen3 remains frozen in both stages. Stage 24 trains only the Adapter and diagnostic heads, and injects Memory / State / Rule signals through a temporary decoder-layer hook.

Read-only baseline:

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_hidden_state_baseline
```

Civilization Adapter benchmark:

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_civilization_adapter
```

The Adapter benchmark evaluates layers 2, 16, and 27 independently. Ablations use centroids built only from full-context training representations, so disabled, zero-scale, missing-path, empty-context, and wrong-context results cannot redefine their own codebook.

Multi-layer Adapter benchmark:

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_multilayer_adapter
```

Surface-group flip repair benchmark:

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_surface_flip_repair
```

## Stage 27 real task migration

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_real_task_migration
```

External public benchmark caches are required by default. Set
`ALLOW_DATASET_DOWNLOAD=1` only when intentionally creating the local cache.

## Stage 28 grounded context repair

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_real_task_grounding_repair
```

This repair benchmark keeps Qwen3 frozen, rebuilds real-task Memory / State / Rule
contexts with `grounded_v1`, and reports both fixed-centroid metrics and
answer-option readout metrics. It is a smoke/medium repair pass, not a full
three-seed migration matrix.

The repair benchmark trains complete five-label surface groups and retains the same fixed full-context centroid rule for every ablation.

## Stage 29 evidence-to-answer alignment

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_evidence_answer_alignment
```

Stage 29 trains two independent `dual_16_24` Adapters. The local-only route
keeps RTE, CB, and BoolQ as zero-shot evaluations; the few-shot route adds only
their isolated train records. Both routes use the same held-out records. Fixed
centroids are built once from full-context train hidden states, while direct
answer-option scoring reads the Adapter delta relative to the disabled base
model.

## Stage 30 memory/rule necessity repair

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_memory_rule_necessity_repair
```

Stage 30 adds paired memory-only, rule-only, and memory/rule-conflict contexts.
It is a repair experiment for Stage 29 failures: Memory/Rule path drops,
fixed-centroid surface flip, and local retention. The experiment still freezes
Qwen3 and writes only Adapter-only checkpoints.

## Stage 31 binary path diagnostic

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_binary_path_diagnostic
```

Stage 31 narrows the problem to local two-option diagnostics. It trains
independent `dual_16_24` Adapters for memory-only, rule-only, memory/rule
conflict, and a combined mixed diagnostic view. Stage A optimizes Adapter-delta
answer-option readout first; Stage B adds only a small full-hidden centroid
regularizer. RTE, CB, and BoolQ are intentionally excluded from this diagnostic
stage.

## Stage 32 path-specific binary diagnostic

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_path_specific_binary_diagnostic
```

Stage 32 keeps the Stage 31 binary diagnostic data and training schedule but
switches the Adapter variant to `path_specific_v2`. This variant splits base,
memory, rule, and state updates into independent residual paths with separate
scales and trace fields, so path ablation can test whether Memory and Rule are
actually necessary instead of merely present in the trace.

## Stage 33 context/readout alignment

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_context_readout_alignment
```

Stage 33 keeps Qwen3 frozen and diagnoses why Stage 32 path-specific deltas do
not map cleanly to answer-option or fixed-centroid readouts. It records
context-to-answer separability, raw Adapter delta readout, projected delta
readout, and fixed-centroid readout. The `PathReadoutProjector` is a diagnostic
readout layer only; it does not enter Qwen3 forward.

## Stage 34 projected binary path necessity

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_projected_binary_path_necessity
```

Stage 34 promotes the Stage 33 `PathReadoutProjector` into the official binary
path necessity readout while keeping Qwen3 frozen. It reruns the Stage 31/32
binary Memory/Rule diagnostic with `path_specific_v2`, exports projected raw
delta and fixed-centroid metrics side by side, and uses projected path
ablation drops as the main evidence for Memory-only and Rule-only necessity.

## Stage 35 rule-conditioned conflict curriculum

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_rule_conflict_curriculum
```

Stage 35 adds a rule-conditioned conflict v2 dataset and a five-stage
curriculum: Memory-only, Rule-only, Rule-conditioned conflict, balanced
combined training, and low-weight fixed-centroid regularization. The official
path gates use projected delta readout; raw delta and fixed-centroid metrics
remain diagnostic and must be reported separately.

## Stage 36 full-hidden fixed-centroid alignment

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_full_hidden_centroid_alignment
```

Stage 36 keeps the Stage 35 projected path gates intact and repairs raw
full-hidden fixed-centroid readout. It trains a full-hidden alignment pass with
dynamic centroid separation and combined curriculum weighting, reports
projected delta, fixed-centroid, and projected full-hidden metrics side by side,
and only allows returning to multiclass repair when both projected path gates
and raw fixed-centroid gates pass.

## Stage 40 Memory delta gradient diagnostic

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_memory_delta_gradient_diagnostic --preferred-device cuda
```

Stage 40 audits five-candidate target mapping and the differentiable Memory
delta gradient path. It trains only the Memory path and projected readout while
Qwen3, Base, State, Rule, and full-hidden projection remain frozen.

## Stage 41 Rule, conflict, and combined group recovery

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_rule_conflict_group_recovery --preferred-device cuda
```

Stage 41 restores the Stage 40 checkpoint, directly supervises the
differentiable Rule delta, and runs strict Rule-only, Rule-conditioned conflict,
and balanced combined five-candidate curricula. Each stage is fail-fast and
must retain the previously passed Memory gate before the next stage can run.

## Stage 42 raw full-hidden centroid integration

```bash
.venv/bin/python -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_group_full_hidden_centroid_integration --preferred-device cuda
```

Stage 42 restores the passed Stage 41 combined checkpoint and writes the
Memory/Rule signal into raw final hidden states. Train centroids are rebuilt
for the representation before and after alignment, while every held-out and
ablation evaluation at a checkpoint shares one fixed train-only centroid set.
The alignment step uses balanced Memory, Rule, and Conflict groups so dynamic
centroid separation contains multiple surfaces for every option label.
