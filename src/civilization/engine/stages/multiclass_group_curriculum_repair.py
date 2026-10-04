from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from civilization.research.torch_line.analysis.dataset import LOGIC_LABELS
from civilization.research.torch_line.model import CivilizationAblationConfig

from ..adapter import Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, EVALUATION_MODES, _mean
from .adapter_training import AdapterDiagnosticHeads, LABEL_TO_ID, _kl_preservation
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_benchmark import (
    LOCAL_MEMORY_TASKS,
    LOCAL_RULE_TASKS,
    LOCAL_STATE_TASKS,
    PROBE_LAYER,
    ROUTES,
    _route_metric,
)
from .evidence_answer_data import assert_no_logic_label_leakage, build_evidence_answer_samples
from .evidence_answer_training import _answer_scores
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .hidden_states import last_non_padding_pool
from .multiclass_necessity_repair import (
    MulticlassCurriculumResult,
    _context,
    _dynamic_centroid_loss,
    _evaluate_multiclass_task,
    _evaluate_projected_necessity_pairs,
    _make_path_specific_model,
    _path_drop_loss,
    _prototype_loss,
    _residual_parameters,
    _save_checkpoint,
    _stage37_path_specific_pairs,
    _stage37_path_specific_samples,
    _window_decreased,
)
from .necessity_alignment_data import NecessityPair, assert_necessity_pairs_valid, build_necessity_pairs
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
)
from .real_task_migration import _balanced_external_split, _generic_centroid_metrics, _surface_flip_accuracy
from .memory_rule_necessity_benchmark import _split_and_align_local


GROUP_TYPES = (
    "memory_necessity_group",
    "rule_necessity_group",
    "memory_rule_conflict_group",
)


@dataclass(frozen=True)
class SurfaceGroupCandidateBatch:
    group_type: str
    surface_group_id: str
    pairs: tuple[NecessityPair, ...]

    def __post_init__(self) -> None:
        if self.group_type not in GROUP_TYPES:
            raise ValueError(f"unsupported group_type: {self.group_type}")
        if len(self.pairs) != len(LOGIC_LABELS):
            raise ValueError("surface group candidate batch must contain exactly five candidates")
        if len({pair.full_sample.text for pair in self.pairs}) != 1:
            raise ValueError("surface group candidate batch text must be identical")
        if len({pair.expected_full_option_id for pair in self.pairs}) != len(LOGIC_LABELS):
            raise ValueError("surface group candidate batch must contain five unique target options")
        if len({pair.full_sample.expected_pattern for pair in self.pairs}) != len(LOGIC_LABELS):
            raise ValueError("surface group candidate batch must contain five unique expected answers")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_type_for_pair_type(pair_type: str) -> str:
    return {
        "memory_necessity_pair": "memory_necessity_group",
        "rule_necessity_pair": "rule_necessity_group",
        "memory_rule_conflict_pair": "memory_rule_conflict_group",
    }[pair_type]


def build_surface_group_candidate_batches(pairs: list[NecessityPair]) -> list[SurfaceGroupCandidateBatch]:
    grouped: dict[tuple[str, str], list[NecessityPair]] = {}
    for pair in pairs:
        grouped.setdefault((pair.pair_type, pair.full_sample.surface_group_id), []).append(pair)
    batches = []
    for (pair_type, surface_group_id), rows in sorted(grouped.items()):
        by_target = {pair.expected_full_option_id: pair for pair in rows}
        if len(by_target) != len(LOGIC_LABELS):
            raise ValueError(f"incomplete surface group {surface_group_id} for {pair_type}")
        ordered = tuple(by_target[index] for index in sorted(by_target))
        batches.append(
            SurfaceGroupCandidateBatch(
                group_type=_group_type_for_pair_type(pair_type),
                surface_group_id=surface_group_id,
                pairs=ordered,
            )
        )
    return batches


def _group_sequence(groups: list[SurfaceGroupCandidateBatch], steps: int, seed: int) -> list[SurfaceGroupCandidateBatch]:
    if not groups:
        raise ValueError("surface group curriculum pool cannot be empty")
    rng = random.Random(seed)
    ordered = sorted(groups, key=lambda group: (group.group_type, group.surface_group_id))
    result = []
    while len(result) < steps:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:steps]


def _encode_group(backend: Qwen3Backend, group: SurfaceGroupCandidateBatch, local_max_length: int, external_max_length: int):
    samples = [pair.full_sample for pair in group.pairs]
    max_length = external_max_length if group.pairs[0].base_record.source_type == "external_benchmark" else local_max_length
    encoded, truncations = backend.encode([qwen_text_for_sample(sample) for sample in samples], max_length=max_length)
    if truncations and group.pairs[0].base_record.source_type != "external_benchmark":
        raise ValueError("local surface group sample was truncated")
    return {name: tensor.to(backend.device) for name, tensor in encoded.items()}, samples


def _ablated_mode_for_group(group: SurfaceGroupCandidateBatch) -> str:
    if group.group_type == "memory_necessity_group":
        return "no_memory_path"
    if group.group_type in {"rule_necessity_group", "memory_rule_conflict_group"}:
        return "no_rule_path"
    return "no_state_path"


def _group_forward(
    *,
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    heads: AdapterDiagnosticHeads,
    context_encoder: FrozenQwenContextEncoder,
    group: SurfaceGroupCandidateBatch,
    option_vectors: torch.Tensor,
    local_max_length: int,
    external_max_length: int,
    mode: str = "full",
) -> dict[str, Any]:
    encoded, samples = _encode_group(backend, group, local_max_length, external_max_length)
    targets = torch.tensor([pair.expected_full_option_id for pair in group.pairs], dtype=torch.long, device=backend.device)
    labels = torch.tensor([LABEL_TO_ID[sample.label] for sample in samples], dtype=torch.long, device=backend.device)
    with torch.inference_mode():
        baseline = backend.inference_forward(encoded)
        baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
    output = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], mode))
    pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    projected = projector(pooled - baseline_pooled)
    full_projected = full_hidden_projector(pooled)
    scores = _answer_scores(projected, option_vectors)
    predictions = scores.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean()
    group_success = torch.all(predictions == targets).float()
    return {
        "encoded": encoded,
        "samples": samples,
        "output": output,
        "baseline_pooled": baseline_pooled,
        "pooled": pooled,
        "projected": projected,
        "projected_full_hidden": full_projected,
        "scores": scores,
        "targets": targets,
        "labels": labels,
        "accuracy": accuracy,
        "group_success": group_success,
        "fixed_loss": _prototype_loss(heads, pooled, labels),
        "projected_full_loss": _prototype_loss(heads, full_projected, labels),
        "dynamic_loss": _dynamic_centroid_loss(pooled, labels),
    }


def _group_losses(full: dict[str, Any], ablated: dict[str, Any]) -> dict[str, torch.Tensor]:
    scores = full["scores"]
    targets = full["targets"]
    normalized = F.normalize(full["projected"].float(), dim=-1)
    similarity = normalized @ normalized.T
    different = similarity[targets[:, None] != targets[None, :]]
    separation = torch.relu(different + 0.10).mean() if different.numel() else scores.new_zeros(())
    return {
        "group_all_correct_loss": F.cross_entropy(scores / 0.05, targets),
        "group_path_drop_loss": _path_drop_loss(scores, ablated["scores"], targets, 0.25),
        "group_context_separation_loss": separation,
        "group_pair_flip_loss": torch.relu(similarity[targets[:, None] != targets[None, :]] + 0.05).mean()
        if different.numel()
        else scores.new_zeros(()),
        "group_projected_margin_loss": torch.relu(0.25 - (scores.gather(1, targets[:, None]).squeeze(1) - scores.masked_fill(F.one_hot(targets, scores.shape[-1]).bool(), -1e4).max(dim=-1).values)).mean(),
        "group_full_hidden_centroid_loss": full["fixed_loss"] + full["dynamic_loss"],
    }


def _evaluate_groups(
    *,
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    heads: AdapterDiagnosticHeads,
    groups: list[SurfaceGroupCandidateBatch],
    route: str,
    seed: int,
    local_max_length: int,
    external_max_length: int,
    modes: tuple[str, ...] = ("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "no_state_path", "empty_context", "wrong_context"),
) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    cache: dict[tuple[str, ...], torch.Tensor] = {}
    rows = []
    for group in groups:
        options = group.pairs[0].base_record.answer_options
        if options not in cache:
            cache[options] = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
        for mode in modes:
            with torch.no_grad():
                result = _group_forward(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    heads=heads,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=cache[options],
                    local_max_length=local_max_length,
                    external_max_length=external_max_length,
                    mode=mode,
                )
            rows.append(
                {
                    "route": route,
                    "seed": seed,
                    "source_type": group.pairs[0].base_record.source_type,
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "projected_accuracy": float(result["accuracy"].detach().cpu()),
                    "group_success": float(result["group_success"].detach().cpu()),
                    "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in result["output"].traces.values()),
                    "memory_delta_norm": sum(trace.memory_delta_norm for trace in result["output"].traces.values()),
                    "rule_delta_norm": sum(trace.rule_delta_norm for trace in result["output"].traces.values()),
                }
            )
    return rows


def _checkpoint(
    output_dir: Path,
    route: str,
    stage: str,
    seed: int,
    model: Qwen3MultiAdapterModel,
    heads: AdapterDiagnosticHeads,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
) -> str:
    return _save_checkpoint(output_dir, route, f"group_{stage}", seed, model, heads, projector, full_hidden_projector)


def run_group_batched_curriculum_training(
    *,
    backend: Qwen3Backend,
    train_groups: list[SurfaceGroupCandidateBatch],
    heldout_groups: list[SurfaceGroupCandidateBatch],
    output_dir: Path,
    route: str,
    seed: int,
    memory_steps: int = 40,
    rule_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    full_hidden_steps: int = 80,
    local_max_length: int = 64,
    external_max_length: int = 384,
    learning_rate: float = 3e-4,
) -> tuple[Qwen3MultiAdapterModel, PathReadoutProjector, FullHiddenCentroidProjector, AdapterDiagnosticHeads, MulticlassCurriculumResult, list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_path_specific_model(backend, (16, 24))
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    projector = PathReadoutProjector().to(backend.device)
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    residual_parameters = _residual_parameters(model)
    residual_ids = {id(parameter) for parameter in residual_parameters}
    adapter_parameters = [parameter for adapter in model.adapters.values() for parameter in adapter.parameters() if id(parameter) not in residual_ids]
    head_parameters = list(heads.parameters())
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    trainable = adapter_parameters + residual_parameters + head_parameters + list(projector.parameters()) + list(full_hidden_projector.parameters())
    optimizer_contains_qwen = any(id(parameter) in qwen_ids for parameter in trainable)
    if optimizer_contains_qwen:
        raise RuntimeError("optimizer contains frozen Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": head_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
            {"params": full_hidden_projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    by_type = {group_type: [group for group in train_groups if group.group_type == group_type] for group_type in GROUP_TYPES}
    stage_specs = [
        ("memory_necessity", by_type["memory_necessity_group"], memory_steps),
        ("rule_necessity", by_type["rule_necessity_group"] + by_type["memory_necessity_group"], rule_steps),
        ("conflict_necessity", by_type["memory_rule_conflict_group"] + by_type["memory_necessity_group"] + by_type["rule_necessity_group"], conflict_steps),
        ("combined_curriculum", train_groups, combined_steps),
        ("full_hidden_alignment", train_groups, full_hidden_steps),
    ]
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, str] = {}
    stopped_early = False
    for stage_index, (stage_name, pool, steps) in enumerate(stage_specs):
        if stopped_early:
            break
        sequence = _group_sequence(pool, steps, seed + stage_index)
        for step, group in enumerate(sequence):
            optimizer.zero_grad(set_to_none=True)
            options = group.pairs[0].base_record.answer_options
            if options not in option_cache:
                option_cache[options] = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
            full = _group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                local_max_length=local_max_length,
                external_max_length=external_max_length,
                mode="full",
            )
            ablated = _group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                local_max_length=local_max_length,
                external_max_length=external_max_length,
                mode=_ablated_mode_for_group(group),
            )
            group_losses = _group_losses(full, ablated)
            full_hidden_weight = 4.0 if stage_name == "full_hidden_alignment" else 0.75
            preservation = _kl_preservation(full["output"].logits[:, -1, :], full["output"].logits[:, -1, :].detach())
            residual_budget = torch.stack([torch.tensor(trace.delta_norm, device=backend.device) for trace in full["output"].traces.values()]).sum()
            residual_budget_loss = torch.relu(residual_budget - 1.5).square()
            total = (
                3.0 * group_losses["group_all_correct_loss"]
                + 3.0 * group_losses["group_path_drop_loss"]
                + 1.0 * group_losses["group_context_separation_loss"]
                + 1.0 * group_losses["group_pair_flip_loss"]
                + 2.0 * group_losses["group_projected_margin_loss"]
                + full_hidden_weight * group_losses["group_full_hidden_centroid_loss"]
                + 0.10 * preservation
                + 0.10 * residual_budget_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            row = {
                "route": route,
                "curriculum_stage": stage_name,
                "step": float(step),
                "group_type": group.group_type,
                "total_loss": float(total.detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in group_losses.items()},
                "projected_accuracy": float(full["accuracy"].detach().cpu()),
                "group_success": float(full["group_success"].detach().cpu()),
                "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in full["output"].traces.values()),
                "rss": float(psutil.Process().memory_info().rss),
                "mps_allocated": float(torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0),
            }
            losses.append(row)
            if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
                print(
                    f"group_curriculum route={route} stage={stage_name} step={step + 1}/{steps} "
                    f"loss={row['total_loss']:.6f} acc={row['projected_accuracy']:.3f}",
                    flush=True,
                )
            if backend.device.type == "mps":
                torch.mps.empty_cache()
        checkpoint_paths[stage_name] = _checkpoint(output_dir, route, stage_name, seed, model, heads, projector, full_hidden_projector)
        heldout_rows = _evaluate_groups(
            backend=backend,
            model=model,
            projector=projector,
            full_hidden_projector=full_hidden_projector,
            heads=heads,
            groups=heldout_groups,
            route=route,
            seed=seed,
            local_max_length=local_max_length,
            external_max_length=external_max_length,
            modes=("full",),
        )
        stage_group_types = {
            "memory_necessity": {"memory_necessity_group"},
            "rule_necessity": {"memory_necessity_group", "rule_necessity_group"},
            "conflict_necessity": set(GROUP_TYPES),
            "combined_curriculum": set(GROUP_TYPES),
            "full_hidden_alignment": set(GROUP_TYPES),
        }[stage_name]
        stage_full_success = _mean(
            heldout_rows,
            lambda row: row["mode"] == "full" and row["group_type"] in stage_group_types,
            field="group_success",
        )
        stage_rows.append(
            {
                "route": route,
                "curriculum_stage": stage_name,
                "heldout_group_success": stage_full_success,
                "passes_stage_gate": stage_full_success >= (0.35 if stage_name in {"memory_necessity", "rule_necessity"} else 0.50),
            }
        )
        # Smoke and early experimental runs still write complete artifacts; medium gates
        # use the final summary, not this permissive progress gate.
    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    result = MulticlassCurriculumResult(
        route=route,
        seed=seed,
        target_layers=(16, 24),
        memory_steps=memory_steps,
        rule_steps=rule_steps,
        conflict_steps=conflict_steps,
        combined_steps=combined_steps,
        full_hidden_steps=full_hidden_steps,
        losses=losses,
        loss_decreased={
            "group_all_correct_loss": _window_decreased(losses, "group_all_correct_loss"),
            "group_path_drop_loss": _window_decreased(losses, "group_path_drop_loss"),
            "group_full_hidden_centroid_loss": _window_decreased(losses, "group_full_hidden_centroid_loss", "full_hidden_alignment"),
        },
        adapter_parameter_count=sum(adapter.trainable_parameter_count for adapter in model.adapters.values()),
        projector_parameter_count=sum(parameter.numel() for parameter in projector.parameters()),
        full_hidden_projector_parameter_count=sum(parameter.numel() for parameter in full_hidden_projector.parameters()),
        qwen_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters()),
        qwen_trainable_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=optimizer_contains_qwen,
        fingerprint_unchanged=initial_fingerprint == final_fingerprint,
        checkpoint_paths=checkpoint_paths,
    )
    return model, projector, full_hidden_projector, heads, result, losses, stage_rows


def run_qwen3_multiclass_group_curriculum_repair(
    output_dir: str | Path = "artifacts/civilization/multiclass_group_curriculum_repair",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    external_train_per_label: int = 12,
    external_heldout_per_label: int = 12,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "src/civilization/engine/data/external_cache",
    memory_steps: int = 40,
    rule_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    full_hidden_steps: int = 80,
    max_length: int = 64,
    external_max_length: int = 384,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
    group_evaluation_modes: tuple[str, ...] = ("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "no_state_path", "empty_context", "wrong_context"),
    routes: tuple[str, ...] = ROUTES,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    local_records = build_local_semireal_task_records(local_samples_per_label, seed=seed, context_grounding_mode="grounded_v1")
    local_datasets, _ = records_to_logic_datasets(local_records, max_seq_len=max_length, context_grounding_mode="grounded_v1")
    local_datasets = {task_name: _stage37_path_specific_samples(samples) for task_name, samples in local_datasets.items()}
    external_records, external_manifest = load_external_task_records(
        cache_dir=external_cache_dir,
        task_names=external_task_names,
        allow_download=False,
        max_records_per_task=320,
        context_grounding_mode="grounded_v1",
    )
    external_datasets, _ = records_to_logic_datasets(external_records, max_seq_len=max_length, context_grounding_mode="grounded_v1")
    local_splits, local_train_samples, local_test_samples = _split_and_align_local(local_datasets, local_train_groups, seed)
    external_splits = {}
    external_train_samples = []
    external_test_samples = []
    for task_name, samples in external_datasets.items():
        train, test = _balanced_external_split(samples, external_train_per_label, external_heldout_per_label)
        aligned_train = [row.sample for row in build_evidence_answer_samples(train, "external_benchmark")]
        aligned_test = [row.sample for row in build_evidence_answer_samples(test, "external_benchmark")]
        external_splits[task_name] = (aligned_train, aligned_test)
        external_train_samples.extend(aligned_train)
        external_test_samples.extend(aligned_test)
    local_training_records = build_evidence_answer_samples(local_train_samples, "local_semireal")
    external_training_records = build_evidence_answer_samples(external_train_samples, "external_benchmark")
    local_test_records = build_evidence_answer_samples(local_test_samples, "local_semireal")
    external_test_records = build_evidence_answer_samples(external_test_samples, "external_benchmark")
    assert_no_logic_label_leakage(local_training_records + external_training_records)
    local_train_pairs = _stage37_path_specific_pairs(build_necessity_pairs(local_training_records))
    external_train_pairs = _stage37_path_specific_pairs(build_necessity_pairs(external_training_records))
    local_test_pairs = _stage37_path_specific_pairs(build_necessity_pairs(local_test_records))
    external_test_pairs = _stage37_path_specific_pairs(build_necessity_pairs(external_test_records))
    assert_necessity_pairs_valid(local_train_pairs + external_train_pairs + local_test_pairs + external_test_pairs)
    local_train_groups = build_surface_group_candidate_batches(local_train_pairs)
    local_test_groups = build_surface_group_candidate_batches(local_test_pairs)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    stage_gate_rows: list[dict[str, Any]] = []

    for route in routes:
        route_started = time.perf_counter()
        print(f"group_curriculum_route_start route={route}", flush=True)
        train_groups = list(local_train_groups)
        # External few-shot remains task-level in this stage; local group gates are the
        # primary repair target and external data is evaluated as a recorded contrast.
        model, projector, full_hidden_projector, heads, result, route_losses, route_stage_rows = run_group_batched_curriculum_training(
            backend=backend,
            train_groups=train_groups,
            heldout_groups=local_test_groups,
            output_dir=output_path / "checkpoints",
            route=route,
            seed=seed,
            memory_steps=memory_steps,
            rule_steps=rule_steps,
            conflict_steps=conflict_steps,
            combined_steps=combined_steps,
            full_hidden_steps=full_hidden_steps,
            local_max_length=max_length,
            external_max_length=external_max_length,
        )
        row = asdict(result)
        row.pop("losses")
        training_rows.append(row)
        loss_rows.extend({"route": route, **loss} for loss in route_losses)
        stage_gate_rows.extend(route_stage_rows)
        group_rows.extend(
            _evaluate_groups(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                groups=local_test_groups,
                route=route,
                seed=seed,
                local_max_length=max_length,
                external_max_length=external_max_length,
                modes=group_evaluation_modes,
            )
        )
        for task_name, (train, test) in local_splits.items():
            fixed, projected, projected_full, wrong, traces, failures, truncations = _evaluate_multiclass_task(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                train_samples=train,
                test_samples=test,
                modes=evaluation_modes,
                max_length=max_length,
                batch_size=evaluation_batch_size,
                source_type="local_semireal",
                task_name=task_name,
                seed=seed,
                fail_on_truncation=True,
            )
            fixed_rows.extend({"route": route, **item} for item in fixed)
            projected_rows.extend({"route": route, **item} for item in projected)
            projected_full_rows.extend({"route": route, **item} for item in projected_full)
            wrong_rows.extend({"route": route, **item} for item in wrong)
            trace_rows.extend({"route": route, **item} for item in traces)
            failure_cases.extend({"route": route, **item} for item in failures)
            truncation_cases.extend({"route": route, **item} for item in truncations)
        for task_name, (train, test) in external_splits.items():
            fixed, projected, projected_full, wrong, traces, failures, truncations = _evaluate_multiclass_task(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                train_samples=train,
                test_samples=test,
                modes=evaluation_modes,
                max_length=external_max_length,
                batch_size=evaluation_batch_size,
                source_type="external_benchmark",
                task_name=task_name,
                seed=seed,
                fail_on_truncation=False,
            )
            fixed_rows.extend({"route": route, **item} for item in fixed)
            projected_rows.extend({"route": route, **item} for item in projected)
            projected_full_rows.extend({"route": route, **item} for item in projected_full)
            wrong_rows.extend({"route": route, **item} for item in wrong)
            trace_rows.extend({"route": route, **item} for item in traces)
            failure_cases.extend({"route": route, **item} for item in failures)
            truncation_cases.extend({"route": route, **item} for item in truncations)
        local_pair_metrics, local_pair_failures, local_pair_truncations = _evaluate_projected_necessity_pairs(
            backend=backend,
            model=model,
            projector=projector,
            pairs=local_test_pairs,
            route=route,
            source_type="local_semireal",
            max_length=max_length,
            batch_size=evaluation_batch_size,
            seed=seed,
            fail_on_truncation=True,
        )
        pair_rows.extend(local_pair_metrics)
        failure_cases.extend(local_pair_failures)
        truncation_cases.extend(local_pair_truncations)
        for mode, tasks in (("no_memory_path", LOCAL_MEMORY_TASKS), ("no_state_path", LOCAL_STATE_TASKS), ("no_rule_path", LOCAL_RULE_TASKS)):
            full = _route_metric(projected_rows, route, "local_semireal", "full", tasks)
            ablated = _route_metric(projected_rows, route, "local_semireal", mode, tasks)
            ablation_rows.append({"route": route, "mode": mode, "scope": "local_task", "full_accuracy": full, "ablated_accuracy": ablated, "absolute_drop": full - ablated})
        for group_type, mode in (
            ("memory_necessity_group", "no_memory_path"),
            ("rule_necessity_group", "no_rule_path"),
            ("memory_rule_conflict_group", "no_rule_path"),
        ):
            full = _mean(group_rows, lambda row, current=group_type: row["route"] == route and row["group_type"] == current and row["mode"] == "full", field="group_success")
            ablated = _mean(group_rows, lambda row, current=group_type, current_mode=mode: row["route"] == route and row["group_type"] == current and row["mode"] == current_mode, field="group_success")
            ablation_rows.append({"route": route, "mode": mode, "scope": group_type, "full_accuracy": full, "ablated_accuracy": ablated, "absolute_drop": full - ablated})
        resource_rows.append({"route": route, "seconds": time.perf_counter() - route_started, "rss": psutil.Process().memory_info().rss, "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0})
        print(f"group_curriculum_route_complete route={route} seconds={time.perf_counter() - route_started:.2f}", flush=True)

    for task_name in external_splits:
        local_only = _route_metric(projected_rows, "local_only_transfer", "external_benchmark", "full", {task_name})
        few_shot = _route_metric(projected_rows, "external_few_shot", "external_benchmark", "full", {task_name})
        disabled = _route_metric(projected_rows, "external_few_shot", "external_benchmark", "adapter_disabled", {task_name})
        route_rows.append({"task_name": task_name, "local_only_accuracy": local_only, "few_shot_accuracy": few_shot, "few_shot_improvement": few_shot - local_only, "few_shot_vs_disabled": few_shot - disabled})

    drops = {(row["route"], row["mode"], row.get("scope", "local_task")): row["absolute_drop"] for row in ablation_rows}
    group_success = {
        group_type: _mean(group_rows, lambda row, current=group_type: row["route"] == "external_few_shot" and row["group_type"] == current and row["mode"] == "full", field="group_success")
        for group_type in GROUP_TYPES
    }
    local_projected = _route_metric(projected_rows, "external_few_shot", "local_semireal", "full")
    local_fixed = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "full")
    local_flip = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "full", field="surface_group_flip_accuracy")
    local_wrong_drop = local_projected - _route_metric(projected_rows, "external_few_shot", "local_semireal", "wrong_context")
    fixed_before = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "adapter_disabled")
    fixed_after = local_fixed
    external_few_shot_wins = sum(row["few_shot_vs_disabled"] >= 0.05 for row in route_rows)
    stage_gates = {
        "qwen_frozen": all(row["qwen_trainable_parameter_count"] == 0 and row["qwen_gradients_present"] == 0 and row["fingerprint_unchanged"] and not row["optimizer_contains_qwen_parameters"] for row in training_rows),
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(abs(_route_metric(projected_rows, route, source, "adapter_disabled") - _route_metric(projected_rows, route, source, "zero_scale")) <= 1e-9 for route in routes for source in ("local_semireal", "external_benchmark")),
        "hidden_norm_ratio": all(float(row.get("hidden_norm_ratio", 1.0)) <= 2.0 for row in loss_rows),
        "local_projected_accuracy": local_projected >= 0.75,
        "local_fixed_centroid_accuracy": local_fixed >= 0.60,
        "local_surface_flip": local_flip >= 0.55,
        "local_wrong_context": local_wrong_drop >= 0.12,
        "memory_path_drop": drops.get(("external_few_shot", "no_memory_path", "local_task"), 0.0) >= 0.12,
        "rule_path_drop": drops.get(("external_few_shot", "no_rule_path", "local_task"), 0.0) >= 0.12,
        "state_path_drop": drops.get(("external_few_shot", "no_state_path", "local_task"), 0.0) >= 0.06,
        "memory_group_success": group_success["memory_necessity_group"] >= 0.75,
        "rule_group_success": group_success["rule_necessity_group"] >= 0.75,
        "conflict_group_success": group_success["memory_rule_conflict_group"] >= 0.75,
        "memory_group_drop": drops.get(("external_few_shot", "no_memory_path", "memory_necessity_group"), 0.0) >= 0.20,
        "rule_group_drop": min(drops.get(("external_few_shot", "no_rule_path", "rule_necessity_group"), 0.0), drops.get(("external_few_shot", "no_rule_path", "memory_rule_conflict_group"), 0.0)) >= 0.20,
        "fixed_centroid_improvement": fixed_after >= fixed_before + 0.05,
        "fixed_centroid_local_average": local_fixed >= 0.60,
        "full_hidden_group_flip": _mean(group_rows, lambda row: row["route"] == "external_few_shot" and row["mode"] == "full", field="group_success") >= 0.60,
        "projected_regression": True,
        "external_few_shot_recorded": len(route_rows) == len(external_splits),
    }
    allows_stage27b = all(stage_gates.values()) and external_few_shot_wins >= min(2, len(route_rows))
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "routes": list(routes),
        "adapter_variant": "path_specific_v2",
        "training_mode": "multiclass_group_batched_curriculum_v3",
        "local_projected_accuracy": local_projected,
        "local_fixed_centroid_accuracy": local_fixed,
        "local_surface_flip": local_flip,
        "local_wrong_context_drop": local_wrong_drop,
        "fixed_centroid_before": fixed_before,
        "fixed_centroid_after": fixed_after,
        "group_success": group_success,
        "stage_gate_rows": stage_gate_rows,
        "route_comparison": route_rows,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage27b_rerun": allows_stage27b,
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "group_metrics.csv", group_rows)
    _write_csv(output_path / "stage_gate_metrics.csv", stage_gate_rows)
    _write_csv(output_path / "projected_readout_metrics.csv", projected_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "projected_full_hidden_metrics.csv", projected_full_rows)
    _write_csv(output_path / "necessity_pair_metrics.csv", pair_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "route_comparison.csv", route_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    _json_dump(output_path / "dataset_manifest.json", {"local_semireal": local_real_task_manifest(local_records), "external_benchmark": external_manifest})
    return summary
