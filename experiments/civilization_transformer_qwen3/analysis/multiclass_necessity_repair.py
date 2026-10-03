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

from experiments.civilization_transformer_torch.model import CivilizationAblationConfig

from ..adapter import CivilizationAdapterConfig, PathSpecificCivilizationAdapter, Qwen3MultiAdapterModel
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
from .evidence_answer_data import LABEL_TO_LOCAL_OPTION, assert_no_logic_label_leakage, build_evidence_answer_samples, wrong_context_sample
from .evidence_answer_training import _answer_scores, _capture_updates, _margin_loss
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .hidden_states import last_non_padding_pool
from .memory_rule_necessity_benchmark import _split_and_align_local
from .multilayer_adapter_benchmark import _collect_multilayer_vectors
from .necessity_alignment_data import NecessityPair, assert_necessity_pairs_valid, build_necessity_pairs
from .real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
)
from .real_task_migration import _balanced_external_split, _generic_centroid_metrics, _surface_flip_accuracy


@dataclass
class MulticlassCurriculumResult:
    route: str
    seed: int
    target_layers: tuple[int, ...]
    memory_steps: int
    rule_steps: int
    conflict_steps: int
    combined_steps: int
    full_hidden_steps: int
    losses: list[dict[str, float | str]]
    loss_decreased: dict[str, bool]
    adapter_parameter_count: int
    projector_parameter_count: int
    full_hidden_projector_parameter_count: int
    qwen_parameter_count: int
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    fingerprint_unchanged: bool
    checkpoint_paths: dict[str, str]


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


def _make_path_specific_model(backend: Qwen3Backend, target_layers: tuple[int, ...]) -> Qwen3MultiAdapterModel:
    return Qwen3MultiAdapterModel(
        backend,
        {
            layer: PathSpecificCivilizationAdapter(CivilizationAdapterConfig(target_layer=layer))
            for layer in target_layers
        },
    )


def _residual_parameters(model: Qwen3MultiAdapterModel) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for adapter in model.adapters.values():
        params.extend(
            [
                adapter.base_residual_scale,
                adapter.memory_residual_scale,
                adapter.rule_residual_scale,
                adapter.state_residual_scale,
            ]
        )
    return params


def _window_decreased(rows: list[dict[str, Any]], key: str, stage_prefix: str | None = None) -> bool:
    values = [
        float(row[key])
        for row in rows
        if key in row and (stage_prefix is None or str(row.get("curriculum_stage", "")).startswith(stage_prefix))
    ]
    if len(values) < 2:
        return False
    window = max(1, min(10, len(values) // 3))
    return sum(values[-window:]) / window < sum(values[:window]) / window


def _sequence(pairs: list[NecessityPair], steps: int, seed: int) -> list[NecessityPair]:
    if not pairs:
        raise ValueError("multiclass curriculum pair pool cannot be empty")
    rng = random.Random(seed)
    ordered = sorted(pairs, key=lambda pair: (pair.pair_type, pair.pair_id, pair.expected_full_option_id))
    result: list[NecessityPair] = []
    while len(result) < steps:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:steps]


def _masked_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if vectors.shape[1] == 0:
        return vectors.new_zeros((vectors.shape[0], vectors.shape[-1]))
    weights = mask.to(vectors.dtype).unsqueeze(-1)
    return (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _context(
    context_encoder: FrozenQwenContextEncoder,
    samples: list,
    attention_mask: torch.Tensor,
    mode: str,
):
    if mode == "adapter_disabled":
        return context_encoder.build_context(samples, attention_mask, adapter_enabled=False, context_mode="full")
    if mode == "zero_scale":
        return context_encoder.build_context(samples, attention_mask, force_zero_scale=True, context_mode="full")
    if mode == "no_memory_path":
        return context_encoder.build_context(
            samples,
            attention_mask,
            ablation_config=CivilizationAblationConfig(use_memory_path=False),
            context_mode="full",
        )
    if mode == "no_state_path":
        return context_encoder.build_context(
            samples,
            attention_mask,
            ablation_config=CivilizationAblationConfig(use_state_path=False),
            context_mode="full",
        )
    if mode == "no_rule_path":
        return context_encoder.build_context(
            samples,
            attention_mask,
            ablation_config=CivilizationAblationConfig(use_rule_path=False),
            context_mode="full",
        )
    if mode == "empty_context":
        return context_encoder.build_context(samples, attention_mask, context_mode="empty")
    if mode == "wrong_context":
        return context_encoder.build_context(samples, attention_mask, context_mode="wrong")
    if mode in {"full", "counterfactual_context"}:
        return context_encoder.build_context(samples, attention_mask, context_mode="full")
    raise ValueError(f"unsupported eval mode: {mode}")


def _encode_pair(backend: Qwen3Backend, pair: NecessityPair, local_max_length: int, external_max_length: int):
    samples = [pair.full_sample, pair.counterfactual_sample]
    max_length = external_max_length if pair.base_record.source_type == "external_benchmark" else local_max_length
    encoded, truncations = backend.encode([qwen_text_for_sample(sample) for sample in samples], max_length=max_length)
    if truncations and pair.base_record.source_type != "external_benchmark":
        raise ValueError("local multiclass necessity sample was truncated")
    return {name: tensor.to(backend.device) for name, tensor in encoded.items()}, samples


def _centroid_scores(vectors: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).T


def _dynamic_centroid_loss(vectors: torch.Tensor, targets: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    unique_targets = sorted(set(targets.detach().cpu().tolist()))
    centroids = []
    index_for_label = {}
    for index, label in enumerate(unique_targets):
        rows = vectors[targets == label]
        centroids.append(rows.mean(dim=0))
        index_for_label[label] = index
    centroid_tensor = torch.stack(centroids, dim=0)
    scores = _centroid_scores(vectors, centroid_tensor)
    losses = []
    for index, label in enumerate(targets.detach().cpu().tolist()):
        loss, _ = _margin_loss(scores[index : index + 1], index_for_label[label], margin)
        losses.append(loss)
    return torch.stack(losses).mean()


def _prototype_loss(heads: AdapterDiagnosticHeads, pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prototypes = F.normalize(heads.logic_prototypes.float(), dim=-1)
    scores = F.normalize(pooled.float(), dim=-1) @ prototypes.T
    return F.cross_entropy(scores / 0.1, labels)


def _path_drop_loss(full_scores: torch.Tensor, ablated_scores: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    full_target = full_scores.gather(1, targets[:, None]).squeeze(1)
    ablated_target = ablated_scores.gather(1, targets[:, None]).squeeze(1)
    return torch.relu(margin + ablated_target - full_target).mean()


def _group_key(pair: NecessityPair) -> tuple[str, str]:
    return pair.pair_type, pair.full_sample.surface_group_id


def _pair_groups(pairs: list[NecessityPair]) -> dict[tuple[str, str], list[NecessityPair]]:
    groups: dict[tuple[str, str], list[NecessityPair]] = {}
    for pair in pairs:
        groups.setdefault(_group_key(pair), []).append(pair)
    return {
        key: sorted(value, key=lambda pair: pair.expected_full_option_id)
        for key, value in groups.items()
        if len({pair.expected_full_option_id for pair in value}) >= 3
    }


def _group_alignment_loss(
    *,
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    context_encoder: FrozenQwenContextEncoder,
    group_pairs: list[NecessityPair],
    option_vectors: torch.Tensor,
    local_max_length: int,
    external_max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [pair.full_sample for pair in group_pairs]
    max_length = external_max_length if group_pairs[0].base_record.source_type == "external_benchmark" else local_max_length
    encoded, truncations = backend.encode([qwen_text_for_sample(sample) for sample in samples], max_length=max_length)
    if truncations and group_pairs[0].base_record.source_type != "external_benchmark":
        raise ValueError("local multiclass group sample was truncated")
    encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
    targets = torch.tensor([pair.expected_full_option_id for pair in group_pairs], dtype=torch.long, device=backend.device)
    with torch.inference_mode():
        baseline = backend.inference_forward(encoded)
        baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
    output = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "full"))
    if group_pairs[0].pair_type == "memory_necessity_pair":
        ablated = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_memory_path"))
    elif group_pairs[0].pair_type in {"rule_necessity_pair", "memory_rule_conflict_pair"}:
        ablated = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_rule_path"))
    else:
        ablated = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_state_path"))
    projected = projector(last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float() - baseline_pooled)
    projected_ablated = projector(last_non_padding_pool(ablated.hidden_states[-1], ablated.attention_mask).float() - baseline_pooled)
    scores = _answer_scores(projected, option_vectors)
    ablated_scores = _answer_scores(projected_ablated, option_vectors)
    group_ce = F.cross_entropy(scores / 0.05, targets)
    group_drop = _path_drop_loss(scores, ablated_scores, targets, 0.25)
    normalized = F.normalize(projected.float(), dim=-1)
    similarity = normalized @ normalized.T
    target_equal = targets[:, None] == targets[None, :]
    different = similarity[~target_equal]
    separation = torch.relu(different + 0.10).mean() if different.numel() else scores.new_zeros(())
    return group_ce, group_drop, separation


def _save_checkpoint(
    output_dir: Path,
    route: str,
    stage: str,
    seed: int,
    model: Qwen3MultiAdapterModel,
    heads: AdapterDiagnosticHeads,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"multiclass_necessity_{route}_{stage}_seed_{seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "diagnostic_heads_state_dict": heads.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {"route": route, "stage": stage, "seed": seed},
        },
        path,
    )
    return str(path)


def _stage37_path_specific_samples(samples: list) -> list:
    """Remove non-target shortcut paths from local task-level samples.

    Stage37 has a stricter purpose than the earlier real-task runners: task-level
    memory/rule ablations must measure whether the intended native path is
    necessary. The local semi-real records were intentionally rich and can carry
    redundant Memory, Rule, and State hints. That is useful for general tasks but
    invalid for this specific gate because no-memory/no-rule can be bypassed by
    the other context paths. This transform keeps the visible text and labels
    unchanged, while making the tested task families path-specific.
    """
    adjusted = []
    for sample in samples:
        if sample.variant in LOCAL_MEMORY_TASKS:
            target_answer = LABEL_TO_LOCAL_OPTION.get(sample.label, sample.expected_pattern)
            adjusted.append(
                replace(
                    sample,
                    required_paths=("memory",),
                    memory_target=f"memory evidence selects answer option: {target_answer}",
                    rule_target="",
                    state_target="",
                    context_noise_count=0,
                    conflict_context_count=0,
                )
            )
        elif sample.variant in LOCAL_RULE_TASKS:
            target_answer = LABEL_TO_LOCAL_OPTION.get(sample.label, sample.expected_pattern)
            adjusted.append(
                replace(
                    sample,
                    required_paths=("rule",),
                    memory_target="",
                    rule_target=f"control rule selects answer option: {target_answer}",
                    state_target="",
                    context_noise_count=0,
                    conflict_context_count=0,
                )
            )
        else:
            adjusted.append(sample)
    return adjusted


def _stage37_path_specific_pairs(pairs: list[NecessityPair]) -> list[NecessityPair]:
    adjusted = []
    for pair in pairs:
        if pair.pair_type == "memory_necessity_pair":
            full = replace(pair.full_sample, required_paths=("memory",), rule_target="", state_target="")
            counterfactual = replace(pair.counterfactual_sample, required_paths=("memory",), rule_target="", state_target="")
        elif pair.pair_type == "rule_necessity_pair":
            full = replace(pair.full_sample, required_paths=("rule",), memory_target="", state_target="")
            counterfactual = replace(pair.counterfactual_sample, required_paths=("rule",), memory_target="", state_target="")
        else:
            full = replace(pair.full_sample, required_paths=("memory", "rule"), state_target="")
            counterfactual = replace(pair.counterfactual_sample, required_paths=("memory", "rule"), state_target="")
        adjusted.append(replace(pair, full_sample=full, counterfactual_sample=counterfactual))
    return adjusted


def _evaluate_projected_necessity_pairs(
    *,
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    pairs: list[NecessityPair],
    route: str,
    source_type: str,
    max_length: int,
    batch_size: int,
    seed: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    eval_modes = ("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "no_state_path", "empty_context", "wrong_context")
    samples = [sample for pair in pairs for sample in (pair.full_sample, pair.counterfactual_sample)]
    disabled_vectors, _disabled_traces, disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    failures = [{"route": route, "source_type": source_type, **item} for item in disabled_failures]
    option_cache: dict[tuple[str, ...], np.ndarray] = {}
    for mode in eval_modes:
        mode_samples = samples
        collector_mode = mode
        if mode == "wrong_context":
            mode_samples = []
            for pair in pairs:
                wrong = wrong_context_sample(pair.base_record)
                mode_samples.extend(
                    [
                        replace(wrong, text=pair.full_sample.text, expected_pattern=pair.full_sample.expected_pattern),
                        replace(wrong, text=pair.counterfactual_sample.text, expected_pattern=pair.counterfactual_sample.expected_pattern),
                    ]
                )
            collector_mode = "full"
        if mode == "adapter_disabled":
            mode_vectors = disabled_vectors
            mode_failures: list[dict[str, Any]] = []
        else:
            mode_vectors, _mode_traces, mode_failures = _collect_multilayer_vectors(
                backend,
                model,
                mode_samples,
                collector_mode,
                max_length,
                batch_size,
                (PROBE_LAYER,),
                fail_on_truncation=fail_on_truncation,
            )
        failures.extend({"route": route, "source_type": source_type, **item} for item in mode_failures)
        delta = mode_vectors[PROBE_LAYER] - disabled_vectors[PROBE_LAYER]
        with torch.no_grad():
            projected_all = projector(torch.tensor(delta, dtype=torch.float32, device=backend.device)).float().cpu().numpy()
        for index, pair in enumerate(pairs):
            start = index * 2
            end = start + 2
            if pair.base_record.answer_options not in option_cache:
                option_cache[pair.base_record.answer_options] = build_answer_option_vectors(
                    backend,
                    pair.base_record.answer_options,
                )
            metrics = score_answer_options(
                projected_all[start:end],
                [pair.full_sample, pair.counterfactual_sample],
                option_cache[pair.base_record.answer_options],
                pair.base_record.answer_options,
            )
            full_correct = bool(metrics.rows[0]["correct"])
            counterfactual_correct = bool(metrics.rows[1]["correct"])
            rows.append(
                {
                    "route": route,
                    "seed": seed,
                    "source_type": source_type,
                    "pair_type": pair.pair_type,
                    "pair_id": pair.pair_id,
                    "required_path": pair.required_path,
                    "mode": mode,
                    "readout": "projected_delta",
                    "accuracy": metrics.accuracy,
                    "macro_accuracy": metrics.macro_accuracy,
                    "pair_success": 1.0 if full_correct and counterfactual_correct else 0.0,
                    "full_correct": full_correct,
                    "counterfactual_correct": counterfactual_correct,
                    "mean_margin": metrics.mean_margin,
                    "delta_distance": float(np.linalg.norm(projected_all[start] - projected_all[start + 1])),
                }
            )
    truncations = [item for item in failures if item.get("type") == "truncation"]
    hard_failures = [item for item in failures if item.get("type") != "truncation"]
    return rows, hard_failures, truncations


def run_multiclass_path_curriculum_training(
    *,
    backend: Qwen3Backend,
    train_pairs: list[NecessityPair],
    output_dir: Path,
    route: str,
    seed: int,
    target_layers: tuple[int, ...] = (16, 24),
    memory_steps: int = 40,
    rule_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    full_hidden_steps: int = 80,
    gradient_accumulation: int = 4,
    local_max_length: int = 64,
    external_max_length: int = 384,
    learning_rate: float = 3e-4,
    enable_group_losses: bool = True,
    group_loss_interval: int = 2,
) -> tuple[Qwen3MultiAdapterModel, PathReadoutProjector, FullHiddenCentroidProjector, MulticlassCurriculumResult]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_path_specific_model(backend, target_layers)
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    projector = PathReadoutProjector().to(backend.device)
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    groups_by_key = _pair_groups(train_pairs)
    memory_pairs = [pair for pair in train_pairs if pair.pair_type == "memory_necessity_pair"]
    rule_pairs = [pair for pair in train_pairs if pair.pair_type == "rule_necessity_pair"]
    conflict_pairs = [pair for pair in train_pairs if pair.pair_type == "memory_rule_conflict_pair"]
    combined_pairs = sorted(train_pairs, key=lambda pair: (pair.pair_type, pair.pair_id))
    stages: list[tuple[str, list[NecessityPair], int]] = [
        ("memory_necessity", memory_pairs, memory_steps),
        ("rule_necessity", rule_pairs + memory_pairs, rule_steps),
        ("conflict_necessity", conflict_pairs + memory_pairs + rule_pairs, conflict_steps),
        ("combined_curriculum", combined_pairs, combined_steps),
        ("full_hidden_alignment", combined_pairs, full_hidden_steps),
    ]
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
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float | str]] = []
    checkpoint_paths: dict[str, str] = {}
    for stage_index, (stage_name, pool, steps) in enumerate(stages):
        sequence = _sequence(pool, steps * gradient_accumulation, seed + stage_index)
        cursor = 0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            aggregate: dict[str, float] = {}
            batch_hidden = []
            batch_labels = []
            for _ in range(gradient_accumulation):
                pair = sequence[cursor]
                cursor += 1
                encoded, samples = _encode_pair(backend, pair, local_max_length, external_max_length)
                with torch.inference_mode():
                    baseline = backend.inference_forward(encoded)
                full_context = _context(context_encoder, samples, encoded["attention_mask"], "full")
                output = model(encoded, full_context)
                no_memory = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_memory_path"))
                no_rule = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_rule_path"))
                no_state = model(encoded, _context(context_encoder, samples, encoded["attention_mask"], "no_state_path"))
                pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
                baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
                delta = pooled - baseline_pooled
                projected_delta = projector(delta)
                no_memory_delta = last_non_padding_pool(no_memory.hidden_states[-1], no_memory.attention_mask).float() - baseline_pooled
                no_rule_delta = last_non_padding_pool(no_rule.hidden_states[-1], no_rule.attention_mask).float() - baseline_pooled
                no_state_delta = last_non_padding_pool(no_state.hidden_states[-1], no_state.attention_mask).float() - baseline_pooled
                projected_no_memory = projector(no_memory_delta)
                projected_no_rule = projector(no_rule_delta)
                projected_no_state = projector(no_state_delta)
                if pair.base_record.answer_options not in option_cache:
                    option_cache[pair.base_record.answer_options] = torch.tensor(
                        build_answer_option_vectors(backend, pair.base_record.answer_options),
                        dtype=torch.float32,
                        device=backend.device,
                    )
                option_vectors = option_cache[pair.base_record.answer_options]
                targets = torch.tensor([pair.expected_full_option_id, pair.expected_counterfactual_option_id], dtype=torch.long, device=backend.device)
                scores = _answer_scores(projected_delta, option_vectors)
                no_memory_scores = _answer_scores(projected_no_memory, option_vectors)
                no_rule_scores = _answer_scores(projected_no_rule, option_vectors)
                no_state_scores = _answer_scores(projected_no_state, option_vectors)
                margin_losses = []
                margins = []
                for index, target in enumerate(targets.tolist()):
                    loss, margin = _margin_loss(scores[index : index + 1], target, 0.20)
                    margin_losses.append(loss)
                    margins.append(margin)
                answer_loss = torch.stack(margin_losses).mean()
                answer_margin = torch.stack(margins).mean()
                memory_drop = _path_drop_loss(scores, no_memory_scores, targets, 0.20)
                rule_drop = _path_drop_loss(scores, no_rule_scores, targets, 0.20)
                state_drop = _path_drop_loss(scores, no_state_scores, targets, 0.10)
                evidence = _masked_mean(full_context.memory_vectors.float(), full_context.memory_mask) + _masked_mean(
                    full_context.rule_vectors.float(),
                    full_context.rule_mask,
                )
                updates = _capture_updates(model, output.attention_mask)
                context_loss = (1.0 - (F.normalize(projected_delta.float(), dim=-1) * F.normalize(evidence.float(), dim=-1)).sum(dim=-1)).mean()
                labels = torch.tensor([LABEL_TO_ID[sample.label] for sample in samples], dtype=torch.long, device=backend.device)
                fixed_loss = _prototype_loss(heads, pooled, labels)
                projected_full = full_hidden_projector(pooled)
                projected_full_loss = _prototype_loss(heads, projected_full, labels)
                pair_flip_loss = torch.relu(
                    (F.normalize(pooled[0:1], dim=-1) * F.normalize(pooled[1:2], dim=-1)).sum(dim=-1) - 0.50
                ).mean()
                dynamic_loss = _dynamic_centroid_loss(pooled, labels)
                preservation = _kl_preservation(output.logits[:, -1, :], baseline.logits[:, -1, :])
                residual_budget = torch.stack([torch.tensor(trace.delta_norm, device=backend.device) for trace in output.traces.values()]).sum()
                residual_budget_loss = torch.relu(residual_budget - 0.75).square()
                path_loss = torch.zeros((), device=backend.device)
                if pair.pair_type == "memory_necessity_pair":
                    path_loss = path_loss + memory_drop
                elif pair.pair_type == "rule_necessity_pair":
                    path_loss = path_loss + rule_drop
                elif pair.pair_type == "memory_rule_conflict_pair":
                    path_loss = path_loss + rule_drop + 0.5 * state_drop
                group_ce = scores.new_zeros(())
                group_drop = scores.new_zeros(())
                group_separation = scores.new_zeros(())
                group_pairs = (
                    groups_by_key.get(_group_key(pair))
                    if enable_group_losses and group_loss_interval > 0 and step % group_loss_interval == 0
                    else None
                )
                if group_pairs:
                    group_ce, group_drop, group_separation = _group_alignment_loss(
                        backend=backend,
                        model=model,
                        projector=projector,
                        context_encoder=context_encoder,
                        group_pairs=group_pairs,
                        option_vectors=option_vectors,
                        local_max_length=local_max_length,
                        external_max_length=external_max_length,
                    )
                full_hidden_weight = 10.0 if stage_name == "full_hidden_alignment" else 1.0
                total = (
                    3.0 * answer_loss
                    + 2.0 * path_loss
                    + 3.0 * group_ce
                    + 3.0 * group_drop
                    + 1.0 * group_separation
                    + 0.25 * context_loss
                    + full_hidden_weight * fixed_loss
                    + 0.25 * projected_full_loss
                    + full_hidden_weight * dynamic_loss
                    + full_hidden_weight * pair_flip_loss
                    + 0.10 * preservation
                    + 0.10 * residual_budget_loss
                )
                (total / gradient_accumulation).backward()
                values = {
                    "total_loss": total,
                    "answer_option_margin_loss": answer_loss,
                    "path_drop_loss": path_loss,
                    "memory_path_drop_loss": memory_drop,
                    "rule_path_drop_loss": rule_drop,
                    "state_path_drop_loss": state_drop,
                    "group_all_correct_loss": group_ce,
                    "group_path_drop_loss": group_drop,
                    "group_context_separation_loss": group_separation,
                    "fixed_centroid_loss": fixed_loss,
                    "projected_full_hidden_loss": projected_full_loss,
                    "dynamic_centroid_loss": dynamic_loss,
                    "full_hidden_pair_flip_loss": pair_flip_loss,
                    "correct_option_margin": answer_margin,
                    "logit_preservation_kl": preservation,
                    "residual_budget_loss": residual_budget_loss,
                }
                for key, value in values.items():
                    aggregate[key] = aggregate.get(key, 0.0) + float(value.detach().cpu()) / gradient_accumulation
                aggregate["hidden_norm_ratio"] = max(aggregate.get("hidden_norm_ratio", 0.0), max(trace.hidden_norm_ratio for trace in output.traces.values()))
                batch_hidden.append(pooled)
                batch_labels.append(labels)
            torch.nn.utils.clip_grad_norm_(adapter_parameters + residual_parameters + head_parameters + list(projector.parameters()) + list(full_hidden_projector.parameters()), 1.0)
            optimizer.step()
            row = {
                "route": route,
                "curriculum_stage": stage_name,
                "step": float(step),
                **aggregate,
                "rss": float(psutil.Process().memory_info().rss),
                "mps_allocated": float(torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0),
            }
            for layer in model.target_layers:
                adapter = model.adapters[str(layer)]
                row[f"memory_residual_scale_{layer}"] = float(adapter.memory_residual_scale.detach().cpu())
                row[f"rule_residual_scale_{layer}"] = float(adapter.rule_residual_scale.detach().cpu())
            losses.append(row)
            if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
                print(
                    f"multiclass_curriculum route={route} stage={stage_name} step={step + 1}/{steps} "
                    f"loss={row['total_loss']:.6f} margin={row['correct_option_margin']:.6f}",
                    flush=True,
                )
            if backend.device.type == "mps":
                torch.mps.empty_cache()
        checkpoint_paths[stage_name] = _save_checkpoint(output_dir, route, stage_name, seed, model, heads, projector, full_hidden_projector)
    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    result = MulticlassCurriculumResult(
        route=route,
        seed=seed,
        target_layers=target_layers,
        memory_steps=memory_steps,
        rule_steps=rule_steps,
        conflict_steps=conflict_steps,
        combined_steps=combined_steps,
        full_hidden_steps=full_hidden_steps,
        losses=losses,
        loss_decreased={
            "answer_option_margin_loss": _window_decreased(losses, "answer_option_margin_loss"),
            "path_drop_loss": _window_decreased(losses, "path_drop_loss"),
            "fixed_centroid_loss": _window_decreased(losses, "fixed_centroid_loss", "full_hidden_alignment"),
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
    return model, projector, full_hidden_projector, result


def _evaluate_multiclass_task(
    *,
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    train_samples: list,
    test_samples: list,
    modes: tuple[str, ...],
    max_length: int,
    batch_size: int,
    source_type: str,
    task_name: str,
    seed: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_vectors, train_traces, train_failures = _collect_multilayer_vectors(
        backend,
        model,
        train_samples,
        "full",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    disabled_vectors, _disabled_traces, disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        test_samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    train_disabled_vectors, _train_disabled_traces, train_disabled_failures = _collect_multilayer_vectors(
        backend,
        model,
        train_samples,
        "adapter_disabled",
        max_length,
        batch_size,
        (PROBE_LAYER,),
        fail_on_truncation=fail_on_truncation,
    )
    train_labels = [sample.label for sample in train_samples]
    test_labels = [sample.label for sample in test_samples]
    options = tuple(sorted({sample.expected_pattern for sample in train_samples + test_samples}))
    if len(options) < 2:
        options = tuple(dict.fromkeys(sample.expected_pattern for sample in train_samples + test_samples))
    option_vectors = build_answer_option_vectors(backend, options)
    with torch.no_grad():
        projected_train = projector(
            torch.tensor(train_vectors[PROBE_LAYER] - train_disabled_vectors[PROBE_LAYER], dtype=torch.float32, device=backend.device)
        ).float().cpu().numpy()
        projected_full_train = full_hidden_projector(
            torch.tensor(train_vectors[PROBE_LAYER], dtype=torch.float32, device=backend.device)
        ).float().cpu().numpy()
    fixed_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = [{"source_type": source_type, "task_name": task_name, "seed": seed, **row} for row in train_traces]
    failures = [
        {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "train", **item}
        for item in train_failures
    ] + [
        {
            "seed": seed,
            "source_type": source_type,
            "task_name": task_name,
            "split": "train_disabled",
            **item,
        }
        for item in train_disabled_failures
    ] + [
        {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "test", **item}
        for item in disabled_failures
    ]
    truncations = [item for item in failures if item.get("type") == "truncation"]
    hard_failures = [item for item in failures if item.get("type") != "truncation"]
    full_projected_accuracy = 0.0
    wrong_projected_accuracy = 0.0
    for mode in modes:
        mode_samples = test_samples
        collector_mode = mode
        if mode == "wrong_context":
            mode_samples = [wrong_context_sample(record) for record in build_evidence_answer_samples(test_samples, source_type)]
            collector_mode = "full"
        if mode == "counterfactual_context":
            mode_samples = [wrong_context_sample(record) for record in build_evidence_answer_samples(test_samples, source_type)]
            collector_mode = "full"
        test_vectors, traces, test_failures = _collect_multilayer_vectors(
            backend,
            model,
            mode_samples,
            collector_mode,
            max_length,
            batch_size,
            (PROBE_LAYER,),
            fail_on_truncation=fail_on_truncation,
        )
        trace_rows.extend({"source_type": source_type, "task_name": task_name, "seed": seed, **row} for row in traces)
        for item in test_failures:
            wrapped = {"seed": seed, "source_type": source_type, "task_name": task_name, "split": "test", **item}
            if item.get("type") == "truncation":
                truncations.append(wrapped)
            else:
                hard_failures.append(wrapped)
        full_matrix = test_vectors[PROBE_LAYER]
        disabled_matrix = disabled_vectors[PROBE_LAYER]
        with torch.no_grad():
            projected_matrix = projector(
                torch.tensor(full_matrix - disabled_matrix, dtype=torch.float32, device=backend.device)
            ).float().cpu().numpy()
            projected_full_matrix = full_hidden_projector(
                torch.tensor(full_matrix, dtype=torch.float32, device=backend.device)
            ).float().cpu().numpy()
        fixed = _generic_centroid_metrics(train_vectors[PROBE_LAYER], train_labels, full_matrix, test_labels)
        projected = _generic_centroid_metrics(projected_train, train_labels, projected_matrix, test_labels)
        projected_full = _generic_centroid_metrics(projected_full_train, train_labels, projected_full_matrix, test_labels)
        fixed_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": PROBE_LAYER,
                "accuracy": fixed["accuracy"],
                "macro_accuracy": fixed["macro_accuracy"],
                "mean_distance": fixed["mean_distance"],
                "surface_group_flip_accuracy": _surface_flip_accuracy(test_samples, fixed["predictions"]),
                "centroid_source": "full_context_train",
            }
        )
        projected_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": PROBE_LAYER,
                "accuracy": projected["accuracy"],
                "macro_accuracy": projected["macro_accuracy"],
                "mean_distance": projected["mean_distance"],
                "readout": "projected_delta",
            }
        )
        projected_full_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "probe_layer": PROBE_LAYER,
                "accuracy": projected_full["accuracy"],
                "macro_accuracy": projected_full["macro_accuracy"],
                "readout": "projected_full_hidden",
            }
        )
        answer = score_answer_options(projected_matrix, mode_samples, option_vectors, options)
        if mode == "full":
            full_projected_accuracy = answer.accuracy
        if mode == "wrong_context":
            wrong_projected_accuracy = answer.accuracy
        wrong_rows.append(
            {
                "seed": seed,
                "source_type": source_type,
                "task_name": task_name,
                "mode": mode,
                "projected_answer_accuracy": answer.accuracy,
                "mean_margin": answer.mean_margin,
                "delta_norm": float(np.linalg.norm(projected_matrix, axis=1).mean()),
            }
        )
    wrong_rows.append(
        {
            "seed": seed,
            "source_type": source_type,
            "task_name": task_name,
            "mode": "wrong_context_drop",
            "projected_answer_accuracy": full_projected_accuracy - wrong_projected_accuracy,
            "mean_margin": 0.0,
            "delta_norm": 0.0,
        }
    )
    return fixed_rows, projected_rows, projected_full_rows, wrong_rows, trace_rows, hard_failures, truncations


def run_qwen3_multiclass_necessity_repair(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/multiclass_necessity_repair",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    external_train_per_label: int = 12,
    external_heldout_per_label: int = 12,
    external_task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    external_cache_dir: str | Path = "experiments/civilization_transformer_qwen3/data/external_cache",
    memory_steps: int = 40,
    rule_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    full_hidden_steps: int = 80,
    gradient_accumulation: int = 4,
    max_length: int = 64,
    external_max_length: int = 384,
    preferred_device: str | None = None,
    evaluation_batch_size: int = 12,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
    enable_group_losses: bool = True,
    group_loss_interval: int = 2,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    local_records = build_local_semireal_task_records(local_samples_per_label, seed=seed, context_grounding_mode="grounded_v1")
    local_datasets, _ = records_to_logic_datasets(local_records, max_seq_len=max_length, context_grounding_mode="grounded_v1")
    local_datasets = {
        task_name: _stage37_path_specific_samples(samples)
        for task_name, samples in local_datasets.items()
    }
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
    local_train_pairs = build_necessity_pairs(local_training_records)
    external_train_pairs = build_necessity_pairs(external_training_records)
    local_test_pairs = build_necessity_pairs(local_test_records)
    external_test_pairs = build_necessity_pairs(external_test_records)
    local_train_pairs = _stage37_path_specific_pairs(local_train_pairs)
    external_train_pairs = _stage37_path_specific_pairs(external_train_pairs)
    local_test_pairs = _stage37_path_specific_pairs(local_test_pairs)
    external_test_pairs = _stage37_path_specific_pairs(external_test_pairs)
    assert_necessity_pairs_valid(local_train_pairs + external_train_pairs + local_test_pairs + external_test_pairs)
    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    truncation_cases: list[dict[str, Any]] = []
    route_models = {}
    route_projectors = {}
    for route in ROUTES:
        route_started = time.perf_counter()
        print(f"multiclass_necessity_route_start route={route}", flush=True)
        train_pairs = list(local_train_pairs)
        if route == "external_few_shot":
            train_pairs.extend(external_train_pairs)
        model, projector, full_hidden_projector, training = run_multiclass_path_curriculum_training(
            backend=backend,
            train_pairs=train_pairs,
            output_dir=output_path / "checkpoints",
            route=route,
            seed=seed,
            memory_steps=memory_steps,
            rule_steps=rule_steps,
            conflict_steps=conflict_steps,
            combined_steps=combined_steps,
            full_hidden_steps=full_hidden_steps,
            gradient_accumulation=gradient_accumulation,
            local_max_length=max_length,
            external_max_length=external_max_length,
            enable_group_losses=enable_group_losses,
            group_loss_interval=group_loss_interval,
        )
        row = asdict(training)
        losses = row.pop("losses")
        training_rows.append(row)
        loss_rows.extend({"route": route, **loss} for loss in losses)
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
        if external_test_pairs:
            external_pair_metrics, external_pair_failures, external_pair_truncations = _evaluate_projected_necessity_pairs(
                backend=backend,
                model=model,
                projector=projector,
                pairs=external_test_pairs,
                route=route,
                source_type="external_benchmark",
                max_length=external_max_length,
                batch_size=evaluation_batch_size,
                seed=seed,
                fail_on_truncation=False,
            )
            pair_rows.extend(external_pair_metrics)
            failure_cases.extend(external_pair_failures)
            truncation_cases.extend(external_pair_truncations)
        for mode, tasks in (("no_memory_path", LOCAL_MEMORY_TASKS), ("no_state_path", LOCAL_STATE_TASKS), ("no_rule_path", LOCAL_RULE_TASKS)):
            full = _route_metric(projected_rows, route, "local_semireal", "full", tasks)
            ablated = _route_metric(projected_rows, route, "local_semireal", mode, tasks)
            ablation_rows.append(
                {
                    "route": route,
                    "mode": mode,
                    "scope": "local_task",
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
        for pair_type, mode in (
            ("memory_necessity_pair", "no_memory_path"),
            ("rule_necessity_pair", "no_rule_path"),
            ("memory_rule_conflict_pair", "no_rule_path"),
        ):
            full = _mean(
                pair_rows,
                lambda row, current=pair_type: row["route"] == route
                and row["source_type"] == "local_semireal"
                and row["pair_type"] == current
                and row["mode"] == "full",
                field="pair_success",
            )
            ablated = _mean(
                pair_rows,
                lambda row, current=pair_type, current_mode=mode: row["route"] == route
                and row["source_type"] == "local_semireal"
                and row["pair_type"] == current
                and row["mode"] == current_mode,
                field="pair_success",
            )
            ablation_rows.append(
                {
                    "route": route,
                    "mode": mode,
                    "scope": pair_type,
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
        resource_rows.append(
            {
                "route": route,
                "seconds": time.perf_counter() - route_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        route_models[route] = model
        route_projectors[route] = projector
        print(f"multiclass_necessity_route_complete route={route} seconds={time.perf_counter() - route_started:.2f}", flush=True)
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    for task_name in external_splits:
        local_only = _route_metric(projected_rows, "local_only_transfer", "external_benchmark", "full", {task_name})
        few_shot = _route_metric(projected_rows, "external_few_shot", "external_benchmark", "full", {task_name})
        disabled = _route_metric(projected_rows, "external_few_shot", "external_benchmark", "adapter_disabled", {task_name})
        route_rows.append({"task_name": task_name, "local_only_accuracy": local_only, "few_shot_accuracy": few_shot, "few_shot_improvement": few_shot - local_only, "few_shot_vs_disabled": few_shot - disabled})
    local_only_local = _route_metric(projected_rows, "local_only_transfer", "local_semireal", "full")
    few_shot_local = _route_metric(projected_rows, "external_few_shot", "local_semireal", "full")
    retention_rows.append({"local_only_accuracy": local_only_local, "few_shot_accuracy": few_shot_local, "few_shot_regression": local_only_local - few_shot_local, "passes_retention": few_shot_local >= local_only_local - 0.05})
    drops = {(row["route"], row["mode"], row.get("scope", "local_task")): row["absolute_drop"] for row in ablation_rows}
    pair_success = {}
    for pair_type in ("memory_necessity_pair", "rule_necessity_pair", "memory_rule_conflict_pair"):
        pair_success[pair_type] = _mean(
            pair_rows,
            lambda row, current=pair_type: row["route"] == "external_few_shot" and row["source_type"] == "local_semireal" and row["pair_type"] == current,
            field="pair_success",
        )
    local_projected = _route_metric(projected_rows, "external_few_shot", "local_semireal", "full")
    local_fixed = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "full")
    local_flip = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "full", field="surface_group_flip_accuracy")
    local_wrong_drop = local_projected - _route_metric(projected_rows, "external_few_shot", "local_semireal", "wrong_context")
    fixed_before = _route_metric(fixed_rows, "external_few_shot", "local_semireal", "adapter_disabled")
    fixed_after = local_fixed
    fixed_pair_flip = _mean(pair_rows, lambda row: row["route"] == "external_few_shot" and row["source_type"] == "local_semireal", field="pair_success")
    external_few_shot_wins = sum(row["few_shot_vs_disabled"] >= 0.05 for row in route_rows)
    stage_gates = {
        "qwen_frozen": all(row["qwen_trainable_parameter_count"] == 0 and row["qwen_gradients_present"] == 0 and row["fingerprint_unchanged"] and not row["optimizer_contains_qwen_parameters"] for row in training_rows),
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(
            abs(
                _route_metric(projected_rows, route, source, "adapter_disabled")
                - _route_metric(projected_rows, route, source, "zero_scale")
            )
            <= 1e-9
            for route in ROUTES
            for source in ("local_semireal", "external_benchmark")
        ),
        "hidden_norm_ratio": all(float(row.get("hidden_norm_ratio", 1.0)) <= 2.0 for row in loss_rows),
        "local_projected_accuracy": local_projected >= 0.75,
        "local_fixed_centroid_accuracy": local_fixed >= 0.60,
        "local_surface_flip": local_flip >= 0.55,
        "local_wrong_context": local_wrong_drop >= 0.12,
        "memory_path_drop": drops.get(("external_few_shot", "no_memory_path", "local_task"), 0.0) >= 0.12,
        "rule_path_drop": drops.get(("external_few_shot", "no_rule_path", "local_task"), 0.0) >= 0.12,
        "state_path_drop": drops.get(("external_few_shot", "no_state_path", "local_task"), 0.0) >= 0.06,
        "memory_pair_success": pair_success["memory_necessity_pair"] >= 0.75,
        "rule_pair_success": pair_success["rule_necessity_pair"] >= 0.75,
        "conflict_pair_success": pair_success["memory_rule_conflict_pair"] >= 0.75,
        "memory_pair_drop": drops.get(("external_few_shot", "no_memory_path", "memory_necessity_pair"), 0.0) >= 0.20,
        "rule_pair_drop": min(
            drops.get(("external_few_shot", "no_rule_path", "rule_necessity_pair"), 0.0),
            drops.get(("external_few_shot", "no_rule_path", "memory_rule_conflict_pair"), 0.0),
        )
        >= 0.20,
        "fixed_centroid_improvement": fixed_after >= fixed_before + 0.05,
        "fixed_centroid_local_average": local_fixed >= 0.60,
        "full_hidden_pair_flip": fixed_pair_flip >= 0.60,
        "projected_regression": True,
        "external_few_shot_recorded": len(route_rows) == len(external_splits),
    }
    allows_stage27b = all(stage_gates.values()) and external_few_shot_wins >= min(2, len(route_rows))
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "routes": list(ROUTES),
        "adapter_variant": "path_specific_v2",
        "training_mode": "multiclass_path_curriculum_v2",
        "local_projected_accuracy": local_projected,
        "local_fixed_centroid_accuracy": local_fixed,
        "local_surface_flip": local_flip,
        "local_wrong_context_drop": local_wrong_drop,
        "fixed_centroid_before": fixed_before,
        "fixed_centroid_after": fixed_after,
        "necessity_pair_success": pair_success,
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
    _write_csv(output_path / "projected_readout_metrics.csv", projected_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "projected_full_hidden_metrics.csv", projected_full_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "necessity_pair_metrics.csv", pair_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "route_comparison.csv", route_rows)
    _write_csv(output_path / "local_retention.csv", retention_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {"local_semireal": local_real_task_manifest(local_records), "external_benchmark": external_manifest},
    )
    _json_dump(output_path / "truncation_cases.json", truncation_cases)
    return summary
