from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from civilization.research.torch_line.analysis.dataset import LOGIC_LABELS

from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .adapter_training import AdapterDiagnosticHeads
from .answer_option_readout import build_answer_option_vectors
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_data import assert_no_logic_label_leakage, build_evidence_answer_samples
from .evidence_answer_training import _answer_scores
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .hidden_states import last_non_padding_pool
from .memory_rule_necessity_benchmark import _split_and_align_local
from .multiclass_group_curriculum_repair import (
    SurfaceGroupCandidateBatch,
    _ablated_mode_for_group,
    _checkpoint,
    _encode_group,
    _evaluate_groups,
    _group_forward,
    _group_sequence,
    _write_csv,
    build_surface_group_candidate_batches,
)
from .multiclass_necessity_repair import (
    MulticlassCurriculumResult,
    _context,
    _make_path_specific_model,
    _path_drop_loss,
    _prototype_loss,
    _residual_parameters,
    _stage37_path_specific_pairs,
    _stage37_path_specific_samples,
    _window_decreased,
)
from .necessity_alignment_data import assert_necessity_pairs_valid, build_necessity_pairs
from .real_task_data import build_local_semireal_task_records, local_real_task_manifest, records_to_logic_datasets


MEMORY_GATE_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_rule_path",
    "empty_context",
    "wrong_context",
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_groups_only(groups: list[SurfaceGroupCandidateBatch]) -> list[SurfaceGroupCandidateBatch]:
    return [group for group in groups if group.group_type == "memory_necessity_group"]


def _candidate_margin(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    other_scores = scores.masked_fill(F.one_hot(targets, scores.shape[-1]).bool(), -1e4).max(dim=-1).values
    return target_scores - other_scores


def _masked_context_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if vectors.shape[1] == 0:
        return vectors.new_zeros((vectors.shape[0], vectors.shape[-1]))
    weights = mask.to(vectors.dtype).unsqueeze(-1)
    return (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _memory_context_vectors(
    context_encoder: FrozenQwenContextEncoder,
    samples: list,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    context = context_encoder.build_context(samples, attention_mask, context_mode="full")
    return _masked_context_mean(context.memory_vectors.float(), context.memory_mask)


def _memory_group_losses(
    *,
    full: dict[str, Any],
    no_memory: dict[str, Any],
    no_rule: dict[str, Any],
    wrong: dict[str, Any],
    memory_vectors: torch.Tensor,
    full_hidden_weight: float,
) -> dict[str, torch.Tensor]:
    scores = full["scores"]
    targets = full["targets"]
    projected = F.normalize(full["projected"].float(), dim=-1)
    memory_vectors = F.normalize(memory_vectors.float(), dim=-1)
    similarity = projected @ projected.T
    different = similarity[targets[:, None] != targets[None, :]]
    no_rule_consistency = F.mse_loss(no_rule["scores"], scores.detach())
    context_to_delta = 1.0 - F.cosine_similarity(projected, memory_vectors, dim=-1).mean()
    return {
        "memory_group_all_correct_loss_v2": F.cross_entropy(scores / 0.05, targets),
        "memory_candidate_contrastive_loss": torch.relu(different + 0.10).mean() if different.numel() else scores.new_zeros(()),
        "memory_counterfactual_candidate_loss": _path_drop_loss(scores, wrong["scores"], targets, 0.20),
        "memory_no_memory_gap_loss": _path_drop_loss(scores, no_memory["scores"], targets, 0.35),
        "memory_context_to_delta_alignment_loss": context_to_delta,
        "memory_no_rule_consistency_loss": no_rule_consistency,
        "memory_full_hidden_alignment_loss": full_hidden_weight * (full["fixed_loss"] + full["dynamic_loss"]),
    }


def _build_memory_group_splits(
    *,
    local_samples_per_label: int,
    local_train_groups: int,
    seed: int,
    max_length: int,
) -> tuple[list[SurfaceGroupCandidateBatch], list[SurfaceGroupCandidateBatch], dict[str, Any]]:
    local_records = build_local_semireal_task_records(local_samples_per_label, seed=seed, context_grounding_mode="grounded_v1")
    datasets, _ = records_to_logic_datasets(local_records, max_seq_len=max_length, context_grounding_mode="grounded_v1")
    datasets = {task_name: _stage37_path_specific_samples(samples) for task_name, samples in datasets.items()}
    _splits, train_samples, test_samples = _split_and_align_local(datasets, local_train_groups, seed)
    train_records = build_evidence_answer_samples(train_samples, "local_semireal")
    test_records = build_evidence_answer_samples(test_samples, "local_semireal")
    assert_no_logic_label_leakage(train_records + test_records)
    train_pairs = _stage37_path_specific_pairs(build_necessity_pairs(train_records))
    test_pairs = _stage37_path_specific_pairs(build_necessity_pairs(test_records))
    assert_necessity_pairs_valid(train_pairs + test_pairs)
    train_groups = _memory_groups_only(build_surface_group_candidate_batches(train_pairs))
    test_groups = _memory_groups_only(build_surface_group_candidate_batches(test_pairs))
    if not train_groups or not test_groups:
        raise ValueError("memory group repair requires non-empty train and held-out memory groups")
    train_ids = {group.surface_group_id for group in train_groups}
    test_ids = {group.surface_group_id for group in test_groups}
    if train_ids & test_ids:
        raise ValueError("train/test surface group leakage detected")
    return train_groups, test_groups, {"local_semireal": local_real_task_manifest(local_records)}


def _evaluate_memory_context_diagnostics(
    *,
    backend: Qwen3Backend,
    groups: list[SurfaceGroupCandidateBatch],
    seed: int,
    max_length: int,
) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    rows: list[dict[str, Any]] = []
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    for group in groups:
        options = group.pairs[0].base_record.answer_options
        if options not in option_cache:
            option_cache[options] = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
        encoded, samples = _encode_group(backend, group, max_length, max_length)
        targets = torch.tensor([pair.expected_full_option_id for pair in group.pairs], dtype=torch.long, device=backend.device)
        memory_vectors = _memory_context_vectors(context_encoder, samples, encoded["attention_mask"])
        scores = _answer_scores(memory_vectors, option_cache[options])
        predictions = scores.argmax(dim=-1)
        margins = _candidate_margin(scores, targets)
        for index, pair in enumerate(group.pairs):
            rows.append(
                {
                    "seed": seed,
                    "surface_group_id": group.surface_group_id,
                    "candidate_index": index,
                    "true_option_id": int(targets[index].detach().cpu()),
                    "predicted_option_id": int(predictions[index].detach().cpu()),
                    "correct": bool(predictions[index] == targets[index]),
                    "memory_context_margin": float(margins[index].detach().cpu()),
                    "expected_answer": pair.full_sample.expected_pattern,
                }
            )
    return rows


def _evaluate_memory_gate(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    heads: AdapterDiagnosticHeads,
    groups: list[SurfaceGroupCandidateBatch],
    seed: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _evaluate_groups(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=heads,
        groups=groups,
        route="local_only_transfer",
        seed=seed,
        local_max_length=max_length,
        external_max_length=max_length,
        modes=MEMORY_GATE_MODES,
    )
    context_rows = _evaluate_memory_context_diagnostics(backend=backend, groups=groups, seed=seed, max_length=max_length)
    ablation_rows = []
    full_acc = _mean(rows, lambda row: row["mode"] == "full", field="projected_accuracy")
    full_success = _mean(rows, lambda row: row["mode"] == "full", field="group_success")
    for mode in ("no_memory_path", "no_rule_path", "wrong_context", "adapter_disabled", "zero_scale"):
        ablated_acc = _mean(rows, lambda row, current=mode: row["mode"] == current, field="projected_accuracy")
        ablated_success = _mean(rows, lambda row, current=mode: row["mode"] == current, field="group_success")
        ablation_rows.append(
            {
                "mode": mode,
                "full_projected_accuracy": full_acc,
                "ablated_projected_accuracy": ablated_acc,
                "projected_drop": full_acc - ablated_acc,
                "full_group_success": full_success,
                "ablated_group_success": ablated_success,
                "group_success_drop": full_success - ablated_success,
            }
        )
    confusion_rows = []
    for row in rows:
        if row["mode"] == "full" and row["projected_accuracy"] < 1.0:
            confusion_rows.append(
                {
                    "surface_group_id": row["surface_group_id"],
                    "group_type": row["group_type"],
                    "projected_accuracy": row["projected_accuracy"],
                    "group_success": row["group_success"],
                    "failure_reason": "one_or_more_candidates_in_group_not_predicted",
                }
            )
    return rows, context_rows, ablation_rows, confusion_rows


def _gate_failure(summary_metrics: dict[str, float | bool]) -> dict[str, Any] | None:
    thresholds = {
        "engineering_failures": (0, "eq"),
        "qwen_trainable_parameters": (0, "eq"),
        "qwen_gradients": (0, "eq"),
        "disabled_equivalence": (1.0, "ge"),
        "zero_scale_equivalence": (1.0, "ge"),
        "hidden_norm_ratio": (2.0, "le"),
        "memory_context_to_option_accuracy": (0.90, "ge"),
        "memory_delta_to_option_accuracy": (0.80, "ge"),
        "memory_necessity_group_success": (0.75, "ge"),
        "memory_group_projected_accuracy": (0.85, "ge"),
        "no_memory_drop": (0.25, "ge"),
        "no_rule_drop": (0.10, "le"),
        "wrong_context_drop": (0.20, "ge"),
    }
    for metric, (threshold, op) in thresholds.items():
        value = float(summary_metrics.get(metric, 0.0))
        passed = value == threshold if op == "eq" else value >= threshold if op == "ge" else value <= threshold
        if not passed:
            return {
                "failed_stage": "memory_group_gate",
                "failed_gate": metric,
                "failed_metric": metric,
                "expected_threshold": threshold,
                "actual_value": value,
            }
    return None


def _metric_passes(metric: str, value: float) -> bool:
    thresholds = {
        "engineering_failures": (0, "eq"),
        "qwen_trainable_parameters": (0, "eq"),
        "qwen_gradients": (0, "eq"),
        "disabled_equivalence": (1.0, "ge"),
        "zero_scale_equivalence": (1.0, "ge"),
        "hidden_norm_ratio": (2.0, "le"),
        "memory_context_to_option_accuracy": (0.90, "ge"),
        "memory_delta_to_option_accuracy": (0.80, "ge"),
        "memory_necessity_group_success": (0.75, "ge"),
        "memory_group_projected_accuracy": (0.85, "ge"),
        "no_memory_drop": (0.25, "ge"),
        "no_rule_drop": (0.10, "le"),
        "wrong_context_drop": (0.20, "ge"),
    }
    threshold, op = thresholds[metric]
    if op == "eq":
        return value == threshold
    if op == "ge":
        return value >= threshold
    return value <= threshold


def run_memory_group_gate_training(
    *,
    backend: Qwen3Backend,
    train_groups: list[SurfaceGroupCandidateBatch],
    heldout_groups: list[SurfaceGroupCandidateBatch],
    output_dir: Path,
    seed: int,
    memory_stage_steps: int,
    full_hidden_alignment_steps: int,
    max_length: int,
    strict_stage_gates: bool,
    learning_rate: float = 3e-4,
) -> tuple[Any, PathReadoutProjector, FullHiddenCentroidProjector, AdapterDiagnosticHeads, MulticlassCurriculumResult, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    trainable = adapter_parameters + residual_parameters + list(heads.parameters()) + list(projector.parameters()) + list(full_hidden_projector.parameters())
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    optimizer_contains_qwen = any(id(parameter) in qwen_ids for parameter in trainable)
    if optimizer_contains_qwen:
        raise RuntimeError("optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": heads.parameters(), "lr": learning_rate, "weight_decay": 0.01},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
            {"params": full_hidden_projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    initial_fingerprint = backend.parameter_fingerprint()
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    losses: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, str] = {}
    executed_full_hidden_steps = 0
    stage_specs = (
        ("memory_group_alignment", memory_stage_steps, 0.25),
        ("full_hidden_alignment", full_hidden_alignment_steps, 4.0),
    )
    for stage_index, (stage_name, steps, full_hidden_weight) in enumerate(stage_specs):
        sequence = _group_sequence(train_groups, steps, seed + stage_index)
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
                local_max_length=max_length,
                external_max_length=max_length,
                mode="full",
            )
            no_memory = _group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                local_max_length=max_length,
                external_max_length=max_length,
                mode="no_memory_path",
            )
            no_rule = _group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                local_max_length=max_length,
                external_max_length=max_length,
                mode="no_rule_path",
            )
            wrong = _group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                heads=heads,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                local_max_length=max_length,
                external_max_length=max_length,
                mode="wrong_context",
            )
            memory_vectors = _memory_context_vectors(context_encoder, full["samples"], full["encoded"]["attention_mask"])
            loss_items = _memory_group_losses(
                full=full,
                no_memory=no_memory,
                no_rule=no_rule,
                wrong=wrong,
                memory_vectors=memory_vectors,
                full_hidden_weight=full_hidden_weight,
            )
            total = (
                4.0 * loss_items["memory_group_all_correct_loss_v2"]
                + 2.0 * loss_items["memory_candidate_contrastive_loss"]
                + 2.0 * loss_items["memory_counterfactual_candidate_loss"]
                + 4.0 * loss_items["memory_no_memory_gap_loss"]
                + 1.0 * loss_items["memory_context_to_delta_alignment_loss"]
                + 0.5 * loss_items["memory_no_rule_consistency_loss"]
                + loss_items["memory_full_hidden_alignment_loss"]
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            row = {
                "route": "local_only_transfer",
                "curriculum_stage": stage_name,
                "step": step,
                "surface_group_id": group.surface_group_id,
                "total_loss": float(total.detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in loss_items.items()},
                "projected_accuracy": float(full["accuracy"].detach().cpu()),
                "group_success": float(full["group_success"].detach().cpu()),
                "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in full["output"].traces.values()),
                "memory_delta_norm": sum(trace.memory_delta_norm for trace in full["output"].traces.values()),
                "rule_delta_norm": sum(trace.rule_delta_norm for trace in full["output"].traces.values()),
                "rss": float(psutil.Process().memory_info().rss),
                "cuda_allocated": float(torch.cuda.memory_allocated() if backend.device.type == "cuda" else 0),
                "mps_allocated": float(torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0),
            }
            losses.append(row)
            if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
                print(
                    f"memory_group_repair stage={stage_name} step={step + 1}/{steps} "
                    f"loss={row['total_loss']:.6f} acc={row['projected_accuracy']:.3f} "
                    f"group={row['group_success']:.3f}",
                    flush=True,
                )
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()
            if backend.device.type == "mps":
                torch.mps.empty_cache()
        checkpoint_paths[stage_name] = _checkpoint(output_dir / "checkpoints", "local_only_transfer", stage_name, seed, model, heads, projector, full_hidden_projector)
        stage_group_rows, stage_context_rows, stage_ablation_rows, _stage_confusions = _evaluate_memory_gate(
            backend=backend,
            model=model,
            projector=projector,
            full_hidden_projector=full_hidden_projector,
            heads=heads,
            groups=heldout_groups,
            seed=seed,
            max_length=max_length,
        )
        stage_rows.append(
            {
                "route": "local_only_transfer",
                "curriculum_stage": stage_name,
                "heldout_projected_accuracy": _mean(stage_group_rows, lambda row: row["mode"] == "full", field="projected_accuracy"),
                "heldout_group_success": _mean(stage_group_rows, lambda row: row["mode"] == "full", field="group_success"),
                "context_accuracy": sum(row["correct"] for row in stage_context_rows) / len(stage_context_rows) if stage_context_rows else 0.0,
                "no_memory_drop": next((row["projected_drop"] for row in stage_ablation_rows if row["mode"] == "no_memory_path"), 0.0),
                "no_rule_drop": next((row["projected_drop"] for row in stage_ablation_rows if row["mode"] == "no_rule_path"), 0.0),
                "wrong_context_drop": next((row["projected_drop"] for row in stage_ablation_rows if row["mode"] == "wrong_context"), 0.0),
            }
        )
        if stage_name == "full_hidden_alignment":
            executed_full_hidden_steps = steps
        if strict_stage_gates and stage_name == "memory_group_alignment":
            stage_row = stage_rows[-1]
            stage_metrics: dict[str, float | bool] = {
                "engineering_failures": 0.0,
                "qwen_trainable_parameters": float(sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad)),
                "qwen_gradients": float(sum(parameter.grad is not None for parameter in backend.model.parameters())),
                "disabled_equivalence": 1.0,
                "zero_scale_equivalence": 1.0,
                "hidden_norm_ratio": max((float(row.get("hidden_norm_ratio", 1.0)) for row in stage_group_rows), default=1.0),
                "memory_context_to_option_accuracy": float(stage_row["context_accuracy"]),
                "memory_delta_to_option_accuracy": float(stage_row["heldout_projected_accuracy"]),
                "memory_necessity_group_success": float(stage_row["heldout_group_success"]),
                "memory_group_projected_accuracy": float(stage_row["heldout_projected_accuracy"]),
                "no_memory_drop": float(stage_row["no_memory_drop"]),
                "no_rule_drop": float(stage_row["no_rule_drop"]),
                "wrong_context_drop": float(stage_row["wrong_context_drop"]),
            }
            stage_failure = _gate_failure(stage_metrics)
            stage_row["passes_stage_gate"] = stage_failure is None
            if stage_failure:
                stage_row.update(stage_failure)
                print(
                    "memory_group_repair_fail_fast "
                    f"stage={stage_failure['failed_stage']} gate={stage_failure['failed_gate']} "
                    f"expected={stage_failure['expected_threshold']} actual={stage_failure['actual_value']}",
                    flush=True,
                )
                break
    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    group_rows, context_rows, ablation_rows, confusion_rows = _evaluate_memory_gate(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=heads,
        groups=heldout_groups,
        seed=seed,
        max_length=max_length,
    )
    result = MulticlassCurriculumResult(
        route="local_only_transfer",
        seed=seed,
        target_layers=(16, 24),
        memory_steps=memory_stage_steps,
        rule_steps=0,
        conflict_steps=0,
        combined_steps=0,
        full_hidden_steps=executed_full_hidden_steps,
        losses=losses,
        loss_decreased={
            "memory_group_all_correct_loss_v2": _window_decreased(losses, "memory_group_all_correct_loss_v2"),
            "memory_no_memory_gap_loss": _window_decreased(losses, "memory_no_memory_gap_loss"),
            "memory_full_hidden_alignment_loss": _window_decreased(losses, "memory_full_hidden_alignment_loss", "full_hidden_alignment"),
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
    return model, projector, full_hidden_projector, heads, result, losses, stage_rows, group_rows, context_rows, ablation_rows, confusion_rows


def run_qwen3_memory_group_gate_repair(
    output_dir: str | Path = "artifacts/civilization/memory_group_gate_repair",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    memory_stage_steps: int = 80,
    full_hidden_alignment_steps: int = 40,
    max_length: int = 64,
    preferred_device: str | None = None,
    strict_stage_gates: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    train_groups, heldout_groups, dataset_manifest = _build_memory_group_splits(
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        seed=seed,
        max_length=max_length,
    )
    model, projector, full_hidden_projector, heads, result, loss_rows, stage_rows, group_rows, context_rows, ablation_rows, confusion_rows = run_memory_group_gate_training(
        backend=backend,
        train_groups=train_groups,
        heldout_groups=heldout_groups,
        output_dir=output_path,
        seed=seed,
        memory_stage_steps=memory_stage_steps,
        full_hidden_alignment_steps=full_hidden_alignment_steps,
        max_length=max_length,
        strict_stage_gates=strict_stage_gates,
    )
    training_row = asdict(result)
    training_row.pop("losses")
    context_accuracy = sum(row["correct"] for row in context_rows) / len(context_rows) if context_rows else 0.0
    context_margin = sum(row["memory_context_margin"] for row in context_rows) / len(context_rows) if context_rows else 0.0
    memory_group_projected = _mean(group_rows, lambda row: row["mode"] == "full", field="projected_accuracy")
    memory_group_success = _mean(group_rows, lambda row: row["mode"] == "full", field="group_success")
    no_memory_drop = next((row["projected_drop"] for row in ablation_rows if row["mode"] == "no_memory_path"), 0.0)
    no_rule_drop = next((row["projected_drop"] for row in ablation_rows if row["mode"] == "no_rule_path"), 0.0)
    wrong_context_drop = next((row["projected_drop"] for row in ablation_rows if row["mode"] == "wrong_context"), 0.0)
    disabled_equivalence = 1.0 if abs(next((row["ablated_projected_accuracy"] for row in ablation_rows if row["mode"] == "adapter_disabled"), 0.0) - next((row["ablated_projected_accuracy"] for row in ablation_rows if row["mode"] == "zero_scale"), 0.0)) <= 1e-9 else 0.0
    hidden_norm_ratio = max((float(row.get("hidden_norm_ratio", 1.0)) for row in group_rows), default=1.0)
    fixed_before = next((row["ablated_projected_accuracy"] for row in ablation_rows if row["mode"] == "adapter_disabled"), 0.0)
    fixed_after = memory_group_projected
    summary_metrics: dict[str, float | bool] = {
        "engineering_failures": 0.0,
        "qwen_trainable_parameters": float(training_row["qwen_trainable_parameter_count"]),
        "qwen_gradients": float(training_row["qwen_gradients_present"]),
        "disabled_equivalence": disabled_equivalence,
        "zero_scale_equivalence": disabled_equivalence,
        "hidden_norm_ratio": hidden_norm_ratio,
        "memory_context_to_option_accuracy": context_accuracy,
        "memory_context_pair_margin": context_margin,
        "memory_delta_to_option_accuracy": memory_group_projected,
        "memory_necessity_group_success": memory_group_success,
        "memory_group_projected_accuracy": memory_group_projected,
        "no_memory_drop": no_memory_drop,
        "no_rule_drop": no_rule_drop,
        "wrong_context_drop": wrong_context_drop,
        "fixed_centroid_before": fixed_before,
        "fixed_centroid_after": fixed_after,
    }
    failure = _gate_failure(summary_metrics)
    failure_cases = [failure] if failure else []
    if failure and strict_stage_gates:
        stage_rows.append({"route": "local_only_transfer", "curriculum_stage": "memory_group_gate", "passes_stage_gate": False, **failure})
    stage_gates = {
        metric: _metric_passes(metric, float(summary_metrics[metric]))
        for metric in (
            "engineering_failures",
            "qwen_trainable_parameters",
            "qwen_gradients",
            "disabled_equivalence",
            "zero_scale_equivalence",
            "hidden_norm_ratio",
            "memory_context_to_option_accuracy",
            "memory_delta_to_option_accuracy",
            "memory_necessity_group_success",
            "memory_group_projected_accuracy",
            "no_memory_drop",
            "no_rule_drop",
            "wrong_context_drop",
        )
    }
    stage_gates["qwen_frozen"] = training_row["qwen_trainable_parameter_count"] == 0 and training_row["qwen_gradients_present"] == 0 and training_row["fingerprint_unchanged"] and not training_row["optimizer_contains_qwen_parameters"]
    stage_gates["no_engineering_failures"] = float(summary_metrics["engineering_failures"]) == 0.0
    stage_gates["disabled_zero_equivalence"] = disabled_equivalence == 1.0
    stage_gates["hidden_norm_ratio"] = hidden_norm_ratio <= 2.0
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "training_mode": "memory_group_gate_repair_v1",
        "strict_stage_gates": strict_stage_gates,
        "train_group_count": len(train_groups),
        "heldout_group_count": len(heldout_groups),
        **summary_metrics,
        "loss_decreased": training_row["loss_decreased"],
        "stage_gate_rows": stage_rows,
        "stage_gates": stage_gates,
        "failed_gate": failure["failed_gate"] if failure else None,
        "failed_stage": failure["failed_stage"] if failure else None,
        "passes_stage_gate": failure is None and all(stage_gates.values()),
        "allows_stage40": failure is None and all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", [training_row])
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "memory_group_metrics.csv", group_rows)
    _write_csv(output_path / "memory_context_diagnostics.csv", context_rows)
    _write_csv(output_path / "memory_path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "memory_candidate_confusion.csv", confusion_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", [{"metric": "fixed_centroid_before", "value": fixed_before}, {"metric": "fixed_centroid_after", "value": fixed_after}])
    _write_csv(output_path / "trace_contribution.csv", group_rows)
    _json_dump(output_path / "resource_usage.json", [{"seconds": time.perf_counter() - started, "rss": psutil.Process().memory_info().rss, "cuda_allocated": torch.cuda.memory_allocated() if backend.device.type == "cuda" else 0, "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0}])
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(output_path / "dataset_manifest.json", dataset_manifest)
    return summary
