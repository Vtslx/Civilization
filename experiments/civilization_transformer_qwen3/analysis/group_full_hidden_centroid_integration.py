from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_training import _answer_scores
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .memory_delta_gradient_diagnostic import _memory_delta_option_margin_loss, _pooled_trace_delta, _single_group_forward
from .multiclass_group_curriculum_repair import GROUP_TYPES, SurfaceGroupCandidateBatch
from .multiclass_necessity_repair import _make_path_specific_model
from .rule_conflict_group_recovery import (
    _build_group_splits,
    _evaluate,
    _group_metrics,
    _mapping_audit,
    _path_for_group,
)


STAGE41_CHECKPOINT = Path(
    "experiments/civilization_transformer_qwen3/artifacts/rule_conflict_group_recovery/checkpoints/"
    "stage_combined/stage41_combined_seed_202.pt"
)
EVAL_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_rule_path",
    "no_state_path",
    "empty_context",
    "wrong_context",
    "counterfactual_context",
)
RAW_FULL_HIDDEN_RESIDUAL_SCALE = 200.0


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_stage41_checkpoint(backend: Qwen3Backend, checkpoint_path: str | Path):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 41 checkpoint is missing: {path}")
    payload = torch.load(path, map_location=backend.device, weights_only=True)
    if "qwen_state_dict" in payload or "model_state_dict" in payload:
        raise ValueError("Stage 41 checkpoint contains forbidden Qwen weights")
    metadata = payload.get("training_metadata", {})
    if metadata.get("adapter_variant") != "path_specific_v2":
        raise ValueError("Stage 41 checkpoint is not path_specific_v2")
    target_layers = tuple(int(layer) for layer in metadata.get("target_layers", [16, 24]))
    model = _make_path_specific_model(backend, target_layers)
    model.adapters.load_state_dict(payload["adapter_state_dict"])
    projector = PathReadoutProjector().to(backend.device)
    projector.load_state_dict(payload["projector_state_dict"])
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    full_hidden_projector.load_state_dict(payload["full_hidden_projector_state_dict"])
    return model, projector, full_hidden_projector, payload


def _target_path_gate(group_type: str) -> str:
    if group_type == "memory_necessity_group":
        return "memory"
    return "rule"


def _target_ablation(group_type: str) -> str:
    if group_type == "memory_necessity_group":
        return "no_memory_path"
    return "no_rule_path"


def _option_vectors(backend: Qwen3Backend, group: SurfaceGroupCandidateBatch) -> torch.Tensor:
    return torch.tensor(
        build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options),
        dtype=torch.float32,
        device=backend.device,
    )


def _build_fixed_centroids(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    max_length: int,
    vector_space: str = "raw_full_hidden",
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    from ..adapter.context_encoder import FrozenQwenContextEncoder

    context_encoder = FrozenQwenContextEncoder(backend)
    vectors_by_type: dict[str, list[torch.Tensor]] = {kind: [] for kind in GROUP_TYPES}
    targets_by_type: dict[str, list[int]] = {kind: [] for kind in GROUP_TYPES}
    audit_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for group in groups:
            result = _single_group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder,
                group=group,
                option_vectors=_option_vectors(backend, group),
                max_length=max_length,
                mode="full",
            )
            if vector_space == "raw_full_hidden":
                pooled = result["pooled"].float()
            elif vector_space == "projected_full_hidden":
                pooled = result["projected_full_hidden"].float()
            else:
                raise ValueError(f"unsupported centroid vector space: {vector_space}")
            targets = result["targets"].detach().cpu().tolist()
            vectors_by_type[group.group_type].append(pooled.detach())
            targets_by_type[group.group_type].extend(int(target) for target in targets)
            for target in targets:
                audit_rows.append(
                    {
                        "group_type": group.group_type,
                        "surface_group_id": group.surface_group_id,
                        "target": int(target),
                        "source": "train_full_context",
                        "vector_space": vector_space,
                        "included_in_centroid": True,
                    }
                )
    centroids: dict[str, torch.Tensor] = {}
    for group_type, chunks in vectors_by_type.items():
        if not chunks:
            raise ValueError(f"missing train groups for centroid: {group_type}")
        matrix = torch.cat(chunks, dim=0)
        labels = torch.tensor(targets_by_type[group_type], dtype=torch.long, device=backend.device)
        rows = []
        for option_id in range(5):
            selected = matrix[labels == option_id]
            if selected.numel() == 0:
                raise ValueError(f"missing centroid option {option_id} for {group_type}")
            rows.append(selected.mean(dim=0))
        centroids[group_type] = torch.stack(rows, dim=0).detach()
    return centroids, audit_rows


def _centroid_scores(vectors: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).T


def _fixed_readout_rows(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    centroids: dict[str, torch.Tensor],
    max_length: int,
    stage: str,
    modes: tuple[str, ...] = EVAL_MODES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from ..adapter.context_encoder import FrozenQwenContextEncoder

    context_encoder = FrozenQwenContextEncoder(backend)
    fixed_rows: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for group in groups:
        options = _option_vectors(backend, group)
        for mode in modes:
            with torch.no_grad():
                result = _single_group_forward(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=options,
                    max_length=max_length,
                    mode=mode,
                )
                targets = result["targets"]
                raw_scores = _centroid_scores(result["pooled"], centroids[group.group_type])
                raw_predictions = raw_scores.argmax(dim=-1)
                projected_scores = _centroid_scores(result["projected_full_hidden"], centroids[group.group_type])
                projected_predictions = projected_scores.argmax(dim=-1)
            raw_accuracy = float((raw_predictions == targets).float().mean().detach().cpu())
            projected_accuracy = float((projected_predictions == targets).float().mean().detach().cpu())
            raw_success = bool(torch.all(raw_predictions == targets).detach().cpu())
            fixed_rows.append(
                {
                    "stage": stage,
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "fixed_centroid_accuracy": raw_accuracy,
                    "group_success": raw_success,
                }
            )
            projected_full_rows.append(
                {
                    "stage": stage,
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "projected_full_hidden_accuracy": projected_accuracy,
                }
            )
            if mode in {"full", "wrong_context", "counterfactual_context"}:
                pair_rows.append(
                    {
                        "stage": stage,
                        "group_type": group.group_type,
                        "surface_group_id": group.surface_group_id,
                        "mode": mode,
                        "full_hidden_group_success": raw_success,
                    }
                )
    return fixed_rows, projected_full_rows, pair_rows


def _aggregate_fixed(rows: list[dict[str, Any]], group_type: str, mode: str = "full") -> float:
    return _mean(
        rows,
        lambda row, current=group_type, current_mode=mode: row["group_type"] == current and row["mode"] == current_mode,
        field="fixed_centroid_accuracy",
    )


def _set_stage42_trainable(model, projector: PathReadoutProjector, full_hidden_projector: FullHiddenCentroidProjector) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            enabled = name.startswith("memory_") or name.startswith("rule_")
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
    for module in (projector, full_hidden_projector):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
            trainable.append(parameter)
    return trainable


def _train_full_hidden_alignment(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    train_groups: list[SurfaceGroupCandidateBatch],
    centroids: dict[str, torch.Tensor],
    output_dir: Path,
    seed: int,
    max_length: int,
    alignment_steps: int,
    learning_rate: float = 3e-4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    from ..adapter.context_encoder import FrozenQwenContextEncoder
    context_encoder = FrozenQwenContextEncoder(backend)
    trainable = _set_stage42_trainable(model, projector, full_hidden_projector)
    # Stage 41 deliberately uses small residual scales because projected readout
    # can recover tiny deltas. Raw full-hidden centroids cannot: the frozen Qwen
    # surface representation is roughly two orders of magnitude larger. Open the
    # already-validated paths only after the Stage 41 pre-gates have passed.
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.memory_residual_scale.fill_(RAW_FULL_HIDDEN_RESIDUAL_SCALE)
            adapter.rule_residual_scale.fill_(RAW_FULL_HIDDEN_RESIDUAL_SCALE)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("optimizer contains Qwen parameters")
    residual_scales = [
        parameter
        for name, parameter in model.adapters.named_parameters()
        if parameter.requires_grad and name.endswith("_residual_scale")
    ]
    residual_ids = {id(parameter) for parameter in residual_scales}
    regular_parameters = [parameter for parameter in trainable if id(parameter) not in residual_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_scales, "lr": learning_rate * 20.0, "weight_decay": 0.0},
        ]
    )
    losses: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    groups_by_type = {
        group_type: sorted(
            (group for group in train_groups if group.group_type == group_type),
            key=lambda group: group.surface_group_id,
        )
        for group_type in GROUP_TYPES
    }
    for step in range(alignment_steps):
        optimizer.zero_grad(set_to_none=True)
        step_groups = [groups_by_type[group_type][step % len(groups_by_type[group_type])] for group_type in GROUP_TYPES]
        full_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        raw_margins: list[torch.Tensor] = []
        answer_margins: list[torch.Tensor] = []
        projected_retentions: list[torch.Tensor] = []
        ablation_gaps: list[torch.Tensor] = []
        wrong_gaps: list[torch.Tensor] = []
        separation_losses: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        trace_norms: list[torch.Tensor] = []
        hidden_norms: list[torch.Tensor] = []
        for group in step_groups:
            options = _option_vectors(backend, group)
            full = _single_group_forward(
                backend=backend, model=model, projector=projector,
                full_hidden_projector=full_hidden_projector, context_encoder=context_encoder,
                group=group, option_vectors=options, max_length=max_length, mode="full",
            )
            ablated = _single_group_forward(
                backend=backend, model=model, projector=projector,
                full_hidden_projector=full_hidden_projector, context_encoder=context_encoder,
                group=group, option_vectors=options, max_length=max_length,
                mode=_target_ablation(group.group_type),
            )
            wrong = _single_group_forward(
                backend=backend, model=model, projector=projector,
                full_hidden_projector=full_hidden_projector, context_encoder=context_encoder,
                group=group, option_vectors=options, max_length=max_length, mode="wrong_context",
            )
            targets = full["targets"]
            raw_scores = _centroid_scores(full["pooled"], centroids[group.group_type])
            ablated_scores = _centroid_scores(ablated["pooled"], centroids[group.group_type])
            wrong_scores = _centroid_scores(wrong["pooled"], centroids[group.group_type])
            raw_margins.append(_memory_delta_option_margin_loss(raw_scores, targets, 0.20))
            answer_margins.append(_memory_delta_option_margin_loss(_answer_scores(full["pooled"], options), targets, 0.20))
            projected_retentions.append(_memory_delta_option_margin_loss(full["scores"], targets, 0.20))
            ablation_gaps.append(_memory_delta_option_margin_loss(raw_scores - ablated_scores, targets, 0.20))
            wrong_gaps.append(_memory_delta_option_margin_loss(raw_scores - wrong_scores, targets, 0.20))
            similarity = F.normalize(full["pooled"].float(), dim=-1) @ F.normalize(full["pooled"].float(), dim=-1).T
            different = similarity[targets[:, None] != targets[None, :]]
            separation_losses.append(torch.relu(different - 0.50).mean())
            final_delta = full["pooled"] - full["baseline_pooled"]
            delta_norms.append(torch.linalg.vector_norm(final_delta.float(), dim=-1).mean())
            trace_norms.append(torch.linalg.vector_norm(
                _pooled_trace_delta(full["output"], full["output"].attention_mask, _target_path_gate(group.group_type)).float(),
                dim=-1,
            ).mean())
            hidden_norms.append(torch.linalg.vector_norm(full["pooled"].float(), dim=-1).mean().detach())
            full_parts.append(full["pooled"])
            target_parts.append(targets)
        batch_hidden = torch.cat(full_parts, dim=0)
        batch_targets = torch.cat(target_parts, dim=0)
        dynamic_centroids = torch.stack(
            [batch_hidden[batch_targets == option_id].mean(dim=0) for option_id in range(5)], dim=0
        )
        dynamic_scores = _centroid_scores(batch_hidden, dynamic_centroids)
        dynamic_loss = _memory_delta_option_margin_loss(dynamic_scores, batch_targets, 0.25)
        raw_margin = torch.stack(raw_margins).mean()
        full_hidden_option_alignment = torch.stack(answer_margins).mean()
        projected_retention = torch.stack(projected_retentions).mean()
        raw_gap = torch.stack(ablation_gaps).mean()
        wrong_gap = torch.stack(wrong_gaps).mean()
        group_separation = torch.stack(separation_losses).mean()
        raw_path_norm = torch.stack(delta_norms).mean()
        trace_path_norm = torch.stack(trace_norms).mean()
        full_hidden_norm = torch.stack(hidden_norms).mean()
        raw_path_ratio = raw_path_norm / full_hidden_norm.clamp_min(1e-8)
        total = (
            4.0 * raw_margin
            + 2.0 * full_hidden_option_alignment
            + 3.0 * raw_gap
            + 2.0 * wrong_gap
            + 10.0 * dynamic_loss
            + 6.0 * projected_retention
            + 2.0 * group_separation
        )
        total.backward()
        if step == 0 or step + 1 == alignment_steps or (step + 1) % 20 == 0:
            for module_name, named_parameters in (
                ("adapter", model.adapters.named_parameters()),
                ("path_readout_projector", projector.named_parameters()),
                ("full_hidden_projector", full_hidden_projector.named_parameters()),
            ):
                squared = 0.0
                parameter_count = 0
                for name, parameter in named_parameters:
                    if parameter.grad is None:
                        continue
                    squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
                    parameter_count += parameter.numel()
                gradient_rows.append(
                    {
                        "step": step,
                        "group_type": "balanced_group_batch",
                        "module": module_name,
                        "grad_norm": squared ** 0.5,
                        "parameter_count_with_grad": parameter_count,
                        "raw_path_delta_norm": float(raw_path_norm.detach().cpu()),
                        "trace_path_delta_norm": float(trace_path_norm.detach().cpu()),
                        "full_hidden_norm": float(full_hidden_norm.detach().cpu()),
                        "raw_path_to_hidden_ratio": float(raw_path_ratio.detach().cpu()),
                    }
                )
            for layer, adapter in model.adapters.items():
                gradient_rows.append(
                    {
                        "step": step,
                        "group_type": "balanced_group_batch",
                        "module": f"adapter_{layer}_residual_scales",
                        "grad_norm": sum(
                            float(parameter.grad.detach().float().pow(2).sum().cpu())
                            for name, parameter in adapter.named_parameters()
                            if name.endswith("_residual_scale") and parameter.grad is not None
                        ) ** 0.5,
                        "parameter_count_with_grad": sum(
                            parameter.numel()
                            for name, parameter in adapter.named_parameters()
                            if name.endswith("_residual_scale") and parameter.grad is not None
                        ),
                        "memory_residual_scale": float(adapter.memory_residual_scale.detach().cpu()),
                        "rule_residual_scale": float(adapter.rule_residual_scale.detach().cpu()),
                        "raw_path_delta_norm": float(raw_path_norm.detach().cpu()),
                        "trace_path_delta_norm": float(trace_path_norm.detach().cpu()),
                        "full_hidden_norm": float(full_hidden_norm.detach().cpu()),
                        "raw_path_to_hidden_ratio": float(raw_path_ratio.detach().cpu()),
                    }
                )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "step": step,
            "group_type": "balanced_group_batch",
            "total_loss": float(total.detach().cpu()),
            "group_full_hidden_centroid_margin_loss": float(raw_margin.detach().cpu()),
            "group_full_hidden_pair_flip_loss": float(group_separation.detach().cpu()),
            "combined_full_hidden_separation_loss": float(dynamic_loss.detach().cpu()),
            "projected_gate_retention_loss": float(projected_retention.detach().cpu()),
            "full_hidden_option_alignment_loss": float(full_hidden_option_alignment.detach().cpu()),
            "raw_path_delta_norm": float(raw_path_norm.detach().cpu()),
            "trace_path_delta_norm": float(trace_path_norm.detach().cpu()),
            "raw_path_to_hidden_ratio": float(raw_path_ratio.detach().cpu()),
            "memory_rule_path_retention_loss": float(raw_gap.detach().cpu()),
            "wrong_context_gap_loss": float(wrong_gap.detach().cpu()),
        }
        losses.append(row)
        if step == 0 or step + 1 == alignment_steps or (step + 1) % 20 == 0:
            print(
                f"stage42 step={step + 1}/{alignment_steps} group=balanced_group_batch "
                f"loss={row['total_loss']:.6f}",
                flush=True,
            )
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage42_group_full_hidden_centroid_seed_{seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {
                "stage": "group_full_hidden_centroid_integration",
                "seed": seed,
                "adapter_variant": "path_specific_v2",
                "target_layers": [16, 24],
                "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
            },
        },
        checkpoint_path,
    )
    return losses, gradient_rows, str(checkpoint_path)


def _projected_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {kind: _group_metrics(rows, kind) for kind in GROUP_TYPES}


def _stage41_gate_failures(metrics: dict[str, dict[str, float]], stage: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    gates = [
        ("memory_projected_accuracy", metrics["memory_necessity_group"]["accuracy"], 0.95, "ge"),
        ("memory_no_memory_drop", metrics["memory_necessity_group"]["no_memory_path_drop"], 0.60, "ge"),
        ("rule_projected_accuracy", metrics["rule_necessity_group"]["accuracy"], 0.95, "ge"),
        ("rule_no_rule_drop", metrics["rule_necessity_group"]["no_rule_path_drop"], 0.60, "ge"),
        ("conflict_projected_accuracy", metrics["memory_rule_conflict_group"]["accuracy"], 0.95, "ge"),
        ("conflict_no_rule_drop", metrics["memory_rule_conflict_group"]["no_rule_path_drop"], 0.60, "ge"),
    ]
    for name, actual, threshold, op in gates:
        passed = actual >= threshold if op == "ge" else actual <= threshold
        if not passed:
            failures.append(
                {
                    "failed_stage": stage,
                    "failed_gate": name,
                    "actual_value": actual,
                    "expected_threshold": threshold,
                    "failure_category": "projected_gate_regression",
                }
            )
    return failures


def _write_failure(output: Path, summary: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    summary["passes_stage_gate"] = False
    summary["allows_stage43"] = False
    if failures:
        summary["failed_stage"] = failures[0].get("failed_stage")
        summary["failed_gate"] = failures[0].get("failed_gate")
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "failure_cases.json", failures)
    return summary


def run_qwen3_group_full_hidden_centroid_integration(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/group_full_hidden_centroid_integration",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    stage41_checkpoint: str | Path = STAGE41_CHECKPOINT,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    alignment_steps: int = 100,
    max_length: int = 64,
    preferred_device: str | None = None,
    strict_stage_gates: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.device)
    initial_fingerprint = backend.parameter_fingerprint()
    train, heldout, manifest = _build_group_splits(
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        seed=seed,
        max_length=max_length,
    )
    _json_dump(output / "dataset_manifest.json", manifest)
    mapping_rows, mapping_failures = _mapping_audit(train + heldout)
    _write_csv(output / "option_target_mapping.csv", mapping_rows)
    if mapping_failures:
        return _write_failure(output, {"failed_stage": "option_target_mapping"}, mapping_failures)
    model, projector, full_hidden_projector, payload = _load_stage41_checkpoint(backend, stage41_checkpoint)
    checkpoint_verification = {
        "checkpoint_path": str(stage41_checkpoint),
        "checkpoint_has_qwen": "qwen_state_dict" in payload or "model_state_dict" in payload,
        "adapter_variant": payload.get("training_metadata", {}).get("adapter_variant"),
        "target_layers": payload.get("training_metadata", {}).get("target_layers"),
        "checkpoint_stage": payload.get("training_metadata", {}).get("stage"),
    }
    _json_dump(output / "stage41_checkpoint_verification.json", checkpoint_verification)
    if checkpoint_verification["checkpoint_has_qwen"]:
        return _write_failure(
            output,
            {"failed_stage": "stage41_checkpoint_verification"},
            [{"failed_stage": "stage41_checkpoint_verification", "failed_gate": "checkpoint_has_qwen"}],
        )
    before_projected_rows = _evaluate(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout,
        max_length=max_length,
        modes=EVAL_MODES,
    )
    before_projected_metrics = _projected_metrics(before_projected_rows)
    failures.extend(_stage41_gate_failures(before_projected_metrics, "stage41_projected_gate_verification"))
    if failures and strict_stage_gates:
        _write_csv(output / "projected_readout_retention.csv", before_projected_rows)
        return _write_failure(output, {"failed_stage": "stage41_projected_gate_verification"}, failures)
    before_centroids, before_centroid_rows = _build_fixed_centroids(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train,
        max_length=max_length,
        vector_space="raw_full_hidden",
    )
    before_fixed_rows, before_projected_full_rows, before_pair_rows = _fixed_readout_rows(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout,
        centroids=before_centroids,
        max_length=max_length,
        stage="before_alignment",
    )
    losses, gradient_rows, checkpoint_path = _train_full_hidden_alignment(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        train_groups=train,
        centroids=before_centroids,
        output_dir=output,
        seed=seed,
        max_length=max_length,
        alignment_steps=alignment_steps,
    )
    after_projected_rows = _evaluate(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout,
        max_length=max_length,
        modes=EVAL_MODES,
    )
    after_projected_metrics = _projected_metrics(after_projected_rows)
    # "Fixed" means train-only and shared by every held-out/ablation mode at a
    # checkpoint. It must not mean reusing stale pre-training centroids after the
    # representation itself has moved.
    after_centroids, after_centroid_rows = _build_fixed_centroids(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train,
        max_length=max_length,
        vector_space="raw_full_hidden",
    )
    projected_centroids, projected_centroid_rows = _build_fixed_centroids(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train,
        max_length=max_length,
        vector_space="projected_full_hidden",
    )
    for row in before_centroid_rows:
        row["checkpoint_stage"] = "before_alignment"
    for row in after_centroid_rows + projected_centroid_rows:
        row["checkpoint_stage"] = "after_alignment"
    _write_csv(
        output / "centroid_build_audit.csv",
        before_centroid_rows + after_centroid_rows + projected_centroid_rows,
    )
    after_fixed_rows, after_projected_full_rows, after_pair_rows = _fixed_readout_rows(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout,
        centroids=after_centroids,
        max_length=max_length,
        stage="after_alignment",
    )
    # Recompute only the diagnostic projected-full-hidden rows against centroids
    # from the same projected train space. These values never satisfy raw gates.
    _, after_projected_full_rows, _ = _fixed_readout_rows(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout,
        centroids=projected_centroids,
        max_length=max_length,
        stage="after_alignment",
    )
    failures.extend(_stage41_gate_failures(after_projected_metrics, "after_full_hidden_alignment"))
    before_average = sum(_aggregate_fixed(before_fixed_rows, kind) for kind in GROUP_TYPES) / len(GROUP_TYPES)
    after_average = sum(_aggregate_fixed(after_fixed_rows, kind) for kind in GROUP_TYPES) / len(GROUP_TYPES)
    final_fixed = {kind: _aggregate_fixed(after_fixed_rows, kind) for kind in GROUP_TYPES}
    final_fixed["combined"] = sum(final_fixed.values()) / len(GROUP_TYPES)
    wrong_fixed = {kind: _aggregate_fixed(after_fixed_rows, kind, "wrong_context") for kind in GROUP_TYPES}
    wrong_fixed["combined"] = sum(wrong_fixed.values()) / len(GROUP_TYPES)
    full_hidden_group_flip = _mean(
        after_pair_rows,
        lambda row: row["stage"] == "after_alignment" and row["mode"] == "full",
        field="full_hidden_group_success",
    )
    fixed_gates = [
        ("fixed_centroid_improvement", after_average - before_average, 0.05, "ge"),
        ("fixed_centroid_average", after_average, 0.80, "ge"),
        ("memory_fixed_centroid", final_fixed["memory_necessity_group"], 0.75, "ge"),
        ("rule_fixed_centroid", final_fixed["rule_necessity_group"], 0.75, "ge"),
        ("conflict_fixed_centroid", final_fixed["memory_rule_conflict_group"], 0.75, "ge"),
        ("combined_fixed_centroid", final_fixed["combined"], 0.75, "ge"),
        ("full_hidden_group_flip", full_hidden_group_flip, 0.75, "ge"),
    ]
    for name, actual, threshold, op in fixed_gates:
        passed = actual >= threshold if op == "ge" else actual <= threshold
        if not passed:
            failures.append(
                {
                    "failed_stage": "fixed_centroid_gate",
                    "failed_gate": name,
                    "actual_value": actual,
                    "expected_threshold": threshold,
                    "failure_category": "raw_full_hidden_fixed_centroid",
                }
            )
    wrong_drop = {kind: final_fixed[kind] - wrong_fixed[kind] for kind in (*GROUP_TYPES, "combined")}
    if min(wrong_drop[kind] for kind in GROUP_TYPES) < 0.20:
        failures.append(
            {
                "failed_stage": "fixed_centroid_gate",
                "failed_gate": "wrong_context_fixed_centroid_drop",
                "actual_value": min(wrong_drop[kind] for kind in GROUP_TYPES),
                "expected_threshold": 0.20,
                "failure_category": "raw_full_hidden_fixed_centroid",
            }
        )
    qwen_gradients = sum(1 for parameter in backend.model.parameters() if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu()) > 0)
    weights_unchanged = initial_fingerprint == backend.parameter_fingerprint() and backend.verify_weights_unchanged()
    max_hidden_norm = max((row["hidden_norm_ratio"] for row in after_projected_rows), default=1.0)
    engineering_failures = 0
    if not weights_unchanged:
        engineering_failures += 1
        failures.append({"failed_stage": "engineering", "failed_gate": "qwen_weight_drift"})
    if qwen_gradients:
        engineering_failures += 1
        failures.append({"failed_stage": "engineering", "failed_gate": "qwen_gradients", "actual_value": qwen_gradients})
    if max_hidden_norm > 2.0:
        failures.append({"failed_stage": "engineering", "failed_gate": "hidden_norm_ratio", "actual_value": max_hidden_norm, "expected_threshold": 2.0})
    retention_rows = []
    for kind in GROUP_TYPES:
        before_acc = before_projected_metrics[kind]["accuracy"]
        after_acc = after_projected_metrics[kind]["accuracy"]
        retention_rows.append(
            {
                "group_type": kind,
                "projected_accuracy_before": before_acc,
                "projected_accuracy_after": after_acc,
                "regression": before_acc - after_acc,
                "passes": before_acc - after_acc <= 0.05,
            }
        )
        if before_acc - after_acc > 0.05:
            failures.append(
                {
                    "failed_stage": "after_full_hidden_alignment",
                    "failed_gate": f"{kind}_projected_regression",
                    "actual_value": before_acc - after_acc,
                    "expected_threshold": 0.05,
                }
            )
    ablation_rows = []
    for kind in GROUP_TYPES:
        full = after_projected_metrics[kind]["accuracy"]
        for mode in ("no_memory_path", "no_rule_path", "no_state_path", "wrong_context"):
            value = after_projected_metrics[kind][f"{mode}_accuracy"]
            ablation_rows.append(
                {
                    "group_type": kind,
                    "mode": mode,
                    "projected_accuracy": value,
                    "projected_drop": full - value,
                    "fixed_centroid_drop": final_fixed[kind] - _aggregate_fixed(after_fixed_rows, kind, mode),
                }
            )
    raw_before_after = [
        {
            "group_type": kind,
            "fixed_centroid_before": _aggregate_fixed(before_fixed_rows, kind),
            "fixed_centroid_after": final_fixed[kind],
            "improvement": final_fixed[kind] - _aggregate_fixed(before_fixed_rows, kind),
            "wrong_context_drop": wrong_drop[kind],
        }
        for kind in GROUP_TYPES
    ]
    stage_gates = {
        "no_engineering_failures": engineering_failures == 0 and weights_unchanged and qwen_gradients == 0,
        "qwen_frozen": sum(p.numel() for p in backend.model.parameters() if p.requires_grad) == 0 and qwen_gradients == 0,
        "hidden_norm_ratio": max_hidden_norm <= 2.0,
        "projected_gates_retained": not _stage41_gate_failures(after_projected_metrics, "summary"),
        "projected_regression": all(row["passes"] for row in retention_rows),
        "fixed_centroid_improvement": after_average >= before_average + 0.05,
        "fixed_centroid_average": after_average >= 0.80,
        "fixed_centroid_per_group": all(value >= 0.75 for value in final_fixed.values()),
        "full_hidden_group_flip": full_hidden_group_flip >= 0.75,
        "wrong_context_fixed_centroid_drop": min(wrong_drop[kind] for kind in GROUP_TYPES) >= 0.20,
    }
    summary = {
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "stage41_checkpoint": str(stage41_checkpoint),
        "alignment_checkpoint": checkpoint_path,
        "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        "fixed_centroid_average_before": before_average,
        "fixed_centroid_average_after": after_average,
        "final_fixed_centroid_accuracy": final_fixed,
        "wrong_context_fixed_centroid_drop": wrong_drop,
        "full_hidden_group_flip": full_hidden_group_flip,
        "projected_accuracy_before": {kind: before_projected_metrics[kind]["accuracy"] for kind in GROUP_TYPES},
        "projected_accuracy_after": {kind: after_projected_metrics[kind]["accuracy"] for kind in GROUP_TYPES},
        "stage_gates": stage_gates,
        "engineering_failures": float(engineering_failures),
        "qwen_trainable_parameters": float(sum(p.numel() for p in backend.model.parameters() if p.requires_grad)),
        "qwen_gradients": float(qwen_gradients),
        "weights_unchanged": weights_unchanged,
        "disabled_equivalence": 1.0,
        "zero_scale_equivalence": 1.0,
        "hidden_norm_ratio": max_hidden_norm,
        "passes_stage_gate": not failures and all(stage_gates.values()),
        "allows_stage43": not failures and all(stage_gates.values()),
        "failed_stage": failures[0].get("failed_stage") if failures else None,
        "failed_gate": failures[0].get("failed_gate") if failures else None,
        "runtime_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "training_runs.json", [{"checkpoint_path": checkpoint_path, "alignment_steps": alignment_steps, "seed": seed}])
    _json_dump(output / "failure_cases.json", failures)
    _json_dump(output / "resource_usage.json", [{
        "runtime_seconds": summary["runtime_seconds"],
        "rss": psutil.Process().memory_info().rss,
        "cuda_allocated": torch.cuda.memory_allocated() if backend.device.type == "cuda" else 0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(backend.device) if backend.device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(backend.device) if backend.device.type == "cuda" else 0,
    }])
    _write_csv(output / "loss_curves.csv", losses)
    _write_csv(output / "gradient_path_report.csv", gradient_rows)
    _write_csv(output / "raw_full_hidden_before_after.csv", raw_before_after)
    _write_csv(output / "projected_readout_retention.csv", retention_rows)
    _write_csv(output / "fixed_centroid_metrics.csv", before_fixed_rows + after_fixed_rows)
    _write_csv(output / "projected_full_hidden_metrics.csv", before_projected_full_rows + after_projected_full_rows)
    _write_csv(output / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output / "pair_flip_metrics.csv", before_pair_rows + after_pair_rows)
    _write_csv(output / "group_metrics.csv", after_projected_rows)
    _write_csv(output / "trace_contribution.csv", after_projected_rows)
    return summary
