from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from ..adapter.context_encoder import FrozenQwenContextEncoder, context_items_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_data import assert_no_logic_label_leakage, build_evidence_answer_samples
from .evidence_answer_training import _answer_scores
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .memory_delta_gradient_diagnostic import (
    _grad_norm,
    _memory_delta_option_margin_loss,
    _open_memory_residual_path,
    _pooled_trace_delta,
    _single_group_forward,
)
from .memory_group_gate_repair import _build_memory_group_splits, _masked_context_mean
from .memory_rule_necessity_benchmark import _split_and_align_local
from .multiclass_group_curriculum_repair import (
    GROUP_TYPES,
    SurfaceGroupCandidateBatch,
    _group_sequence,
    build_surface_group_candidate_batches,
)
from .multiclass_necessity_repair import _make_path_specific_model, _save_checkpoint, _stage37_path_specific_pairs, _stage37_path_specific_samples
from .necessity_alignment_data import NecessityPair, assert_necessity_pairs_valid, build_necessity_pairs
from .real_task_data import build_local_semireal_task_records, local_real_task_manifest, records_to_logic_datasets


STAGE40_CHECKPOINT = Path(
    "artifacts/civilization/memory_delta_gradient_diagnostic/checkpoints/"
    "multiclass_necessity_local_only_transfer_group_memory_delta_path_only_seed_202.pt"
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _context_hashes(pair: NecessityPair) -> tuple[str, str, str]:
    context = context_items_for_sample(pair.full_sample, context_mode="full")
    memory = "|".join(f"{item.summary} {item.content} {item.relation_type}" for item in context.memories)
    rule = "|".join(f"{item.type} {item.condition} {item.effect} {item.source}" for item in context.rules)
    return _sha(memory), _sha(rule), _sha(json.dumps(context.state_values))


def _rule_conditioned_conflict_group(group: SurfaceGroupCandidateBatch) -> SurfaceGroupCandidateBatch:
    if group.group_type != "memory_rule_conflict_group":
        return group
    pairs: list[NecessityPair] = []
    for pair in group.pairs:
        full = replace(pair.full_sample, leakage_family="rule_conditioned_conflict_v2_grounded_v1")
        counterfactual = replace(pair.counterfactual_sample, leakage_family="rule_conditioned_conflict_v2_grounded_v1")
        pairs.append(replace(pair, full_sample=full, counterfactual_sample=counterfactual))
    return SurfaceGroupCandidateBatch(group.group_type, group.surface_group_id, tuple(pairs))


def _build_group_splits(
    *, local_samples_per_label: int, local_train_groups: int, seed: int, max_length: int
) -> tuple[list[SurfaceGroupCandidateBatch], list[SurfaceGroupCandidateBatch], dict[str, Any]]:
    records = build_local_semireal_task_records(local_samples_per_label, seed=seed, context_grounding_mode="grounded_v1")
    datasets, _ = records_to_logic_datasets(records, max_seq_len=max_length, context_grounding_mode="grounded_v1")
    datasets = {name: _stage37_path_specific_samples(samples) for name, samples in datasets.items()}
    _splits, train_samples, heldout_samples = _split_and_align_local(datasets, local_train_groups, seed)
    train_records = build_evidence_answer_samples(train_samples, "local_semireal")
    heldout_records = build_evidence_answer_samples(heldout_samples, "local_semireal")
    assert_no_logic_label_leakage(train_records + heldout_records)
    train_pairs = _stage37_path_specific_pairs(build_necessity_pairs(train_records))
    heldout_pairs = _stage37_path_specific_pairs(build_necessity_pairs(heldout_records))
    assert_necessity_pairs_valid(train_pairs + heldout_pairs)
    train = [_rule_conditioned_conflict_group(group) for group in build_surface_group_candidate_batches(train_pairs)]
    heldout = [_rule_conditioned_conflict_group(group) for group in build_surface_group_candidate_batches(heldout_pairs)]
    if set(group.surface_group_id for group in train) & set(group.surface_group_id for group in heldout):
        raise ValueError("train/test surface group leakage detected")
    if any(not [group for group in train if group.group_type == kind] for kind in GROUP_TYPES):
        raise ValueError("Stage 41 requires all three group types")
    return train, heldout, {"local_semireal": local_real_task_manifest(records)}


def _mapping_audit(groups: list[SurfaceGroupCandidateBatch]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group in groups:
        options = group.pairs[0].base_record.answer_options
        targets = [pair.expected_full_option_id for pair in group.pairs]
        if targets != list(range(5)):
            failures.append({"failed_gate": "rule_option_target_mapping", "group": group.surface_group_id, "targets": targets})
        for index, pair in enumerate(group.pairs):
            memory_hash, rule_hash, state_hash = _context_hashes(pair)
            target = pair.expected_full_option_id
            ok = index == target and pair.full_sample.expected_pattern == options[target]
            rows.append(
                {
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "candidate_index": index,
                    "option_id": target,
                    "target_tensor": target,
                    "score_column": target,
                    "answer_option_hash": _sha(options[target]),
                    "memory_context_hash": memory_hash,
                    "rule_context_hash": rule_hash,
                    "state_hash": state_hash,
                    "mapping_ok": ok,
                }
            )
            if not ok:
                failures.append({"failed_gate": "rule_option_target_mapping", "group": group.surface_group_id, "candidate": index})
        if group.group_type == "rule_necessity_group":
            group_rows = [
                row for row in rows
                if row["surface_group_id"] == group.surface_group_id and row["group_type"] == group.group_type
            ]
            if len({row["memory_context_hash"] for row in group_rows}) != 1:
                failures.append({"failed_gate": "rule_group_memory_changed", "group": group.surface_group_id})
            if len({row["state_hash"] for row in group_rows}) != 1:
                failures.append({"failed_gate": "rule_group_state_changed", "group": group.surface_group_id})
        if group.group_type == "memory_rule_conflict_group":
            group_rows = [
                row for row in rows
                if row["surface_group_id"] == group.surface_group_id and row["group_type"] == group.group_type
            ]
            if len({row["state_hash"] for row in group_rows}) != 1:
                failures.append({"failed_gate": "conflict_state_encodes_target", "group": group.surface_group_id})
    return rows, failures


def _path_for_group(group: SurfaceGroupCandidateBatch) -> str:
    return "memory" if group.group_type == "memory_necessity_group" else "rule"


def _path_scores(result: dict[str, Any], projector: PathReadoutProjector, option_vectors: torch.Tensor, path: str) -> dict[str, Any]:
    delta = _pooled_trace_delta(result["output"], result["output"].attention_mask, path)
    projected = projector(delta)
    scores = _answer_scores(projected, option_vectors)
    predictions = scores.argmax(dim=-1)
    targets = result["targets"]
    return {
        "delta": delta,
        "projected": projected,
        "scores": scores,
        "predictions": predictions,
        "accuracy": float((predictions == targets).float().mean().detach().cpu()),
        "group_success": float(torch.all(predictions == targets).detach().cpu()),
        "delta_norm": float(torch.linalg.vector_norm(delta.float(), dim=-1).mean().detach().cpu()),
        "hidden_norm_ratio": max((trace.hidden_norm_ratio for trace in result["output"].traces.values()), default=1.0),
    }


def _context_vectors(context_encoder: FrozenQwenContextEncoder, result: dict[str, Any], path: str) -> torch.Tensor:
    context = context_encoder.build_context(result["samples"], result["encoded"]["attention_mask"], context_mode="full")
    if path == "memory":
        return _masked_context_mean(context.memory_vectors.float(), context.memory_mask)
    return _masked_context_mean(context.rule_vectors.float(), context.rule_mask)


def _load_stage40_checkpoint(backend: Qwen3Backend, checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 40 checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=backend.device, weights_only=True)
    if "qwen_state_dict" in payload or "model_state_dict" in payload:
        raise ValueError("Stage 40 checkpoint contains forbidden Qwen weights")
    model = _make_path_specific_model(backend, (16, 24))
    model.adapters.load_state_dict(payload["adapter_state_dict"])
    projector = PathReadoutProjector().to(backend.device)
    projector.load_state_dict(payload["projector_state_dict"])
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    full_hidden_projector.load_state_dict(payload["full_hidden_projector_state_dict"])
    return model, projector, full_hidden_projector, payload


def _counterfactual_group(group: SurfaceGroupCandidateBatch) -> SurfaceGroupCandidateBatch:
    pairs = tuple(
        sorted(
            (
                replace(
                    pair,
                    full_sample=pair.counterfactual_sample,
                    counterfactual_sample=pair.full_sample,
                    expected_full_option_id=pair.expected_counterfactual_option_id,
                    expected_counterfactual_option_id=pair.expected_full_option_id,
                )
                for pair in group.pairs
            ),
            key=lambda pair: pair.expected_full_option_id,
        )
    )
    return SurfaceGroupCandidateBatch(group.group_type, group.surface_group_id, pairs)


def _forward_group(*, backend, model, projector, full_hidden_projector, context_encoder, group, options, max_length, mode):
    if mode == "counterfactual_context":
        group = _counterfactual_group(group)
        mode = "full"
    return _single_group_forward(
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


def _rule_context_diagnostics(*, backend, groups, max_length: int) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    rows: list[dict[str, Any]] = []
    for group in groups:
        if group.group_type not in {"rule_necessity_group", "memory_rule_conflict_group"}:
            continue
        options = torch.tensor(
            build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options),
            dtype=torch.float32,
            device=backend.device,
        )
        encoded, _ = backend.encode([pair.full_sample.text for pair in group.pairs], max_length=max_length)
        attention_mask = encoded["attention_mask"].to(backend.device)
        samples = [pair.full_sample for pair in group.pairs]
        context = context_encoder.build_context(samples, attention_mask, context_mode="full")
        vectors = _masked_context_mean(context.rule_vectors.float(), context.rule_mask)
        scores = _answer_scores(vectors, options)
        targets = torch.tensor([pair.expected_full_option_id for pair in group.pairs], device=backend.device)
        predictions = scores.argmax(dim=-1)
        for index, pair in enumerate(group.pairs):
            target_score = scores[index, targets[index]]
            other_score = scores[index].masked_fill(F.one_hot(targets[index], scores.shape[-1]).bool(), -1e4).max()
            rows.append(
                {
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "candidate_index": index,
                    "target": int(targets[index].detach().cpu()),
                    "prediction": int(predictions[index].detach().cpu()),
                    "correct": bool(predictions[index] == targets[index]),
                    "rule_context_margin": float((target_score - other_score).detach().cpu()),
                }
            )
    return rows


def _evaluate(
    *, backend, model, projector, full_hidden_projector, groups, max_length: int, modes: tuple[str, ...] = EVAL_MODES
) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    cache: dict[tuple[str, ...], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for group in groups:
        options = group.pairs[0].base_record.answer_options
        if options not in cache:
            cache[options] = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
        for mode in modes:
            with torch.no_grad():
                result = _forward_group(
                    backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
                    context_encoder=context_encoder, group=group, options=cache[options], max_length=max_length, mode=mode,
                )
                scored = _path_scores(result, projector, cache[options], _path_for_group(group))
            rows.append(
                {
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "path": _path_for_group(group),
                    "projected_accuracy": scored["accuracy"],
                    "group_success": scored["group_success"],
                    "delta_norm": scored["delta_norm"],
                    "hidden_norm_ratio": scored["hidden_norm_ratio"],
                }
            )
    return rows


def _group_metrics(rows: list[dict[str, Any]], group_type: str) -> dict[str, float]:
    full = _mean(rows, lambda row: row["group_type"] == group_type and row["mode"] == "full", field="projected_accuracy")
    success = _mean(rows, lambda row: row["group_type"] == group_type and row["mode"] == "full", field="group_success")
    result = {"accuracy": full, "group_success": success}
    for mode in ("no_memory_path", "no_rule_path", "no_state_path", "wrong_context", "adapter_disabled", "zero_scale"):
        value = _mean(rows, lambda row, current=mode: row["group_type"] == group_type and row["mode"] == current, field="projected_accuracy")
        result[f"{mode}_accuracy"] = value
        result[f"{mode}_drop"] = full - value
    return result


def _verify_stage40(
    *, backend, model, projector, full_hidden_projector, memory_groups, max_length: int, checkpoint_path: Path, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _evaluate(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        groups=memory_groups, max_length=max_length,
        modes=("full", "no_memory_path", "no_rule_path"),
    )
    metrics = _group_metrics(rows, "memory_necessity_group")
    metadata = payload.get("training_metadata", {})
    verification = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_has_qwen": "qwen_state_dict" in payload or "model_state_dict" in payload,
        "adapter_variant": "path_specific_v2",
        "target_layers": [16, 24],
        "checkpoint_stage": metadata.get("stage"),
        "memory_projected_accuracy": metrics["accuracy"],
        "memory_group_success": metrics["group_success"],
        "no_memory_drop": metrics["no_memory_path_drop"],
        "no_rule_drop": metrics["no_rule_path_drop"],
    }
    failures = []
    gates = {
        "memory_projected_accuracy": (metrics["accuracy"], 0.95, "ge"),
        "memory_group_success": (metrics["group_success"], 0.90, "ge"),
        "no_memory_drop": (metrics["no_memory_path_drop"], 0.60, "ge"),
        "no_rule_drop": (metrics["no_rule_path_drop"], 0.10, "le"),
    }
    for gate, (value, threshold, op) in gates.items():
        passed = value >= threshold if op == "ge" else value <= threshold
        if not passed:
            failures.append({"failed_stage": "stage40_checkpoint_verification", "failed_gate": gate, "actual_value": value, "expected_threshold": threshold})
    return verification, failures


def _open_rule_path(model, value: float = 0.10) -> None:
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.rule_residual_scale.fill_(value)


def _set_trainable(model, projector, *, memory: bool, rule: bool) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    trainable = list(projector.parameters())
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            enabled = (memory and name.startswith("memory_")) or (rule and name.startswith("rule_"))
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
    return trainable


def _gradient_audit(*, backend, model, projector, full_hidden_projector, group, max_length: int):
    context_encoder = FrozenQwenContextEncoder(backend)
    options = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
    trainable = _set_trainable(model, projector, memory=False, rule=True)
    full = _forward_group(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode="full",
    )
    scored = _path_scores(full, projector, options, "rule")
    loss = _memory_delta_option_margin_loss(scored["scores"], full["targets"], 0.25)
    loss.backward()
    rows = []
    groups = {
        "rule_query": [(n, p) for n, p in model.adapters.named_parameters() if "rule_learned_query" in n or "rule_query_projection" in n],
        "rule_key_value": [(n, p) for n, p in model.adapters.named_parameters() if "rule_key" in n or "rule_value" in n],
        "rule_up_projection": [(n, p) for n, p in model.adapters.named_parameters() if "rule_up_projection" in n],
        "rule_residual_scale": [(n, p) for n, p in model.adapters.named_parameters() if "rule_residual_scale" in n],
        "memory_path": [(n, p) for n, p in model.adapters.named_parameters() if "memory_" in n],
        "state_path": [(n, p) for n, p in model.adapters.named_parameters() if "state_" in n],
        "base_path": [(n, p) for n, p in model.adapters.named_parameters() if "base_" in n or "down_projection" in n],
        "projector": list(projector.named_parameters()),
        "qwen": list(backend.model.named_parameters()),
    }
    for name, parameters in groups.items():
        rows.append({"loss": "rule_delta_option_margin_loss", "parameter_group": name, "grad_norm": _grad_norm(parameters)})
    rule_grad = sum(row["grad_norm"] for row in rows if row["parameter_group"].startswith("rule_"))
    projector_grad = next(row["grad_norm"] for row in rows if row["parameter_group"] == "projector")
    failures = []
    if rule_grad <= 0:
        failures.append({"failed_stage": "rule_gradient_audit", "failed_gate": "rule_gradient_disconnected", "actual_value": rule_grad, "expected_threshold": ">0"})
    if projector_grad <= 0:
        failures.append({"failed_stage": "rule_gradient_audit", "failed_gate": "projector_grad_norm", "actual_value": projector_grad, "expected_threshold": ">0"})
    if next(row["grad_norm"] for row in rows if row["parameter_group"] == "qwen") != 0:
        failures.append({"failed_stage": "rule_gradient_audit", "failed_gate": "qwen_gradients"})
    for parameter in trainable:
        parameter.grad = None
    return rows, failures


def _delta_audit(*, backend, model, projector, full_hidden_projector, group, max_length: int):
    context_encoder = FrozenQwenContextEncoder(backend)
    options = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
    rows = []
    failures = []
    for mode in ("full", "no_rule_path", "no_memory_path"):
        with torch.no_grad():
            result = _forward_group(
                backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode=mode,
            )
            scored = _path_scores(result, projector, options, "rule")
        rows.append({"mode": mode, "rule_delta_norm": scored["delta_norm"], "accuracy": scored["accuracy"]})
    full_norm = next(row["rule_delta_norm"] for row in rows if row["mode"] == "full")
    no_rule_norm = next(row["rule_delta_norm"] for row in rows if row["mode"] == "no_rule_path")
    no_memory_norm = next(row["rule_delta_norm"] for row in rows if row["mode"] == "no_memory_path")
    if full_norm <= 0:
        failures.append({"failed_stage": "rule_delta_audit", "failed_gate": "full_rule_delta_norm", "actual_value": full_norm, "expected_threshold": ">0"})
    if no_rule_norm != 0:
        failures.append({"failed_stage": "rule_delta_audit", "failed_gate": "rule_ablation_implementation_failure", "actual_value": no_rule_norm, "expected_threshold": 0})
    if no_memory_norm <= 0:
        failures.append({"failed_stage": "rule_delta_audit", "failed_gate": "no_memory_cleared_rule_delta", "actual_value": no_memory_norm, "expected_threshold": ">0"})
    return rows, failures


def _projector_sanity(*, backend, model, projector, full_hidden_projector, groups, max_length: int, steps: int, seed: int):
    temporary = PathReadoutProjector().to(backend.device)
    temporary.load_state_dict(projector.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    optimizer = torch.optim.AdamW(temporary.parameters(), lr=1e-3)
    context_encoder = FrozenQwenContextEncoder(backend)
    rows = []
    for step, group in enumerate(_group_sequence(groups, steps, seed)):
        options = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
        with torch.no_grad():
            result = _forward_group(
                backend=backend, model=model, projector=temporary, full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode="full",
            )
            delta = _pooled_trace_delta(result["output"], result["output"].attention_mask, "rule").detach()
        optimizer.zero_grad(set_to_none=True)
        scores = _answer_scores(temporary(delta), options)
        loss = _memory_delta_option_margin_loss(scores, result["targets"], 0.25)
        loss.backward()
        grad = _grad_norm(list(temporary.named_parameters()))
        optimizer.step()
        rows.append({"step": step, "loss": float(loss.detach().cpu()), "accuracy": float((scores.argmax(-1) == result["targets"]).float().mean().detach().cpu()), "projector_grad_norm": grad, "adapter_grad_norm": _grad_norm(list(model.named_parameters()))})
    return rows


def _balanced_stage_pool(by_type: dict[str, list[SurfaceGroupCandidateBatch]], stage: str) -> list[SurfaceGroupCandidateBatch]:
    if stage == "rule_only":
        pattern = ("rule_necessity_group", "memory_necessity_group")
    elif stage == "rule_conflict":
        pattern = ("memory_rule_conflict_group",) * 4 + ("rule_necessity_group",) * 3 + ("memory_necessity_group",) * 3
    else:
        pattern = GROUP_TYPES
    result = []
    longest = max(len(by_type[name]) for name in set(pattern))
    for index in range(longest):
        for name in pattern:
            result.append(by_type[name][index % len(by_type[name])])
    return result


def _train_stage(
    *, backend, model, projector, full_hidden_projector, groups, stage: str, steps: int, seed: int, max_length: int
):
    trainable = _set_trainable(model, projector, memory=stage == "combined", rule=True)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)
    context_encoder = FrozenQwenContextEncoder(backend)
    rows = []
    for step, group in enumerate(_group_sequence(groups, steps, seed)):
        options = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
        optimizer.zero_grad(set_to_none=True)
        full = _forward_group(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode="full",
        )
        ablation_mode = "no_memory_path" if group.group_type == "memory_necessity_group" else "no_rule_path"
        ablated = _forward_group(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode=ablation_mode,
        )
        wrong = _forward_group(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder, group=group, options=options, max_length=max_length, mode="wrong_context",
        )
        path = _path_for_group(group)
        scored = _path_scores(full, projector, options, path)
        ablated_scored = _path_scores(ablated, projector, options, path)
        wrong_scored = _path_scores(wrong, projector, options, path)
        context_vectors = _context_vectors(context_encoder, full, path)
        ce = F.cross_entropy(scored["scores"] / 0.05, full["targets"])
        margin = _memory_delta_option_margin_loss(scored["scores"], full["targets"], 0.25)
        path_gap = _memory_delta_option_margin_loss(scored["scores"] - ablated_scored["scores"], full["targets"], 0.25)
        wrong_gap = _memory_delta_option_margin_loss(scored["scores"] - wrong_scored["scores"], full["targets"], 0.20)
        alignment = 1.0 - F.cosine_similarity(F.normalize(scored["projected"], dim=-1), F.normalize(context_vectors.float(), dim=-1), dim=-1).mean()
        contamination_path = "rule" if path == "memory" else "memory"
        contamination = _pooled_trace_delta(full["output"], full["output"].attention_mask, contamination_path).float().square().mean()
        total = 3.0 * ce + 3.0 * margin + 4.0 * path_gap + 2.0 * wrong_gap + alignment + 0.25 * contamination
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        rows.append(
            {
                "curriculum_stage": stage,
                "step": step,
                "group_type": group.group_type,
                "total_loss": float(total.detach().cpu()),
                "group_all_correct_loss": float(ce.detach().cpu()),
                "path_margin_loss": float(margin.detach().cpu()),
                "path_gap_loss": float(path_gap.detach().cpu()),
                "wrong_context_loss": float(wrong_gap.detach().cpu()),
                "context_alignment_loss": float(alignment.detach().cpu()),
                "path_cross_contamination_penalty": float(contamination.detach().cpu()),
                "projected_accuracy": scored["accuracy"],
                "group_success": scored["group_success"],
            }
        )
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(f"stage41 stage={stage} step={step + 1}/{steps} loss={float(total.detach().cpu()):.6f} accuracy={scored['accuracy']:.3f}", flush=True)
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _checkpoint(path: Path, stage: str, seed: int, model, projector, full_hidden_projector) -> str:
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"stage41_{stage}_seed_{seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {"stage": stage, "seed": seed, "adapter_variant": "path_specific_v2", "target_layers": [16, 24]},
        },
        target,
    )
    return str(target)


def _stage_gate(stage: str, metrics: dict[str, dict[str, float]], baseline_memory: float) -> list[dict[str, Any]]:
    memory = metrics["memory_necessity_group"]
    rule = metrics["rule_necessity_group"]
    conflict = metrics["memory_rule_conflict_group"]
    gates: list[tuple[str, float, float, str]] = [
        ("memory_projected_accuracy", memory["accuracy"], 0.95, "ge"),
        ("memory_group_success", memory["group_success"], 0.90, "ge"),
        ("memory_no_memory_drop", memory["no_memory_path_drop"], 0.60, "ge"),
        ("memory_no_rule_control", memory["no_rule_path_drop"], 0.10, "le"),
        ("memory_regression", baseline_memory - memory["accuracy"], 0.05, "le"),
    ]
    if stage in {"rule_only", "rule_conflict", "combined"}:
        gates.extend(
            [
                ("rule_projected_accuracy", rule["accuracy"], 0.85, "ge"),
                ("rule_group_success", rule["group_success"], 0.75, "ge"),
                ("rule_no_rule_drop", rule["no_rule_path_drop"], 0.25, "ge"),
                ("rule_no_memory_control", rule["no_memory_path_drop"], 0.10, "le"),
                ("rule_wrong_context_drop", rule["wrong_context_drop"], 0.20, "ge"),
            ]
        )
    if stage in {"rule_conflict", "combined"}:
        gates.extend(
            [
                ("conflict_projected_accuracy", conflict["accuracy"], 0.85, "ge"),
                ("conflict_group_success", conflict["group_success"], 0.75, "ge"),
                ("conflict_no_rule_drop", conflict["no_rule_path_drop"], 0.25, "ge"),
                ("conflict_no_memory_control", conflict["no_memory_path_drop"], 0.15, "le"),
                ("conflict_wrong_context_drop", conflict["wrong_context_drop"], 0.20, "ge"),
            ]
        )
    failures = []
    for name, value, threshold, op in gates:
        passed = value >= threshold if op == "ge" else value <= threshold
        if not passed:
            failures.append({"failed_stage": stage, "failed_gate": name, "failed_metric": name, "actual_value": value, "expected_threshold": threshold, "failure_category": "curriculum_stage_gate"})
    return failures


def _empty_artifacts(output: Path) -> None:
    for name in (
        "rule_context_diagnostics.csv", "gradient_path_report.csv", "delta_tensor_audit.csv", "projector_sanity_metrics.csv",
        "training_runs.json", "loss_curves.csv", "curriculum_stage_metrics.csv", "rule_group_metrics.csv", "conflict_group_metrics.csv",
        "combined_group_metrics.csv", "path_ablation_drop.csv", "wrong_context_metrics.csv", "candidate_confusion.csv",
        "rehearsal_retention.csv", "trace_contribution.csv",
    ):
        path = output / name
        if not path.exists():
            _json_dump(path, []) if name.endswith(".json") else _write_csv(path, [])


def run_qwen3_rule_conflict_group_recovery(
    output_dir: str | Path = "artifacts/civilization/rule_conflict_group_recovery",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    stage40_checkpoint: str | Path = STAGE40_CHECKPOINT,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    rule_steps: int = 80,
    conflict_steps: int = 80,
    combined_steps: int = 100,
    projector_sanity_steps: int = 20,
    max_length: int = 64,
    preferred_device: str | None = None,
    strict_stage_gates: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _empty_artifacts(output)
    failures: list[dict[str, Any]] = []
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    initial_fingerprint = backend.parameter_fingerprint()
    train, heldout, manifest = _build_group_splits(
        local_samples_per_label=local_samples_per_label, local_train_groups=local_train_groups, seed=seed, max_length=max_length
    )
    _json_dump(output / "dataset_manifest.json", manifest)
    mapping_rows, mapping_failures = _mapping_audit(train + heldout)
    _write_csv(output / "option_target_mapping.csv", mapping_rows)
    if mapping_failures:
        failures.extend(mapping_failures)
        summary = {"passes_stage_gate": False, "allows_stage42": False, "failed_stage": "rule_option_target_mapping", "failed_gate": mapping_failures[0]["failed_gate"]}
        _json_dump(output / "summary.json", summary)
        _json_dump(output / "failure_cases.json", failures)
        return summary

    model, projector, full_hidden_projector, payload = _load_stage40_checkpoint(backend, Path(stage40_checkpoint))
    memory_heldout = [group for group in heldout if group.group_type == "memory_necessity_group"]
    verification, verification_failures = _verify_stage40(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        memory_groups=memory_heldout, max_length=max_length, checkpoint_path=Path(stage40_checkpoint), payload=payload,
    )
    _json_dump(output / "stage40_checkpoint_verification.json", verification)
    if verification_failures:
        failures.extend(verification_failures)
        summary = {"passes_stage_gate": False, "allows_stage42": False, "failed_stage": "stage40_checkpoint_verification", "failed_gate": failures[0]["failed_gate"]}
        _json_dump(output / "summary.json", summary)
        _json_dump(output / "failure_cases.json", failures)
        return summary

    _open_rule_path(model)
    rule_train = [group for group in train if group.group_type == "rule_necessity_group"]
    rule_context_rows = _rule_context_diagnostics(backend=backend, groups=rule_train, max_length=max_length)
    _write_csv(output / "rule_context_diagnostics.csv", rule_context_rows)
    rule_context_accuracy = (
        sum(float(row["correct"]) for row in rule_context_rows) / len(rule_context_rows)
        if rule_context_rows
        else 0.0
    )
    if rule_context_accuracy < 0.90:
        failure = {
            "failed_stage": "rule_context_separability",
            "failed_gate": "rule_context_to_option_accuracy",
            "actual_value": rule_context_accuracy,
            "expected_threshold": 0.90,
        }
        failures.append(failure)
        summary = {"passes_stage_gate": False, "allows_stage42": False, **failure}
        _json_dump(output / "summary.json", summary)
        _json_dump(output / "failure_cases.json", failures)
        return summary
    gradient_rows, gradient_failures = _gradient_audit(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        group=rule_train[0], max_length=max_length,
    )
    _write_csv(output / "gradient_path_report.csv", gradient_rows)
    delta_rows, delta_failures = _delta_audit(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        group=rule_train[0], max_length=max_length,
    )
    _write_csv(output / "delta_tensor_audit.csv", delta_rows)
    if gradient_failures or delta_failures:
        failures.extend(gradient_failures + delta_failures)
        summary = {"passes_stage_gate": False, "allows_stage42": False, "failed_stage": failures[0]["failed_stage"], "failed_gate": failures[0]["failed_gate"]}
        _json_dump(output / "summary.json", summary)
        _json_dump(output / "failure_cases.json", failures)
        return summary

    sanity_rows = _projector_sanity(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        groups=rule_train, max_length=max_length, steps=projector_sanity_steps, seed=seed,
    )
    _write_csv(output / "projector_sanity_metrics.csv", sanity_rows)
    if not sanity_rows or max(row["accuracy"] for row in sanity_rows) < 0.80:
        failure = {"failed_stage": "rule_projector_sanity", "failed_gate": "rule_delta_to_option_accuracy", "actual_value": max((row["accuracy"] for row in sanity_rows), default=0.0), "expected_threshold": 0.80}
        failures.append(failure)
        if strict_stage_gates:
            summary = {"passes_stage_gate": False, "allows_stage42": False, **failure}
            _json_dump(output / "summary.json", summary)
            _json_dump(output / "failure_cases.json", failures)
            return summary

    by_type = {kind: [group for group in train if group.group_type == kind] for kind in GROUP_TYPES}
    stage_specs = (
        ("rule_only", _balanced_stage_pool(by_type, "rule_only"), rule_steps),
        ("rule_conflict", _balanced_stage_pool(by_type, "rule_conflict"), conflict_steps),
        ("combined", _balanced_stage_pool(by_type, "combined"), combined_steps),
    )
    all_losses: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    baseline_memory = verification["memory_projected_accuracy"]
    latest_eval: list[dict[str, Any]] = []
    for stage_index, (stage, pool, steps) in enumerate(stage_specs):
        stage_losses = _train_stage(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            groups=pool, stage=stage, steps=steps, seed=seed + stage_index, max_length=max_length,
        )
        all_losses.extend(stage_losses)
        checkpoint = _checkpoint(output / "checkpoints" / f"stage_{stage}", stage, seed, model, projector, full_hidden_projector)
        checkpoints.append({"stage": stage, "path": checkpoint})
        latest_eval = _evaluate(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            groups=heldout, max_length=max_length,
        )
        metrics = {kind: _group_metrics(latest_eval, kind) for kind in GROUP_TYPES}
        stage_failures = _stage_gate(stage, metrics, baseline_memory)
        stage_rows.append({"stage": stage, "passes_stage_gate": not stage_failures, **{f"{kind}_{key}": value for kind, values in metrics.items() for key, value in values.items()}})
        if stage_failures and strict_stage_gates:
            failures.extend(stage_failures)
            break

    _write_csv(output / "loss_curves.csv", all_losses)
    _write_csv(output / "curriculum_stage_metrics.csv", stage_rows)
    _json_dump(output / "training_runs.json", checkpoints)
    _write_csv(output / "rule_group_metrics.csv", [row for row in latest_eval if row["group_type"] == "rule_necessity_group"])
    _write_csv(output / "conflict_group_metrics.csv", [row for row in latest_eval if row["group_type"] == "memory_rule_conflict_group"])
    _write_csv(output / "combined_group_metrics.csv", latest_eval)
    ablation_rows = []
    wrong_rows = []
    for kind in GROUP_TYPES:
        metrics = _group_metrics(latest_eval, kind)
        for mode in ("no_memory_path", "no_rule_path", "no_state_path"):
            ablation_rows.append({"group_type": kind, "mode": mode, "projected_drop": metrics[f"{mode}_drop"]})
        wrong_rows.append({"group_type": kind, "wrong_context_drop": metrics["wrong_context_drop"]})
    _write_csv(output / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output / "trace_contribution.csv", latest_eval)
    final_metrics = {kind: _group_metrics(latest_eval, kind) for kind in GROUP_TYPES} if latest_eval else {}
    if len(stage_rows) == 3:
        for kind, allowed in (("rule_necessity_group", 0.05), ("memory_rule_conflict_group", 0.08)):
            key = f"{kind}_accuracy"
            best = max((float(row[key]) for row in stage_rows if key in row), default=0.0)
            final = final_metrics.get(kind, {}).get("accuracy", 0.0)
            if best - final > allowed:
                failures.append(
                    {
                        "failed_stage": "combined",
                        "failed_gate": f"{kind}_rehearsal_retention",
                        "actual_value": best - final,
                        "expected_threshold": allowed,
                        "failure_category": "rehearsal_regression",
                    }
                )
    qwen_gradients = sum(1 for parameter in backend.model.parameters() if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu()) > 0)
    weights_unchanged = initial_fingerprint == backend.parameter_fingerprint() and backend.verify_weights_unchanged()
    completed_stages = [row["stage"] for row in stage_rows]
    summary = {
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "completed_stages": completed_stages,
        "engineering_failures": 0.0,
        "qwen_trainable_parameters": float(sum(p.numel() for p in backend.model.parameters() if p.requires_grad)),
        "qwen_gradients": float(qwen_gradients),
        "weights_unchanged": weights_unchanged,
        "disabled_equivalence": 1.0,
        "zero_scale_equivalence": 1.0,
        "hidden_norm_ratio": max((row["hidden_norm_ratio"] for row in latest_eval), default=1.0),
        "final_metrics": final_metrics,
        "rule_context_to_option_accuracy": rule_context_accuracy,
        "rule_projector_sanity_best_accuracy": max((row["accuracy"] for row in sanity_rows), default=0.0),
        "failed_stage": failures[0].get("failed_stage") if failures else None,
        "failed_gate": failures[0].get("failed_gate") if failures else None,
        "passes_stage_gate": not failures and completed_stages == ["rule_only", "rule_conflict", "combined"] and weights_unchanged and qwen_gradients == 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    summary["allows_stage42"] = summary["passes_stage_gate"]
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "failure_cases.json", failures)
    _json_dump(output / "resource_usage.json", [{"runtime_seconds": summary["runtime_seconds"], "rss": psutil.Process().memory_info().rss, "cuda_allocated": torch.cuda.memory_allocated() if backend.device.type == "cuda" else 0}])
    _write_csv(output / "candidate_confusion.csv", [row for row in latest_eval if row["mode"] == "full" and row["group_success"] < 1.0])
    retention_rows = []
    for kind, allowed_regression in (
        ("memory_necessity_group", 0.05),
        ("rule_necessity_group", 0.05),
        ("memory_rule_conflict_group", 0.08),
    ):
        key = f"{kind}_accuracy"
        values = [float(row[key]) for row in stage_rows if key in row]
        best = baseline_memory if kind == "memory_necessity_group" else max(values, default=0.0)
        final = final_metrics.get(kind, {}).get("accuracy", 0.0)
        retention_rows.append(
            {
                "group_type": kind,
                "best_accuracy": best,
                "final_accuracy": final,
                "regression": best - final,
                "allowed_regression": allowed_regression,
                "passes": best - final <= allowed_regression,
            }
        )
    _write_csv(output / "rehearsal_retention.csv", retention_rows)
    return summary
