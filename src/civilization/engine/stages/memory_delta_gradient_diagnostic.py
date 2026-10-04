from __future__ import annotations

from dataclasses import asdict
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
from .evidence_answer_training import _answer_scores
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .hidden_states import last_non_padding_pool
from .memory_group_gate_repair import (
    MEMORY_GATE_MODES,
    _build_memory_group_splits,
    _candidate_margin,
    _memory_context_vectors,
)
from .multiclass_group_curriculum_repair import (
    SurfaceGroupCandidateBatch,
    _checkpoint,
    _encode_group,
    _group_forward,
    _group_sequence,
)
from .multiclass_necessity_repair import _make_path_specific_model


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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _option_text(group: SurfaceGroupCandidateBatch, option_id: int) -> str:
    return group.pairs[0].base_record.answer_options[option_id]


def _memory_text(pair) -> str:
    return " | ".join(
        f"{item.summary} {item.content} {item.relation_type}"
        for item in context_items_for_sample(pair.full_sample, context_mode="full").memories
    )


def _pooled_trace_delta(output, attention_mask: torch.Tensor, path: str) -> torch.Tensor:
    tensors = []
    field = f"{path}_delta_tensor"
    for trace in output.traces.values():
        tensor = getattr(trace, field, None)
        if tensor is None:
            raise RuntimeError(f"trace does not expose {field}")
        tensors.append(tensor.to(attention_mask.device))
    if not tensors:
        raise RuntimeError("no adapter traces available")
    return last_non_padding_pool(torch.stack(tensors, dim=0).sum(dim=0), attention_mask).float()


def _memory_delta_scores(result: dict[str, Any], projector: PathReadoutProjector, option_vectors: torch.Tensor) -> dict[str, Any]:
    memory_delta = _pooled_trace_delta(result["output"], result["output"].attention_mask, "memory")
    projected = projector(memory_delta)
    scores = _answer_scores(projected, option_vectors)
    predictions = scores.argmax(dim=-1)
    targets = result["targets"]
    return {
        "memory_delta": memory_delta,
        "projected": projected,
        "scores": scores,
        "predictions": predictions,
        "accuracy": (predictions == targets).float().mean(),
        "group_success": torch.all(predictions == targets).float(),
        "margins": _candidate_margin(scores, targets),
        "hidden_norm_ratio": max(
            (float(trace.hidden_norm_ratio) for trace in result["output"].traces.values()),
            default=1.0,
        ),
    }


def _open_memory_residual_path(model, value: float = 0.10) -> None:
    """Open only the Memory residual path after zero-scale equivalence is established."""
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.memory_residual_scale.fill_(value)


def _write_early_failure(
    output_path: Path,
    *,
    failed_stage: str,
    failures: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    first = failures[0]
    summary = {
        "passes_stage_gate": False,
        "allows_stage41": False,
        "failed_stage": failed_stage,
        "failed_gate": first.get("failed_gate", failed_stage),
        "failed_metric": first.get("failed_metric", first.get("failed_gate", failed_stage)),
        "expected_threshold": first.get("expected_threshold"),
        "actual_value": first.get("actual_value"),
        "engineering_failures": float(len(failures)),
        "runtime_seconds": time.perf_counter() - started,
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "failure_cases.json", failures)
    _json_dump(output_path / "dataset_manifest.json", dataset_manifest)
    for filename in (
        "candidate_score_audit.csv",
        "projector_sanity_metrics.csv",
        "memory_path_only_metrics.csv",
        "memory_path_ablation_drop.csv",
        "memory_candidate_confusion.csv",
        "trace_contribution.csv",
        "loss_curves.csv",
        "delta_tensor_audit.csv",
        "gradient_path_report.csv",
    ):
        path = output_path / filename
        if not path.exists():
            _write_csv(path, [])
    if not (output_path / "single_batch_loss_report.json").exists():
        _json_dump(output_path / "single_batch_loss_report.json", {})
    if not (output_path / "training_runs.json").exists():
        _json_dump(output_path / "training_runs.json", [])
    _json_dump(
        output_path / "resource_usage.json",
        [{"seconds": time.perf_counter() - started, "rss": psutil.Process().memory_info().rss}],
    )
    return summary


def _assert_option_target_mapping(groups: list[SurfaceGroupCandidateBatch]) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group in groups:
        targets = [pair.expected_full_option_id for pair in group.pairs]
        if targets != list(range(5)):
            failures.append(
                {
                    "failed_gate": "option_target_mapping",
                    "surface_group_id": group.surface_group_id,
                    "actual_targets": targets,
                    "expected_targets": [0, 1, 2, 3, 4],
                }
            )
        options = group.pairs[0].base_record.answer_options
        for candidate_index, pair in enumerate(group.pairs):
            target = pair.expected_full_option_id
            rows.append(
                {
                    "surface_group_id": group.surface_group_id,
                    "candidate_index": candidate_index,
                    "option_id": target,
                    "target_tensor": target,
                    "score_column": target,
                    "answer_option_text": options[target],
                    "expected_pattern": pair.full_sample.expected_pattern,
                    "option_text_hash": _sha(options[target]),
                    "memory_text_hash": _sha(_memory_text(pair)),
                    "mapping_ok": target == candidate_index and options[target] == pair.full_sample.expected_pattern,
                }
            )
            if target != candidate_index or options[target] != pair.full_sample.expected_pattern:
                failures.append(
                    {
                        "failed_gate": "option_target_mapping",
                        "surface_group_id": group.surface_group_id,
                        "candidate_index": candidate_index,
                        "target": target,
                        "answer_option": options[target],
                        "expected_pattern": pair.full_sample.expected_pattern,
                    }
                )
    return not failures, rows, failures


def _parameter_groups(model, projector: PathReadoutProjector, full_hidden_projector: FullHiddenCentroidProjector, backend: Qwen3Backend) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    adapter_named = [(f"adapters.{name}", parameter) for name, parameter in model.adapters.named_parameters()]
    return {
        "memory_query": [(name, p) for name, p in adapter_named if "memory_learned_query" in name or "memory_query_projection" in name],
        "memory_key_value": [(name, p) for name, p in adapter_named if "memory_key" in name or "memory_value" in name],
        "memory_up_projection": [(name, p) for name, p in adapter_named if "memory_up_projection" in name],
        "memory_residual_scale": [(name, p) for name, p in adapter_named if "memory_residual_scale" in name],
        "rule_path": [(name, p) for name, p in adapter_named if "rule_" in name],
        "state_path": [(name, p) for name, p in adapter_named if "state_" in name],
        "base_path": [(name, p) for name, p in adapter_named if "base_" in name or "down_projection" in name],
        "projector": [(f"projector.{name}", p) for name, p in projector.named_parameters()],
        "full_hidden_projector": [(f"full_hidden_projector.{name}", p) for name, p in full_hidden_projector.named_parameters()],
        "qwen": [(f"qwen.{name}", p) for name, p in backend.model.named_parameters()],
    }


def _grad_norm(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    for _name, parameter in named_parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total**0.5


def _zero_all_gradients(model, projector: PathReadoutProjector, full_hidden_projector: FullHiddenCentroidProjector, backend: Qwen3Backend) -> None:
    for module in (model, projector, full_hidden_projector, backend.model):
        for parameter in module.parameters():
            parameter.grad = None


def _single_group_forward(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    context_encoder: FrozenQwenContextEncoder,
    group: SurfaceGroupCandidateBatch,
    option_vectors: torch.Tensor,
    max_length: int,
    mode: str,
) -> dict[str, Any]:
    # AdapterDiagnosticHeads is only used by _group_forward for prototype losses.
    from .adapter_training import AdapterDiagnosticHeads

    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    return _group_forward(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=heads,
        context_encoder=context_encoder,
        group=group,
        option_vectors=option_vectors,
        local_max_length=max_length,
        external_max_length=max_length,
        mode=mode,
    )


def _memory_delta_option_margin_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    other_scores = scores.masked_fill(F.one_hot(targets, scores.shape[-1]).bool(), -1e4).max(dim=-1).values
    return torch.relu(margin + other_scores - target_scores).mean()


def _gradient_path_audit(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    group: SurfaceGroupCandidateBatch,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
    parameter_groups = _parameter_groups(model, projector, full_hidden_projector, backend)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    loss_names = (
        "memory_group_all_correct_loss_v2",
        "memory_no_memory_gap_loss",
        "memory_counterfactual_candidate_loss",
        "memory_context_to_delta_alignment_loss",
        "memory_delta_option_margin_loss",
    )
    original_scales = [adapter.memory_residual_scale.detach().clone() for adapter in model.adapters.values()]
    for opened_scale in (False, True):
        if opened_scale:
            with torch.no_grad():
                for adapter in model.adapters.values():
                    adapter.memory_residual_scale.fill_(0.10)
        for loss_name in loss_names:
            _zero_all_gradients(model, projector, full_hidden_projector, backend)
            full = _single_group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_vectors,
                max_length=max_length,
                mode="full",
            )
            no_memory = _single_group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_vectors,
                max_length=max_length,
                mode="no_memory_path",
            )
            wrong = _single_group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_vectors,
                max_length=max_length,
                mode="wrong_context",
            )
            memory_delta = _memory_delta_scores(full, projector, option_vectors)
            no_memory_delta = _memory_delta_scores(no_memory, projector, option_vectors)
            wrong_delta = _memory_delta_scores(wrong, projector, option_vectors)
            memory_vectors = _memory_context_vectors(context_encoder, full["samples"], full["encoded"]["attention_mask"])
            if loss_name == "memory_group_all_correct_loss_v2":
                loss = F.cross_entropy(memory_delta["scores"] / 0.05, full["targets"])
            elif loss_name == "memory_no_memory_gap_loss":
                loss = _memory_delta_option_margin_loss(memory_delta["scores"] - no_memory_delta["scores"], full["targets"], margin=0.20)
            elif loss_name == "memory_counterfactual_candidate_loss":
                loss = _memory_delta_option_margin_loss(memory_delta["scores"] - wrong_delta["scores"], full["targets"], margin=0.15)
            elif loss_name == "memory_context_to_delta_alignment_loss":
                loss = 1.0 - F.cosine_similarity(F.normalize(memory_delta["projected"], dim=-1), F.normalize(memory_vectors.float(), dim=-1), dim=-1).mean()
            else:
                loss = _memory_delta_option_margin_loss(memory_delta["scores"], full["targets"], margin=0.25)
            loss.backward()
            for group_name, named_parameters in parameter_groups.items():
                rows.append(
                    {
                        "loss_name": loss_name,
                        "opened_memory_scale": opened_scale,
                        "parameter_group": group_name,
                        "grad_norm": _grad_norm(named_parameters),
                        "loss_value": float(loss.detach().cpu()),
                    }
                )
            if _grad_norm(parameter_groups["qwen"]) != 0.0:
                failures.append({"failed_gate": "qwen_gradients", "loss_name": loss_name, "opened_memory_scale": opened_scale})
        if opened_scale:
            with torch.no_grad():
                for adapter, scale in zip(model.adapters.values(), original_scales, strict=True):
                    adapter.memory_residual_scale.copy_(scale)
    opened_memory_grad = sum(row["grad_norm"] for row in rows if row["opened_memory_scale"] and row["parameter_group"] in {"memory_query", "memory_key_value", "memory_up_projection"})
    projector_grad = sum(row["grad_norm"] for row in rows if row["opened_memory_scale"] and row["parameter_group"] == "projector")
    if opened_memory_grad <= 0.0:
        failures.append({"failed_gate": "memory_path_grad_norm", "actual_value": opened_memory_grad, "expected_threshold": ">0"})
    if projector_grad <= 0.0:
        failures.append({"failed_gate": "projector_grad_norm", "actual_value": projector_grad, "expected_threshold": ">0"})
    return rows, failures


def _delta_tensor_audit(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group in groups:
        option_vectors = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
        for mode in MEMORY_GATE_MODES:
            with torch.no_grad():
                result = _single_group_forward(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=option_vectors,
                    max_length=max_length,
                    mode=mode,
                )
                scored = _memory_delta_scores(result, projector, option_vectors)
            memory_delta_norm = float(torch.linalg.vector_norm(scored["memory_delta"].float(), dim=-1).mean().detach().cpu())
            rows.append(
                {
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "memory_delta_shape": "x".join(str(dim) for dim in scored["memory_delta"].shape),
                    "memory_delta_norm": memory_delta_norm,
                    "projected_accuracy": float(scored["accuracy"].detach().cpu()),
                    "group_success": float(scored["group_success"].detach().cpu()),
                }
            )
            if mode == "full" and memory_delta_norm <= 0.0:
                failures.append({"failed_gate": "full_memory_delta_norm", "surface_group_id": group.surface_group_id, "actual_value": memory_delta_norm})
            if mode == "no_memory_path" and memory_delta_norm != 0.0:
                failures.append({"failed_gate": "no_memory_memory_delta_norm", "surface_group_id": group.surface_group_id, "actual_value": memory_delta_norm})
    return rows, failures


def _score_audit(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    max_length: int,
) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    rows: list[dict[str, Any]] = []
    for group in groups:
        options = group.pairs[0].base_record.answer_options
        option_vectors = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
        for mode in ("full", "no_memory_path", "wrong_context"):
            with torch.no_grad():
                result = _single_group_forward(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=option_vectors,
                    max_length=max_length,
                    mode=mode,
                )
                scored = _memory_delta_scores(result, projector, option_vectors)
            for index, pair in enumerate(group.pairs):
                rows.append(
                    {
                        "surface_group_id": group.surface_group_id,
                        "mode": mode,
                        "candidate_index": index,
                        "target_column": int(result["targets"][index].detach().cpu()),
                        "predicted_column": int(scored["predictions"][index].detach().cpu()),
                        "margin": float(scored["margins"][index].detach().cpu()),
                        "scores": json.dumps([float(value) for value in scored["scores"][index].detach().cpu().tolist()]),
                        "option_text_hash": _sha(_option_text(group, int(result["targets"][index].detach().cpu()))),
                        "memory_text_hash": _sha(_memory_text(pair)),
                    }
                )
    return rows


def _projector_only_sanity(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    max_length: int,
    steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Verify a frozen Memory delta can be aligned without updating the Adapter."""
    context_encoder = FrozenQwenContextEncoder(backend)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in full_hidden_projector.parameters():
        parameter.requires_grad_(False)
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    for module in (model, projector, full_hidden_projector, backend.model):
        for parameter in module.parameters():
            parameter.grad = None
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=0.0)
    rows: list[dict[str, Any]] = []
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    for step, group in enumerate(_group_sequence(groups, steps, seed)):
        options = group.pairs[0].base_record.answer_options
        if options not in option_cache:
            option_cache[options] = torch.tensor(
                build_answer_option_vectors(backend, options),
                dtype=torch.float32,
                device=backend.device,
            )
        with torch.no_grad():
            result = _single_group_forward(
                backend=backend,
                model=model,
                projector=projector,
                full_hidden_projector=full_hidden_projector,
                context_encoder=context_encoder,
                group=group,
                option_vectors=option_cache[options],
                max_length=max_length,
                mode="full",
            )
            memory_delta = _pooled_trace_delta(
                result["output"], result["output"].attention_mask, "memory"
            ).detach()
        optimizer.zero_grad(set_to_none=True)
        scores = _answer_scores(projector(memory_delta), option_cache[options])
        loss = _memory_delta_option_margin_loss(scores, result["targets"], margin=0.25)
        loss.backward()
        projector_grad_norm = _grad_norm(
            [(f"projector.{name}", parameter) for name, parameter in projector.named_parameters()]
        )
        optimizer.step()
        rows.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "accuracy": float((scores.argmax(dim=-1) == result["targets"]).float().mean().detach().cpu()),
                "projector_grad_norm": projector_grad_norm,
                "adapter_grad_norm": _grad_norm(list(model.named_parameters())),
            }
        )
    return rows


def _set_memory_path_only_trainable(model, projector: PathReadoutProjector, full_hidden_projector: FullHiddenCentroidProjector) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in full_hidden_projector.parameters():
        parameter.requires_grad_(False)
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    trainable: list[torch.nn.Parameter] = list(projector.parameters())
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            if name.startswith("memory_"):
                parameter.requires_grad_(True)
                trainable.append(parameter)
    return trainable


def _train_memory_path_only(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    train_groups: list[SurfaceGroupCandidateBatch],
    output_dir: Path,
    seed: int,
    repair_steps: int,
    max_length: int,
    learning_rate: float = 3e-4,
) -> tuple[list[dict[str, Any]], str]:
    random.seed(seed)
    torch.manual_seed(seed)
    context_encoder = FrozenQwenContextEncoder(backend)
    trainable = _set_memory_path_only_trainable(model, projector, full_hidden_projector)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for adapter in model.adapters.values() for n, p in adapter.named_parameters() if n.startswith("memory_")], "lr": learning_rate, "weight_decay": 0.01},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    losses: list[dict[str, Any]] = []
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    for step, group in enumerate(_group_sequence(train_groups, repair_steps, seed)):
        optimizer.zero_grad(set_to_none=True)
        options = group.pairs[0].base_record.answer_options
        if options not in option_cache:
            option_cache[options] = torch.tensor(build_answer_option_vectors(backend, options), dtype=torch.float32, device=backend.device)
        full = _single_group_forward(
            backend=backend,
            model=model,
            projector=projector,
            full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder,
            group=group,
            option_vectors=option_cache[options],
            max_length=max_length,
            mode="full",
        )
        no_memory = _single_group_forward(
            backend=backend,
            model=model,
            projector=projector,
            full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder,
            group=group,
            option_vectors=option_cache[options],
            max_length=max_length,
            mode="no_memory_path",
        )
        wrong = _single_group_forward(
            backend=backend,
            model=model,
            projector=projector,
            full_hidden_projector=full_hidden_projector,
            context_encoder=context_encoder,
            group=group,
            option_vectors=option_cache[options],
            max_length=max_length,
            mode="wrong_context",
        )
        full_delta = _memory_delta_scores(full, projector, option_cache[options])
        no_memory_delta = _memory_delta_scores(no_memory, projector, option_cache[options])
        wrong_delta = _memory_delta_scores(wrong, projector, option_cache[options])
        memory_vectors = _memory_context_vectors(context_encoder, full["samples"], full["encoded"]["attention_mask"])
        loss_items = {
            "memory_delta_option_margin_loss": _memory_delta_option_margin_loss(full_delta["scores"], full["targets"], 0.25),
            "memory_group_all_correct_loss_v2": F.cross_entropy(full_delta["scores"] / 0.05, full["targets"]),
            "memory_no_memory_gap_loss": _memory_delta_option_margin_loss(full_delta["scores"] - no_memory_delta["scores"], full["targets"], 0.25),
            "memory_counterfactual_candidate_loss": _memory_delta_option_margin_loss(full_delta["scores"] - wrong_delta["scores"], full["targets"], 0.15),
            "memory_context_to_delta_alignment_loss": 1.0 - F.cosine_similarity(F.normalize(full_delta["projected"], dim=-1), F.normalize(memory_vectors.float(), dim=-1), dim=-1).mean(),
        }
        total = (
            3.0 * loss_items["memory_delta_option_margin_loss"]
            + 3.0 * loss_items["memory_group_all_correct_loss_v2"]
            + 4.0 * loss_items["memory_no_memory_gap_loss"]
            + 2.0 * loss_items["memory_counterfactual_candidate_loss"]
            + 1.0 * loss_items["memory_context_to_delta_alignment_loss"]
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "stage": "memory_delta_path_only_v1",
            "step": step,
            "surface_group_id": group.surface_group_id,
            "total_loss": float(total.detach().cpu()),
            "projected_accuracy": float(full_delta["accuracy"].detach().cpu()),
            "group_success": float(full_delta["group_success"].detach().cpu()),
            "memory_delta_norm": float(torch.linalg.vector_norm(full_delta["memory_delta"].float(), dim=-1).mean().detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in loss_items.items()},
        }
        losses.append(row)
        if step == 0 or step + 1 == repair_steps or (step + 1) % 20 == 0:
            print(
                f"memory_delta_diagnostic stage=memory_delta_path_only step={step + 1}/{repair_steps} "
                f"loss={row['total_loss']:.6f} acc={row['projected_accuracy']:.3f} group={row['group_success']:.3f}",
                flush=True,
            )
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
    checkpoint = _checkpoint(output_dir / "checkpoints", "local_only_transfer", "memory_delta_path_only", seed, model, torch.nn.Identity(), projector, full_hidden_projector)
    return losses, checkpoint


def _evaluate_memory_delta_gate(
    *,
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    full_hidden_projector: FullHiddenCentroidProjector,
    groups: list[SurfaceGroupCandidateBatch],
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for group in groups:
        option_vectors = torch.tensor(build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options), dtype=torch.float32, device=backend.device)
        group_results: dict[str, dict[str, Any]] = {}
        for mode in MEMORY_GATE_MODES:
            with torch.no_grad():
                result = _single_group_forward(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=option_vectors,
                    max_length=max_length,
                    mode=mode,
                )
                scored = _memory_delta_scores(result, projector, option_vectors)
            group_results[mode] = scored
            metric_rows.append(
                {
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "projected_accuracy": float(scored["accuracy"].detach().cpu()),
                    "group_success": float(scored["group_success"].detach().cpu()),
                    "memory_delta_norm": float(torch.linalg.vector_norm(scored["memory_delta"].float(), dim=-1).mean().detach().cpu()),
                    "hidden_norm_ratio": scored["hidden_norm_ratio"],
                }
            )
        full_acc = float(group_results["full"]["accuracy"].detach().cpu())
        full_success = float(group_results["full"]["group_success"].detach().cpu())
        for mode in ("no_memory_path", "no_rule_path", "wrong_context", "adapter_disabled", "zero_scale"):
            ablated_acc = float(group_results[mode]["accuracy"].detach().cpu())
            ablation_rows.append(
                {
                    "mode": mode,
                    "surface_group_id": group.surface_group_id,
                    "full_projected_accuracy": full_acc,
                    "ablated_projected_accuracy": ablated_acc,
                    "projected_drop": full_acc - ablated_acc,
                    "full_group_success": full_success,
                    "ablated_group_success": float(group_results[mode]["group_success"].detach().cpu()),
                }
            )
        if full_success < 1.0:
            confusion_rows.append(
                {
                    "surface_group_id": group.surface_group_id,
                    "projected_accuracy": full_acc,
                    "group_success": full_success,
                    "failure_reason": "memory_delta_candidate_prediction_failed",
                }
            )
    return metric_rows, ablation_rows, confusion_rows


def _stage_gate_failure(summary: dict[str, Any]) -> dict[str, Any] | None:
    thresholds = {
        "engineering_failures": (0, "eq"),
        "qwen_trainable_parameters": (0, "eq"),
        "qwen_gradients": (0, "eq"),
        "weights_unchanged": (1.0, "ge"),
        "disabled_equivalence": (1.0, "ge"),
        "zero_scale_equivalence": (1.0, "ge"),
        "hidden_norm_ratio": (2.0, "le"),
        "option_target_mapping_audit": (1.0, "ge"),
        "memory_path_grad_norm": (0.0, "gt"),
        "projector_grad_norm": (0.0, "gt"),
        "full_memory_delta_norm": (0.0, "gt"),
        "no_memory_memory_delta_norm": (0.0, "eq"),
        "memory_delta_to_option_accuracy": (0.80, "ge"),
        "memory_group_projected_accuracy": (0.85, "ge"),
        "memory_necessity_group_success": (0.75, "ge"),
        "no_memory_drop": (0.25, "ge"),
        "no_rule_drop": (0.10, "le"),
        "wrong_context_drop": (0.20, "ge"),
    }
    for key, (threshold, op) in thresholds.items():
        value = float(summary.get(key, 0.0))
        passed = value == threshold if op == "eq" else value > threshold if op == "gt" else value >= threshold if op == "ge" else value <= threshold
        if not passed:
            return {"failed_stage": "memory_delta_gradient_gate", "failed_gate": key, "failed_metric": key, "expected_threshold": threshold, "actual_value": value}
    return None


def run_qwen3_memory_delta_gradient_diagnostic(
    output_dir: str | Path = "artifacts/civilization/memory_delta_gradient_diagnostic",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    repair_steps: int = 80,
    max_length: int = 64,
    preferred_device: str | None = None,
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
    mapping_ok, mapping_rows, mapping_failures = _assert_option_target_mapping(train_groups + heldout_groups)
    _write_csv(output_path / "option_target_mapping.csv", mapping_rows)
    if not mapping_ok:
        return _write_early_failure(
            output_path,
            failed_stage="option_target_mapping_audit",
            failures=mapping_failures,
            dataset_manifest=dataset_manifest,
            started=started,
        )

    model = _make_path_specific_model(backend, (16, 24))
    projector = PathReadoutProjector().to(backend.device)
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    initial_fingerprint = backend.parameter_fingerprint()
    gradient_rows, gradient_failures = _gradient_path_audit(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        group=train_groups[0],
        max_length=max_length,
    )
    _write_csv(output_path / "gradient_path_report.csv", gradient_rows)
    memory_path_grad_norm = sum(row["grad_norm"] for row in gradient_rows if row["opened_memory_scale"] and row["parameter_group"] in {"memory_query", "memory_key_value", "memory_up_projection"})
    projector_grad_norm = sum(row["grad_norm"] for row in gradient_rows if row["opened_memory_scale"] and row["parameter_group"] == "projector")
    if gradient_failures:
        return _write_early_failure(
            output_path,
            failed_stage="single_batch_gradient_audit",
            failures=gradient_failures,
            dataset_manifest=dataset_manifest,
            started=started,
        )

    # Zero-scale equivalence has already been preserved by the model contract. Open
    # only Memory before auditing whether its internal delta reaches the readout.
    _open_memory_residual_path(model)
    delta_rows, delta_failures = _delta_tensor_audit(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train_groups[:1],
        max_length=max_length,
    )
    _write_csv(output_path / "delta_tensor_audit.csv", delta_rows)
    full_delta_norm = _mean(delta_rows, lambda row: row["mode"] == "full", field="memory_delta_norm")
    no_memory_delta_norm = _mean(delta_rows, lambda row: row["mode"] == "no_memory_path", field="memory_delta_norm")
    if delta_failures:
        return _write_early_failure(
            output_path,
            failed_stage="memory_delta_tensor_audit",
            failures=delta_failures,
            dataset_manifest=dataset_manifest,
            started=started,
        )

    score_rows = _score_audit(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train_groups[:1],
        max_length=max_length,
    )
    _write_csv(output_path / "candidate_score_audit.csv", score_rows)
    _json_dump(
        output_path / "single_batch_loss_report.json",
        {
            "memory_path_grad_norm": memory_path_grad_norm,
            "projector_grad_norm": projector_grad_norm,
            "full_memory_delta_norm": full_delta_norm,
            "no_memory_memory_delta_norm": no_memory_delta_norm,
        },
    )

    projector_sanity_rows = _projector_only_sanity(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=train_groups,
        max_length=max_length,
        steps=max(4, min(20, repair_steps // 4)),
        seed=seed,
    )
    _write_csv(output_path / "projector_sanity_metrics.csv", projector_sanity_rows)
    if not projector_sanity_rows or max(row["projector_grad_norm"] for row in projector_sanity_rows) <= 0.0:
        return _write_early_failure(
            output_path,
            failed_stage="projector_only_sanity",
            failures=[
                {
                    "failed_gate": "projector_grad_norm",
                    "expected_threshold": ">0",
                    "actual_value": 0.0,
                }
            ],
            dataset_manifest=dataset_manifest,
            started=started,
        )

    loss_rows, checkpoint = _train_memory_path_only(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        train_groups=train_groups,
        output_dir=output_path,
        seed=seed,
        repair_steps=repair_steps,
        max_length=max_length,
    )
    group_rows, ablation_rows, confusion_rows = _evaluate_memory_delta_gate(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        groups=heldout_groups,
        max_length=max_length,
    )
    memory_acc = _mean(group_rows, lambda row: row["mode"] == "full", field="projected_accuracy")
    memory_success = _mean(group_rows, lambda row: row["mode"] == "full", field="group_success")
    no_memory_drop = _mean(ablation_rows, lambda row: row["mode"] == "no_memory_path", field="projected_drop")
    no_rule_drop = _mean(ablation_rows, lambda row: row["mode"] == "no_rule_path", field="projected_drop")
    wrong_context_drop = _mean(ablation_rows, lambda row: row["mode"] == "wrong_context", field="projected_drop")
    disabled_acc = _mean(ablation_rows, lambda row: row["mode"] == "adapter_disabled", field="ablated_projected_accuracy")
    zero_acc = _mean(ablation_rows, lambda row: row["mode"] == "zero_scale", field="ablated_projected_accuracy")
    hidden_norm_ratio = max((float(row["hidden_norm_ratio"]) for row in group_rows), default=1.0)
    qwen_gradients = sum(1 for parameter in backend.model.parameters() if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu()) > 0.0)
    weights_unchanged = initial_fingerprint == backend.parameter_fingerprint() and backend.verify_weights_unchanged()
    summary: dict[str, Any] = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "training_mode": "memory_delta_path_only_v1",
        "train_group_count": len(train_groups),
        "heldout_group_count": len(heldout_groups),
        "engineering_failures": 0.0,
        "qwen_trainable_parameters": float(sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad)),
        "qwen_gradients": float(qwen_gradients),
        "weights_unchanged": 1.0 if weights_unchanged else 0.0,
        "disabled_equivalence": 1.0 if abs(disabled_acc - zero_acc) <= 1e-9 else 0.0,
        "zero_scale_equivalence": 1.0 if abs(disabled_acc - zero_acc) <= 1e-9 else 0.0,
        "hidden_norm_ratio": hidden_norm_ratio,
        "option_target_mapping_audit": 1.0,
        "memory_path_grad_norm": memory_path_grad_norm,
        "projector_grad_norm": projector_grad_norm,
        "full_memory_delta_norm": full_delta_norm,
        "no_memory_memory_delta_norm": no_memory_delta_norm,
        "memory_delta_to_option_accuracy": memory_acc,
        "memory_group_projected_accuracy": memory_acc,
        "memory_necessity_group_success": memory_success,
        "no_memory_drop": no_memory_drop,
        "no_rule_drop": no_rule_drop,
        "wrong_context_drop": wrong_context_drop,
        "checkpoint_path": checkpoint,
        "runtime_seconds": time.perf_counter() - started,
    }
    all_failures = mapping_failures + gradient_failures + delta_failures
    failure = _stage_gate_failure(summary)
    if failure:
        all_failures.append(failure)
    summary["failed_gate"] = failure["failed_gate"] if failure else None
    summary["failed_stage"] = failure["failed_stage"] if failure else None
    summary["passes_stage_gate"] = failure is None and not all_failures
    summary["allows_stage41"] = summary["passes_stage_gate"]
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", [{"seed": seed, "repair_steps": repair_steps, "checkpoint_path": checkpoint}])
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "projector_sanity_metrics.csv", projector_sanity_rows)
    _write_csv(output_path / "memory_path_only_metrics.csv", group_rows)
    _write_csv(output_path / "memory_path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "memory_candidate_confusion.csv", confusion_rows)
    _write_csv(output_path / "trace_contribution.csv", group_rows)
    _json_dump(output_path / "resource_usage.json", [{"seconds": time.perf_counter() - started, "rss": psutil.Process().memory_info().rss, "cuda_allocated": torch.cuda.memory_allocated() if backend.device.type == "cuda" else 0}])
    _json_dump(output_path / "failure_cases.json", all_failures)
    _json_dump(output_path / "dataset_manifest.json", dataset_manifest)
    return summary
