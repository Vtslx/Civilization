from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable

import psutil
import torch
import torch.nn.functional as F

from civilization.research.torch_line.analysis.dataset import LOGIC_LABELS, LogicSample

from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .adapter_training import AdapterDiagnosticHeads
from .answer_option_readout import build_answer_option_vectors
from .evidence_answer_data import (
    EvidenceAnswerSample,
    LOCAL_ANSWER_OPTIONS,
    assert_no_logic_label_leakage,
    build_evidence_answer_samples,
)
from .evidence_answer_training import _answer_scores
from .group_full_hidden_centroid_integration import (
    RAW_FULL_HIDDEN_RESIDUAL_SCALE,
    _load_stage41_checkpoint,
)
from .hidden_states import last_non_padding_pool
from .multiclass_group_curriculum_repair import _group_forward
from .multiclass_necessity_repair import _context, _dynamic_centroid_loss
from .real_task_data import (
    LOCAL_REAL_TASK_TYPES,
    RealTaskRecord,
    build_local_semireal_task_records,
    local_real_task_manifest,
    records_to_logic_datasets,
)
from .rule_conflict_group_recovery import (
    _build_group_splits as _build_generic_group_splits,
    _evaluate as _evaluate_generic_groups,
    _group_metrics as _generic_group_metrics,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44a_local_multiclass_integration"
)
DEFAULT_STAGE43_ROOT = Path(
    "artifacts/civilization/stage43_residual_scale_stress"
)
TASK_PATH = {
    "operation_decision": "memory",
    "causal_trace": "memory",
    "priority_selection": "rule",
    "rule_conflict": "rule",
    "condition_check": "state",
    "negation_constraint": "state",
}
TASK_STAGE = {
    "operation_decision": "memory_transfer",
    "causal_trace": "memory_transfer",
    "priority_selection": "rule_transfer",
    "rule_conflict": "conflict_transfer",
    "condition_check": "state_transfer",
    "negation_constraint": "state_transfer",
}
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


@dataclass(frozen=True)
class Stage44AContextPair:
    base_record: EvidenceAnswerSample
    full_sample: LogicSample
    counterfactual_sample: LogicSample
    expected_full_option_id: int
    expected_counterfactual_option_id: int


@dataclass(frozen=True)
class Stage44ALocalGroup:
    group_type: str
    surface_group_id: str
    pairs: tuple[Stage44AContextPair, ...]

    def __post_init__(self) -> None:
        if self.group_type not in LOCAL_REAL_TASK_TYPES:
            raise ValueError(f"unsupported Stage44A group type: {self.group_type}")
        if len(self.pairs) != len(LOGIC_LABELS):
            raise ValueError("Stage44A group must contain exactly five candidates")
        if len({pair.full_sample.text for pair in self.pairs}) != 1:
            raise ValueError("Stage44A group surface text must be identical")
        if {pair.expected_full_option_id for pair in self.pairs} != set(range(len(LOGIC_LABELS))):
            raise ValueError("Stage44A group must map score columns 0..4 exactly once")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stage43_checkpoint(root: Path, seed: int, max_length: int) -> Path:
    return (
        root
        / f"seed_{seed}_len_{max_length}"
        / "stage42"
        / "checkpoints"
        / f"stage42_group_full_hidden_centroid_seed_{seed}.pt"
    )


def _stage43_summary(root: Path, seed: int, max_length: int) -> Path:
    return root / f"seed_{seed}_len_{max_length}" / "stage42" / "summary.json"


def _verify_stage43_checkpoint(
    root: Path,
    checkpoint: Path,
    seed: int,
    max_length: int,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = _stage43_summary(root, seed, max_length)
    failures: list[dict[str, Any]] = []
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata = payload.get("training_metadata", {})
    verification = {
        "checkpoint_path": str(checkpoint),
        "summary_path": str(summary_path),
        "checkpoint_has_qwen": "qwen_state_dict" in payload or "model_state_dict" in payload,
        "checkpoint_seed": metadata.get("seed"),
        "adapter_variant": metadata.get("adapter_variant"),
        "target_layers": metadata.get("target_layers"),
        "raw_full_hidden_residual_scale": metadata.get("raw_full_hidden_residual_scale"),
        "stage43_passes": bool(summary.get("passes_stage_gate")),
        "stage43_fixed_centroid": float(summary.get("fixed_centroid_average_after", 0.0)),
        "stage43_wrong_context_drop": float(summary.get("wrong_context_fixed_centroid_drop", {}).get("combined", 0.0)),
        "stage43_hidden_norm_ratio": float(summary.get("hidden_norm_ratio", 0.0)),
    }
    checks = {
        "checkpoint_has_qwen": not verification["checkpoint_has_qwen"],
        "checkpoint_seed": verification["checkpoint_seed"] == seed,
        "adapter_variant": verification["adapter_variant"] == "path_specific_v2",
        "target_layers": verification["target_layers"] == [16, 24],
        "residual_scale": float(verification["raw_full_hidden_residual_scale"] or 0.0) == RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        "stage43_passes": verification["stage43_passes"],
        "stage43_fixed_centroid": verification["stage43_fixed_centroid"] >= 0.90,
        "stage43_wrong_context": verification["stage43_wrong_context_drop"] >= 0.20,
        "stage43_hidden_norm": verification["stage43_hidden_norm_ratio"] <= 2.0,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(
                {
                    "failed_stage": "stage43_checkpoint_verification",
                    "failed_gate": name,
                    "failed_metric": name,
                    "expected_threshold": True,
                    "actual_value": verification.get(name),
                    "failure_category": "checkpoint_verification",
                }
            )
    return verification, failures


def build_stage44a_groups(
    samples_by_task: dict[str, list[LogicSample]],
) -> list[Stage44ALocalGroup]:
    groups: list[Stage44ALocalGroup] = []
    for task_type, samples in sorted(samples_by_task.items()):
        records = build_evidence_answer_samples(samples, "local_semireal")
        by_surface: dict[str, dict[int, EvidenceAnswerSample]] = {}
        for record in records:
            by_surface.setdefault(record.sample.surface_group_id, {})[record.correct_option_id] = record
        for surface_group_id, by_option in sorted(by_surface.items()):
            if set(by_option) != set(range(len(LOGIC_LABELS))):
                raise ValueError(f"incomplete Stage44A surface group: {surface_group_id}")
            pairs: list[Stage44AContextPair] = []
            for option_id in range(len(LOGIC_LABELS)):
                record = by_option[option_id]
                wrong_id = (option_id + 1) % len(LOGIC_LABELS)
                counterfactual = by_option[wrong_id].sample
                pairs.append(
                    Stage44AContextPair(
                        base_record=record,
                        full_sample=record.sample,
                        counterfactual_sample=counterfactual,
                        expected_full_option_id=option_id,
                        expected_counterfactual_option_id=wrong_id,
                    )
                )
            groups.append(Stage44ALocalGroup(task_type, surface_group_id, tuple(pairs)))
    return groups


def split_stage44a_groups(
    groups: list[Stage44ALocalGroup], train_groups: int, seed: int
) -> tuple[list[Stage44ALocalGroup], list[Stage44ALocalGroup]]:
    train: list[Stage44ALocalGroup] = []
    heldout: list[Stage44ALocalGroup] = []
    for task_type in LOCAL_REAL_TASK_TYPES:
        rows = sorted((group for group in groups if group.group_type == task_type), key=lambda group: group.surface_group_id)
        random.Random(seed + list(LOCAL_REAL_TASK_TYPES).index(task_type)).shuffle(rows)
        if not 0 < train_groups < len(rows):
            raise ValueError(f"train_groups must leave held-out groups for {task_type}")
        train.extend(rows[:train_groups])
        heldout.extend(rows[train_groups:])
    if {group.surface_group_id for group in train} & {group.surface_group_id for group in heldout}:
        raise ValueError("Stage44A surface group leakage detected")
    return train, heldout


def _context_ownership_audit(records: dict[str, list[RealTaskRecord]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    forbidden = set(LOGIC_LABELS)
    for task_type, task_records in records.items():
        by_group: dict[str, list[RealTaskRecord]] = {}
        for record in task_records:
            by_group.setdefault(record.surface_group_id, []).append(record)
            text = " ".join(
                [
                    *(f"{item.summary} {item.content} {item.relation_type}" for item in record.memory_items),
                    *(f"{item.condition} {item.effect} {item.source}" for item in record.rule_items),
                ]
            ).lower()
            leaked = forbidden & set(text.replace("_", " ").split())
            row = {
                "task_type": task_type,
                "surface_group_id": record.surface_group_id,
                "source_id": record.source_id,
                "required_path": record.required_path,
                "context_owner": record.context_owner,
                "context_hash": record.context_hash,
                "memory_hash": _sha("|".join(item.content for item in record.memory_items)),
                "rule_hash": _sha("|".join(item.effect for item in record.rule_items)),
                "state_hash": _sha(json.dumps(record.state_values)),
                "logic_label_leakage": bool(leaked),
            }
            rows.append(row)
            if leaked:
                failures.append({"failed_gate": "logic_label_leakage", **row})
        for surface_group_id, group_records in by_group.items():
            expected_owner = TASK_PATH[task_type]
            if len(group_records) != len(LOGIC_LABELS):
                failures.append({"failed_gate": "incomplete_surface_group", "task_type": task_type, "surface_group_id": surface_group_id})
            memory_count = len({row.context_hash for row in group_records})
            if memory_count != len(LOGIC_LABELS):
                failures.append({"failed_gate": "context_not_unique", "task_type": task_type, "surface_group_id": surface_group_id})
            if expected_owner not in group_records[0].required_path:
                failures.append({"failed_gate": "wrong_context_owner", "task_type": task_type, "surface_group_id": surface_group_id})
    return rows, failures


def _group_sequence(groups: list[Stage44ALocalGroup], steps: int, seed: int) -> list[Stage44ALocalGroup]:
    if not groups:
        raise ValueError("Stage44A curriculum pool is empty")
    result: list[Stage44ALocalGroup] = []
    ordered = sorted(groups, key=lambda group: (group.group_type, group.surface_group_id))
    rng = random.Random(seed)
    while len(result) < steps:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:steps]


def _path_for_task(task_type: str) -> str:
    return TASK_PATH[task_type]


def _ablation_for_task(task_type: str) -> str:
    return f"no_{_path_for_task(task_type)}_path"


def _path_projected(result: dict[str, Any], projector, path: str) -> torch.Tensor:
    tensors = []
    field = f"{path}_delta_tensor"
    attention_mask = result["output"].attention_mask
    for trace in result["output"].traces.values():
        tensor = getattr(trace, field, None)
        if tensor is None:
            raise RuntimeError(f"Stage44A trace does not expose {field}")
        if tensor.shape[0] == attention_mask.shape[0] and tensor.shape[1] == 1 and path == "state":
            tensor = tensor.expand(-1, attention_mask.shape[1], -1)
        if tensor.shape[:2] != attention_mask.shape:
            raise RuntimeError(
                f"Stage44A trace/mask shape mismatch for {field}: "
                f"{tuple(tensor.shape[:2])} != {tuple(attention_mask.shape)}"
            )
        tensors.append(tensor)
    if not tensors:
        raise RuntimeError("Stage44A forward did not produce adapter traces")
    stacked = torch.stack(tensors, dim=0).sum(dim=0)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if torch.any(indices < 0):
        raise RuntimeError("Stage44A attention mask contains an empty sequence")
    delta = stacked[torch.arange(stacked.shape[0], device=stacked.device), indices]
    return projector(delta)


def _forward_local_group(
    *, backend, model, projector, full_hidden_projector, heads, context_encoder, group, option_vectors, max_length, mode
):
    if mode == "counterfactual_context":
        pairs = tuple(
            replace(
                pair,
                full_sample=pair.counterfactual_sample,
                expected_full_option_id=pair.expected_counterfactual_option_id,
            )
            for pair in group.pairs
        )
        group = Stage44ALocalGroup(group.group_type, group.surface_group_id, pairs)
        mode = "full"
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


def _evaluate_groups(
    *, backend, model, projector, full_hidden_projector, heads, groups, max_length, modes=EVAL_MODES
) -> list[dict[str, Any]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    options = torch.tensor(build_answer_option_vectors(backend, LOCAL_ANSWER_OPTIONS), dtype=torch.float32, device=backend.device)
    rows: list[dict[str, Any]] = []
    for group in groups:
        for mode in modes:
            with torch.no_grad():
                result = _forward_local_group(
                    backend=backend,
                    model=model,
                    projector=projector,
                    full_hidden_projector=full_hidden_projector,
                    heads=heads,
                    context_encoder=context_encoder,
                    group=group,
                    option_vectors=options,
                    max_length=max_length,
                    mode=mode,
                )
                projected = _path_projected(result, projector, _path_for_task(group.group_type))
                scores = _answer_scores(projected, options)
                predictions = scores.argmax(dim=-1)
                accuracy = float((predictions == result["targets"]).float().mean().cpu())
                success = float(torch.all(predictions == result["targets"]).cpu())
            rows.append(
                {
                    "task_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "required_path": _path_for_task(group.group_type),
                    "mode": mode,
                    "projected_accuracy": accuracy,
                    "surface_group_success": success,
                    "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in result["output"].traces.values()),
                    "memory_delta_norm": sum(trace.memory_delta_norm for trace in result["output"].traces.values()),
                    "rule_delta_norm": sum(trace.rule_delta_norm for trace in result["output"].traces.values()),
                    "state_delta_norm": sum(trace.state_delta_norm for trace in result["output"].traces.values()),
                }
            )
    return rows


def _mean(rows: Iterable[dict[str, Any]], field: str, **filters: Any) -> float:
    values = [float(row[field]) for row in rows if all(row.get(key) == value for key, value in filters.items())]
    return sum(values) / len(values) if values else 0.0


def _task_metrics(rows: list[dict[str, Any]], task_type: str) -> dict[str, float]:
    full = _mean(rows, "projected_accuracy", task_type=task_type, mode="full")
    result = {
        "projected_accuracy": full,
        "surface_group_success": _mean(rows, "surface_group_success", task_type=task_type, mode="full"),
    }
    for mode in EVAL_MODES[1:]:
        value = _mean(rows, "projected_accuracy", task_type=task_type, mode=mode)
        result[f"{mode}_accuracy"] = value
        result[f"{mode}_drop"] = full - value
    return result


def _set_trainable(model, projector, full_hidden_projector, paths: set[str], train_full_hidden: bool) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    for parameter in full_hidden_projector.parameters():
        parameter.requires_grad_(train_full_hidden)
        parameter.grad = None
    trainable = list(projector.parameters())
    if train_full_hidden:
        trainable.extend(full_hidden_projector.parameters())
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            enabled = any(name.startswith(f"{path}_") for path in paths)
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
    return trainable


def _open_stage44a_residual_paths(model, paths: set[str]) -> None:
    with torch.no_grad():
        for adapter in model.adapters.values():
            for path in paths:
                scale = getattr(adapter, f"{path}_residual_scale")
                if abs(float(scale.detach().cpu())) < 1.0:
                    old_scale = float(scale.detach().cpu())
                    up_projection = getattr(adapter, f"{path}_up_projection")
                    # Keep an existing nonzero path function unchanged. A path that
                    # was closed in Stage43 is opened at unit effective scale, while
                    # retaining 200.0 as the explicit residual-scale convention.
                    effective_scale = old_scale if old_scale != 0.0 else 1.0
                    up_projection.weight.mul_(effective_scale / RAW_FULL_HIDDEN_RESIDUAL_SCALE)
                    scale.fill_(RAW_FULL_HIDDEN_RESIDUAL_SCALE)


def _train_stage(
    *, backend, model, projector, full_hidden_projector, heads, pool, steps, seed, checkpoint_seed, stage, paths, max_length, output_dir
) -> tuple[list[dict[str, Any]], str]:
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors = torch.tensor(build_answer_option_vectors(backend, LOCAL_ANSWER_OPTIONS), dtype=torch.float32, device=backend.device)
    full_hidden = stage == "full_hidden_alignment"
    _open_stage44a_residual_paths(model, paths)
    trainable = _set_trainable(model, projector, full_hidden_projector, paths, full_hidden)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("Stage44A optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)
    rows: list[dict[str, Any]] = []
    for step, group in enumerate(_group_sequence(pool, steps, seed)):
        optimizer.zero_grad(set_to_none=True)
        full = _forward_local_group(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, context_encoder=context_encoder, group=group, option_vectors=option_vectors,
            max_length=max_length, mode="full",
        )
        ablated = _forward_local_group(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, context_encoder=context_encoder, group=group, option_vectors=option_vectors,
            max_length=max_length, mode=_ablation_for_task(group.group_type),
        )
        path = _path_for_task(group.group_type)
        projected = _path_projected(full, projector, path)
        ablated_projected = _path_projected(ablated, projector, path)
        scores = _answer_scores(projected, option_vectors)
        ablated_scores = _answer_scores(ablated_projected, option_vectors)
        targets = full["targets"]
        target_scores = scores.gather(1, targets[:, None]).squeeze(1)
        other_scores = scores.masked_fill(F.one_hot(targets, scores.shape[-1]).bool(), -1e4).max(dim=-1).values
        path_gap = target_scores - ablated_scores.gather(1, targets[:, None]).squeeze(1)
        projected_loss = F.cross_entropy(scores / 0.05, targets)
        margin_loss = torch.relu(0.25 - (target_scores - other_scores)).mean()
        gap_loss = torch.relu(0.20 - path_gap).mean()
        pooled = full["pooled"]
        dynamic_loss = _dynamic_centroid_loss(pooled, full["labels"])
        full_option_scores = _answer_scores(F.normalize(pooled.float(), dim=-1), option_vectors)
        full_hidden_loss = F.cross_entropy(full_option_scores / 0.05, targets)
        projected_full_loss = F.cross_entropy(_answer_scores(full_hidden_projector(pooled), option_vectors) / 0.05, targets)
        total = 3.0 * projected_loss + 2.0 * margin_loss + 2.0 * gap_loss
        if full_hidden:
            total = total + 4.0 * dynamic_loss + 2.0 * full_hidden_loss + projected_full_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "curriculum_stage": stage,
            "step": step,
            "task_type": group.group_type,
            "total_loss": float(total.detach().cpu()),
            "projected_loss": float(projected_loss.detach().cpu()),
            "margin_loss": float(margin_loss.detach().cpu()),
            "path_gap_loss": float(gap_loss.detach().cpu()),
            "dynamic_centroid_loss": float(dynamic_loss.detach().cpu()),
            "projected_accuracy": float((scores.argmax(dim=-1) == targets).float().mean().detach().cpu()),
            "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in full["output"].traces.values()),
            "rss": psutil.Process().memory_info().rss,
        }
        rows.append(row)
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(
                f"stage44a stage={stage} step={step + 1}/{steps} task={group.group_type} "
                f"loss={row['total_loss']:.6f} acc={row['projected_accuracy']:.3f}",
                flush=True,
            )
    checkpoint_dir = output_dir / "checkpoints" / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"stage44a_{stage}_seed_{checkpoint_seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {
                "stage": stage,
                "seed": checkpoint_seed,
                "adapter_variant": "path_specific_v2",
                "target_layers": [16, 24],
                "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
            },
        },
        checkpoint,
    )
    return rows, str(checkpoint)


def _build_centroids(*, backend, model, projector, full_hidden_projector, heads, groups, max_length, projected: bool):
    context_encoder = FrozenQwenContextEncoder(backend)
    options = torch.tensor(build_answer_option_vectors(backend, LOCAL_ANSWER_OPTIONS), dtype=torch.float32, device=backend.device)
    vectors: dict[str, dict[int, list[torch.Tensor]]] = {
        task: {index: [] for index in range(len(LOGIC_LABELS))}
        for task in LOCAL_REAL_TASK_TYPES
    }
    audit: list[dict[str, Any]] = []
    with torch.no_grad():
        for group in groups:
            result = _forward_local_group(
                backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
                heads=heads, context_encoder=context_encoder, group=group, option_vectors=options,
                max_length=max_length, mode="full",
            )
            value = full_hidden_projector(result["pooled"]) if projected else result["pooled"]
            for index, target in enumerate(result["targets"].tolist()):
                vectors[group.group_type][target].append(value[index].detach())
                audit.append({"split": "train", "mode": "full", "centroid_scope": group.group_type, "task_type": group.group_type, "surface_group_id": group.surface_group_id, "target": target, "used": True})
    centroids = {
        task: torch.stack(
            [torch.stack(by_target[index]).mean(dim=0) for index in range(len(LOGIC_LABELS))]
        )
        for task, by_target in vectors.items()
    }
    return centroids, audit


def _evaluate_fixed_centroids(
    *, backend, model, projector, full_hidden_projector, heads, groups, centroids, projected_centroids, max_length
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    options = torch.tensor(build_answer_option_vectors(backend, LOCAL_ANSWER_OPTIONS), dtype=torch.float32, device=backend.device)
    fixed_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    for group in groups:
        for mode in EVAL_MODES:
            with torch.no_grad():
                result = _forward_local_group(
                    backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
                    heads=heads, context_encoder=context_encoder, group=group, option_vectors=options,
                    max_length=max_length, mode=mode,
                )
                raw_centroids = centroids[group.group_type]
                projected_task_centroids = projected_centroids[group.group_type]
                raw_scores = F.normalize(result["pooled"].float(), dim=-1) @ F.normalize(raw_centroids.float(), dim=-1).T
                projected = full_hidden_projector(result["pooled"])
                projected_scores = F.normalize(projected.float(), dim=-1) @ F.normalize(projected_task_centroids.float(), dim=-1).T
                targets = result["targets"]
            for destination, scores in ((fixed_rows, raw_scores), (projected_rows, projected_scores)):
                predictions = scores.argmax(dim=-1)
                destination.append(
                    {
                        "task_type": group.group_type,
                        "surface_group_id": group.surface_group_id,
                        "mode": mode,
                        "accuracy": float((predictions == targets).float().mean().cpu()),
                        "surface_group_success": float(torch.all(predictions == targets).cpu()),
                    }
                )
    return fixed_rows, projected_rows


def _stage_gate(stage: str, metrics: dict[str, dict[str, float]], strict: bool) -> list[dict[str, Any]]:
    if not strict:
        return []
    expected_tasks = {
        "memory_transfer": ("operation_decision", "causal_trace"),
        "rule_transfer": ("priority_selection",),
        "state_transfer": ("condition_check", "negation_constraint"),
        "conflict_transfer": ("rule_conflict",),
        "combined_curriculum": LOCAL_REAL_TASK_TYPES,
        "full_hidden_alignment": LOCAL_REAL_TASK_TYPES,
    }[stage]
    failures: list[dict[str, Any]] = []
    for task in expected_tasks:
        values = metrics[task]
        checks = {
            "projected_accuracy": (values["projected_accuracy"], 0.80, "ge"),
            "surface_group_success": (values["surface_group_success"], 0.70, "ge"),
            f"{_ablation_for_task(task)}_drop": (values[f"{_ablation_for_task(task)}_drop"], 0.20, "ge"),
        }
        if TASK_PATH[task] == "memory":
            checks["no_rule_path_drop"] = (values["no_rule_path_drop"], 0.10, "le")
        if TASK_PATH[task] == "rule":
            checks["no_memory_path_drop"] = (values["no_memory_path_drop"], 0.15 if task == "rule_conflict" else 0.10, "le")
        for name, (value, threshold, op) in checks.items():
            passed = value >= threshold if op == "ge" else value <= threshold
            if not passed:
                failures.append(
                    {
                        "failed_stage": stage,
                        "failed_gate": name,
                        "failed_task": task,
                        "failed_metric": name,
                        "expected_threshold": threshold,
                        "actual_value": value,
                        "failure_category": "curriculum_stage_gate",
                    }
                )
    return failures


def _generic_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {kind: _generic_group_metrics(rows, kind) for kind in ("memory_necessity_group", "rule_necessity_group", "memory_rule_conflict_group")}


def _empty_artifacts(output: Path) -> None:
    for name in (
        "context_ownership_audit.csv", "surface_group_split_audit.csv", "training_runs.json", "loss_curves.csv",
        "curriculum_stage_metrics.csv", "task_metrics.csv", "projected_readout_metrics.csv", "fixed_centroid_metrics.csv",
        "projected_full_hidden_metrics.csv", "path_ablation_drop.csv", "surface_group_flip_metrics.csv",
        "wrong_context_metrics.csv", "generic_rehearsal_retention.csv", "centroid_build_audit.csv",
        "trace_contribution.csv", "resource_usage.json", "failure_cases.json",
    ):
        path = output / name
        if path.suffix == ".json":
            _json_dump(path, [])
        else:
            _write_csv(path, [])


def _run_seed(
    *, output: Path, model_path, stage43_root: Path, seed: int, samples_per_label: int, train_groups: int,
    max_length: int, preferred_device: str | None, strict_stage_gates: bool, stage_steps: dict[str, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    _empty_artifacts(output)
    checkpoint = _stage43_checkpoint(stage43_root, seed, max_length)
    if not checkpoint.exists():
        failure = {"failed_stage": "stage43_checkpoint_verification", "failed_gate": "checkpoint_missing", "actual_value": str(checkpoint)}
        _json_dump(output / "failure_cases.json", [failure])
        summary = {"seed": seed, "passes_stage_gate": False, "failed_stage": failure["failed_stage"], "failed_gate": failure["failed_gate"]}
        _json_dump(output / "summary.json", summary)
        return summary
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    model, projector, full_hidden_projector, payload = _load_stage41_checkpoint(backend, checkpoint)
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    verification, failures = _verify_stage43_checkpoint(stage43_root, checkpoint, seed, max_length, payload)
    _json_dump(output / "stage43_checkpoint_verification.json", verification)
    if failures:
        _json_dump(output / "failure_cases.json", failures)
        summary = {"seed": seed, "passes_stage_gate": False, **failures[0]}
        _json_dump(output / "summary.json", summary)
        return summary

    records = build_local_semireal_task_records(samples_per_label, seed, "stage44a_path_grounded_v2")
    datasets, _ = records_to_logic_datasets(records, max_length, "stage44a_path_grounded_v2")
    evidence = [record for samples in datasets.values() for record in build_evidence_answer_samples(samples, "local_semireal")]
    assert_no_logic_label_leakage(evidence)
    ownership_rows, ownership_failures = _context_ownership_audit(records)
    _write_csv(output / "context_ownership_audit.csv", ownership_rows)
    groups = build_stage44a_groups(datasets)
    train, heldout = split_stage44a_groups(groups, train_groups, seed)
    split_rows = [
        {"split": split, "task_type": group.group_type, "surface_group_id": group.surface_group_id}
        for split, rows in (("train", train), ("heldout", heldout)) for group in rows
    ]
    _write_csv(output / "surface_group_split_audit.csv", split_rows)
    _json_dump(output / "dataset_manifest.json", {"records": local_real_task_manifest(records), "seed": seed, "train_groups": train_groups})
    if ownership_failures:
        _json_dump(output / "failure_cases.json", ownership_failures)
        summary = {"seed": seed, "passes_stage_gate": False, "failed_stage": "dataset_audit", **ownership_failures[0]}
        _json_dump(output / "summary.json", summary)
        return summary

    generic_train, generic_heldout, _ = _build_generic_group_splits(
        local_samples_per_label=samples_per_label,
        local_train_groups=train_groups,
        seed=seed,
        max_length=max_length,
    )
    generic_baseline_rows = _evaluate_generic_groups(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        groups=generic_heldout, max_length=max_length, modes=("full", "no_memory_path", "no_rule_path", "wrong_context"),
    )
    generic_baseline = _generic_metrics(generic_baseline_rows)
    loss_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    stage_specs = [
        ("memory_transfer", {"operation_decision", "causal_trace"}, {"memory"}),
        ("rule_transfer", {"priority_selection", "operation_decision", "causal_trace"}, {"rule"}),
        ("state_transfer", {"condition_check", "negation_constraint", "operation_decision", "priority_selection"}, {"state"}),
        ("conflict_transfer", {"rule_conflict", "operation_decision", "priority_selection", "condition_check"}, {"rule"}),
        ("combined_curriculum", set(LOCAL_REAL_TASK_TYPES), {"memory", "rule", "state"}),
    ]
    for stage_index, (stage, tasks, paths) in enumerate(stage_specs):
        pool = [group for group in train if group.group_type in tasks]
        losses, stage_checkpoint = _train_stage(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, pool=pool, steps=stage_steps[stage], seed=seed + stage_index, checkpoint_seed=seed, stage=stage,
            paths=paths, max_length=max_length, output_dir=output,
        )
        loss_rows.extend(losses)
        eval_rows = _evaluate_groups(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=heldout, max_length=max_length,
            modes=("full", "no_memory_path", "no_rule_path", "no_state_path", "wrong_context"),
        )
        metrics = {task: _task_metrics(eval_rows, task) for task in LOCAL_REAL_TASK_TYPES}
        stage_failures = _stage_gate(stage, metrics, strict_stage_gates)
        generic_rows = _evaluate_generic_groups(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            groups=generic_heldout, max_length=max_length, modes=("full", "no_memory_path", "no_rule_path", "wrong_context"),
        )
        generic = _generic_metrics(generic_rows)
        stage_rows.append(
            {
                "curriculum_stage": stage,
                "passes_stage_gate": not stage_failures,
                "checkpoint_path": stage_checkpoint,
                **{f"{task}_accuracy": values["projected_accuracy"] for task, values in metrics.items()},
            }
        )
        training_rows.append({"stage": stage, "checkpoint_path": stage_checkpoint, "steps": stage_steps[stage]})
        if stage_failures:
            all_failures.extend({**failure, "checkpoint_path": stage_checkpoint} for failure in stage_failures)
            break
        for kind, values in generic.items():
            if values["accuracy"] < generic_baseline[kind]["accuracy"] - 0.05:
                all_failures.append(
                    {
                        "failed_stage": stage,
                        "failed_gate": "generic_rehearsal_regression",
                        "failed_task": kind,
                        "expected_threshold": generic_baseline[kind]["accuracy"] - 0.05,
                        "actual_value": values["accuracy"],
                        "failure_category": "catastrophic_forgetting",
                        "checkpoint_path": stage_checkpoint,
                    }
                )
                break
        if all_failures:
            break

    projected_before_alignment = 0.0
    fixed_before = 0.0
    fixed_after = 0.0
    centroid_audit: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    if not all_failures:
        projected_before_rows = _evaluate_groups(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=heldout, max_length=max_length, modes=("full",),
        )
        projected_before_alignment = _mean(projected_before_rows, "projected_accuracy", mode="full")
        centroids_before, centroid_audit = _build_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=train, max_length=max_length, projected=False,
        )
        projected_centroids_before, _ = _build_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=train, max_length=max_length, projected=True,
        )
        before_rows, _ = _evaluate_fixed_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=heldout, centroids=centroids_before, projected_centroids=projected_centroids_before,
            max_length=max_length,
        )
        fixed_before = _mean(before_rows, "accuracy", mode="full")
        losses, stage_checkpoint = _train_stage(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, pool=train, steps=stage_steps["full_hidden_alignment"], seed=seed + 5, checkpoint_seed=seed,
            stage="full_hidden_alignment", paths={"memory", "rule", "state"}, max_length=max_length, output_dir=output,
        )
        loss_rows.extend(losses)
        training_rows.append({"stage": "full_hidden_alignment", "checkpoint_path": stage_checkpoint, "steps": stage_steps["full_hidden_alignment"]})
        centroids_after, centroid_audit = _build_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=train, max_length=max_length, projected=False,
        )
        projected_centroids_after, _ = _build_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=train, max_length=max_length, projected=True,
        )
        fixed_rows, projected_full_rows = _evaluate_fixed_centroids(
            backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
            heads=heads, groups=heldout, centroids=centroids_after, projected_centroids=projected_centroids_after,
            max_length=max_length,
        )
        fixed_after = _mean(fixed_rows, "accuracy", mode="full")

    projected_rows = _evaluate_groups(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        heads=heads, groups=heldout, max_length=max_length,
    )
    task_metrics = {task: _task_metrics(projected_rows, task) for task in LOCAL_REAL_TASK_TYPES}
    generic_final_rows = _evaluate_generic_groups(
        backend=backend, model=model, projector=projector, full_hidden_projector=full_hidden_projector,
        groups=generic_heldout, max_length=max_length, modes=("full", "no_memory_path", "no_rule_path", "wrong_context"),
    )
    generic_final = _generic_metrics(generic_final_rows)
    overall_projected = _mean(projected_rows, "projected_accuracy", mode="full")
    overall_success = _mean(projected_rows, "surface_group_success", mode="full")
    wrong_drop = overall_projected - _mean(projected_rows, "projected_accuracy", mode="wrong_context")
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    hidden_norm_ratio = max((float(row["hidden_norm_ratio"]) for row in projected_rows), default=1.0)
    if fixed_before >= 0.95:
        fixed_improvement_passed = fixed_after >= fixed_before - 0.01
    else:
        fixed_improvement_passed = fixed_after >= fixed_before + 0.05
    final_checks = {
        "overall_projected": overall_projected >= 0.85,
        "overall_fixed": fixed_after >= 0.75,
        "surface_group_success": overall_success >= 0.75,
        "wrong_context_drop": wrong_drop >= 0.20,
        "fixed_improvement": fixed_improvement_passed,
        "projected_regression": overall_projected >= projected_before_alignment - 0.05,
        "disabled_zero_equivalence": all(
            abs(values["adapter_disabled_accuracy"] - values["zero_scale_accuracy"]) <= 1e-9
            for values in task_metrics.values()
        ),
        "per_task_projected": all(values["projected_accuracy"] >= 0.80 for values in task_metrics.values()),
        "per_task_surface_group": all(values["surface_group_success"] >= 0.70 for values in task_metrics.values()),
        "per_task_fixed": all(
            _mean(fixed_rows, "accuracy", task_type=task, mode="full") >= 0.70
            for task in LOCAL_REAL_TASK_TYPES
        ),
        "fixed_wrong_context_drop": fixed_after - _mean(fixed_rows, "accuracy", mode="wrong_context") >= 0.20,
        "hidden_norm": hidden_norm_ratio <= 2.0,
        "qwen_gradients": qwen_gradients == 0,
        "qwen_frozen": sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad) == 0,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    if strict_stage_gates:
        for name, passed in final_checks.items():
            if not passed:
                all_failures.append(
                    {
                        "failed_stage": "final_evaluation",
                        "failed_gate": name,
                        "failed_metric": name,
                        "expected_threshold": True,
                        "actual_value": passed,
                        "failure_category": "final_gate",
                    }
                )

    ablation_rows = []
    for task, values in task_metrics.items():
        for mode in ("no_memory_path", "no_rule_path", "no_state_path"):
            ablation_rows.append({"task_type": task, "mode": mode, "full_accuracy": values["projected_accuracy"], "ablated_accuracy": values[f"{mode}_accuracy"], "absolute_drop": values[f"{mode}_drop"]})
    flip_rows = [row for row in projected_rows if row["mode"] in {"full", "counterfactual_context"}]
    wrong_rows = [row for row in projected_rows if row["mode"] == "wrong_context"]
    trace_rows = [{key: row[key] for key in ("task_type", "surface_group_id", "mode", "memory_delta_norm", "rule_delta_norm", "state_delta_norm", "hidden_norm_ratio")} for row in projected_rows]
    generic_rows = [
        {"phase": phase, "group_type": kind, **values}
        for phase, metrics in (("before", generic_baseline), ("after", generic_final))
        for kind, values in metrics.items()
    ]
    _write_csv(output / "loss_curves.csv", loss_rows)
    _write_csv(output / "curriculum_stage_metrics.csv", stage_rows)
    _json_dump(output / "training_runs.json", training_rows)
    _write_csv(output / "projected_readout_metrics.csv", projected_rows)
    _write_csv(output / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output / "projected_full_hidden_metrics.csv", projected_full_rows)
    _write_csv(output / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output / "surface_group_flip_metrics.csv", flip_rows)
    _write_csv(output / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output / "generic_rehearsal_retention.csv", generic_rows)
    _write_csv(output / "centroid_build_audit.csv", centroid_audit)
    _write_csv(output / "trace_contribution.csv", trace_rows)
    _write_csv(output / "task_metrics.csv", [{"task_type": task, **values} for task, values in task_metrics.items()])
    _json_dump(output / "resource_usage.json", {"runtime_seconds": time.perf_counter() - started, "rss": psutil.Process().memory_info().rss})
    _json_dump(output / "failure_cases.json", all_failures)
    summary = {
        "seed": seed,
        "max_length": max_length,
        "samples_per_label": samples_per_label,
        "train_groups": train_groups,
        "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        "overall_projected_accuracy": overall_projected,
        "overall_fixed_centroid_accuracy": fixed_after,
        "fixed_centroid_before": fixed_before,
        "fixed_centroid_after": fixed_after,
        "projected_before_alignment": projected_before_alignment,
        "surface_group_success": overall_success,
        "wrong_context_drop": wrong_drop,
        "hidden_norm_ratio": hidden_norm_ratio,
        "task_metrics": task_metrics,
        "generic_before": generic_baseline,
        "generic_after": generic_final,
        "qwen_trainable_parameters": sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        "qwen_gradients": qwen_gradients,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "completed_stages": [row["curriculum_stage"] for row in stage_rows] + (["full_hidden_alignment"] if any(row["stage"] == "full_hidden_alignment" for row in training_rows) else []),
        "failed_stage": all_failures[0].get("failed_stage") if all_failures else None,
        "failed_gate": all_failures[0].get("failed_gate") if all_failures else None,
        "passes_stage_gate": not all_failures and all(final_checks.values()),
        "runtime_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "summary.json", summary)
    return summary


def run_qwen3_stage44a_local_multiclass_integration(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    stage43_root: str | Path = DEFAULT_STAGE43_ROOT,
    seeds: tuple[int, ...] = (202,),
    samples_per_label: int = 24,
    train_groups: int = 16,
    max_length: int = 128,
    preferred_device: str | None = "cuda",
    strict_stage_gates: bool = True,
    memory_steps: int = 40,
    rule_steps: int = 40,
    state_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    full_hidden_steps: int = 80,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = {
        "memory_transfer": memory_steps,
        "rule_transfer": rule_steps,
        "state_transfer": state_steps,
        "conflict_transfer": conflict_steps,
        "combined_curriculum": combined_steps,
        "full_hidden_alignment": full_hidden_steps,
    }
    summaries = []
    for seed in seeds:
        print(f"stage44a_seed_start seed={seed} max_length={max_length}", flush=True)
        summaries.append(
            _run_seed(
                output=output / f"seed_{seed}", model_path=model_path, stage43_root=Path(stage43_root), seed=seed,
                samples_per_label=samples_per_label, train_groups=train_groups, max_length=max_length,
                preferred_device=preferred_device, strict_stage_gates=strict_stage_gates, stage_steps=steps,
            )
        )
        if strict_stage_gates and not summaries[-1].get("passes_stage_gate"):
            break
    passed = [summary for summary in summaries if summary.get("passes_stage_gate")]
    projected = [float(summary.get("overall_projected_accuracy", 0.0)) for summary in summaries]
    fixed = [float(summary.get("overall_fixed_centroid_accuracy", 0.0)) for summary in summaries]
    flips = [float(summary.get("surface_group_success", 0.0)) for summary in summaries]
    aggregate = {
        "seeds": list(seeds),
        "completed_seed_count": len(summaries),
        "passed_seed_count": len(passed),
        "projected_accuracy_mean": statistics.mean(projected) if projected else 0.0,
        "projected_accuracy_std": statistics.stdev(projected) if len(projected) > 1 else 0.0,
        "fixed_centroid_mean": statistics.mean(fixed) if fixed else 0.0,
        "fixed_centroid_std": statistics.stdev(fixed) if len(fixed) > 1 else 0.0,
        "surface_group_success_mean": statistics.mean(flips) if flips else 0.0,
        "surface_group_success_std": statistics.stdev(flips) if len(flips) > 1 else 0.0,
        "per_seed": summaries,
    }
    aggregate["passes_stage_gate"] = (
        len(summaries) == len(seeds)
        and len(passed) == len(seeds)
        and aggregate["projected_accuracy_std"] <= 0.08
        and aggregate["fixed_centroid_std"] <= 0.08
        and aggregate["surface_group_success_std"] <= 0.10
    )
    aggregate["allows_stage44b"] = aggregate["passes_stage_gate"]
    _json_dump(output / "summary.json", aggregate)
    _write_csv(output / "seed_stability.csv", [{"seed": summary.get("seed"), "passes_stage_gate": summary.get("passes_stage_gate"), "projected_accuracy": summary.get("overall_projected_accuracy"), "fixed_centroid_accuracy": summary.get("overall_fixed_centroid_accuracy"), "surface_group_success": summary.get("surface_group_success"), "failed_stage": summary.get("failed_stage"), "failed_gate": summary.get("failed_gate")} for summary in summaries])
    return aggregate
