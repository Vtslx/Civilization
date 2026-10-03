from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
import torch.nn.functional as F

from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .binary_path_diagnostic_data import (
    RULE_CONDITIONED_CONFLICT_MODE,
    BinaryDiagnosticPair,
    assert_binary_pairs_valid,
    assert_rule_conditioned_conflict_pairs_valid,
    build_binary_path_diagnostic_pairs,
    build_rule_conditioned_conflict_pairs,
    split_binary_pairs,
)
from .binary_path_diagnostic_training import _residual_parameters
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_training import _answer_scores, _margin_loss
from .projected_binary_path_necessity import (
    EVALUATION_MODES,
    _centroid_predictions,
    _evaluate_projected_matrix,
    _forward_vectors,
    _load_projected_checkpoint,
    _pair_forward,
    _save_projected_checkpoint,
    _targets_for_pairs,
)
from .rule_conflict_curriculum import (
    CURRICULUM_MODES,
    _balanced_combined_pairs,
    _train_curriculum,
)


class FullHiddenCentroidProjector(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(vector.float()))


@dataclass
class FullHiddenAlignmentResult:
    seed: int
    target_layers: tuple[int, ...]
    alignment_steps: int
    losses: list[dict[str, float]]
    loss_decreased: dict[str, bool]
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    checkpoint_path: str


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


def _window_decreased(rows: list[dict[str, float]], key: str) -> bool:
    values = [float(row[key]) for row in rows if key in row]
    if len(values) < 2:
        return False
    window = max(1, min(10, len(values) // 3))
    return sum(values[-window:]) / window < sum(values[:window]) / window


def _samples_for_pairs(pairs: list[BinaryDiagnosticPair]):
    return [sample for pair in pairs for sample in (pair.full_sample, pair.counterfactual_sample)]


def _build_stage36_pairs(
    pairs_per_mode: int,
    train_pairs: int,
    held_out_pairs: int,
    seed: int,
    max_length: int,
) -> tuple[dict[str, list[BinaryDiagnosticPair]], dict[str, list[BinaryDiagnosticPair]]]:
    base = build_binary_path_diagnostic_pairs(pairs_per_mode=pairs_per_mode, seed=seed, max_seq_len=max_length)
    conflict_v2 = build_rule_conditioned_conflict_pairs(pairs_per_mode=pairs_per_mode, seed=seed, max_seq_len=max_length)
    assert_binary_pairs_valid([pair for mode in ("memory_only_diagnostic", "rule_only_diagnostic") for pair in base[mode]])
    assert_rule_conditioned_conflict_pairs_valid(conflict_v2)
    train_by_mode: dict[str, list[BinaryDiagnosticPair]] = {}
    test_by_mode: dict[str, list[BinaryDiagnosticPair]] = {}
    for mode, pairs in (
        ("memory_only_diagnostic", base["memory_only_diagnostic"]),
        ("rule_only_diagnostic", base["rule_only_diagnostic"]),
        (RULE_CONDITIONED_CONFLICT_MODE, conflict_v2),
    ):
        train, test = split_binary_pairs(pairs, train_pairs, held_out_pairs)
        train_by_mode[mode] = train
        test_by_mode[mode] = test
    train_by_mode["combined_binary_curriculum"] = _balanced_combined_pairs(
        train_by_mode["memory_only_diagnostic"],
        train_by_mode["rule_only_diagnostic"],
        train_by_mode[RULE_CONDITIONED_CONFLICT_MODE],
        pairs_per_mode=train_pairs,
    )
    test_by_mode["combined_binary_curriculum"] = _balanced_combined_pairs(
        test_by_mode["memory_only_diagnostic"],
        test_by_mode["rule_only_diagnostic"],
        test_by_mode[RULE_CONDITIONED_CONFLICT_MODE],
        pairs_per_mode=held_out_pairs,
    )
    return train_by_mode, test_by_mode


def _centroid_matrix(
    backend: Qwen3Backend,
    model,
    context_encoder: FrozenQwenContextEncoder,
    train_pairs: list[BinaryDiagnosticPair],
    max_length: int,
    projector: FullHiddenCentroidProjector | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = []
    targets = _targets_for_pairs(train_pairs)
    with torch.no_grad():
        for pair in train_pairs:
            full_hidden, _delta = _pair_forward(backend, model, context_encoder, pair, max_length, "full")
            if projector is not None:
                full_hidden = projector(full_hidden)
            vectors.append(full_hidden.float())
    matrix = torch.cat(vectors, dim=0)
    labels = torch.tensor(targets, dtype=torch.long, device=backend.device)
    centroids = []
    for option_id in (0, 1):
        rows = matrix[labels == option_id]
        if rows.numel() == 0:
            raise ValueError("missing fixed centroid rows")
        centroids.append(rows.mean(dim=0))
    return torch.stack(centroids, dim=0).detach(), labels


def _centroid_scores(vectors: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    normalized_vectors = F.normalize(vectors.float(), dim=-1)
    normalized_centroids = F.normalize(centroids.float(), dim=-1)
    return normalized_vectors @ normalized_centroids.T


def _train_full_hidden_alignment(
    backend: Qwen3Backend,
    model,
    projector: PathReadoutProjector,
    train_by_mode: dict[str, list[BinaryDiagnosticPair]],
    output_dir: Path,
    seed: int,
    target_layers: tuple[int, ...],
    alignment_steps: int,
    max_length: int,
    learning_rate: float,
) -> tuple[FullHiddenCentroidProjector, FullHiddenAlignmentResult]:
    torch.manual_seed(seed)
    random.seed(seed)
    full_hidden_projector = FullHiddenCentroidProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    fixed_centroids_by_mode = {
        mode: _centroid_matrix(backend, model, context_encoder, pairs, max_length)[0]
        for mode, pairs in train_by_mode.items()
    }
    projected_centroids_by_mode = {
        mode: _centroid_matrix(backend, model, context_encoder, pairs, max_length, full_hidden_projector)[0]
        for mode, pairs in train_by_mode.items()
    }
    shuffled_by_mode: dict[str, list[BinaryDiagnosticPair]] = {}
    for mode in CURRICULUM_MODES:
        rows = sorted(train_by_mode[mode], key=lambda pair: pair.pair_id)
        random.Random(seed + len(mode)).shuffle(rows)
        shuffled_by_mode[mode] = rows
    option_vectors = torch.tensor(
        build_answer_option_vectors(backend, train_by_mode["memory_only_diagnostic"][0].answer_options),
        dtype=torch.float32,
        device=backend.device,
    )
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    residual_parameters = _residual_parameters(model)
    residual_ids = {id(parameter) for parameter in residual_parameters}
    adapter_parameters = [parameter for parameter in model.parameters() if id(parameter) not in residual_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
            {"params": full_hidden_projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    losses: list[dict[str, float]] = []
    for step in range(alignment_steps):
        optimizer.zero_grad(set_to_none=True)
        batch_modes = (
            "memory_only_diagnostic",
            "rule_only_diagnostic",
            RULE_CONDITIONED_CONFLICT_MODE,
            "combined_binary_curriculum",
            "combined_binary_curriculum",
            "combined_binary_curriculum",
            "combined_binary_curriculum",
            "combined_binary_curriculum",
            "combined_binary_curriculum",
            "combined_binary_curriculum",
        )
        full_losses = []
        projected_losses = []
        delta_losses = []
        answer_retention_losses = []
        full_answer_losses = []
        pair_flip_losses = []
        batch_hidden_parts = []
        batch_target_parts = []
        for batch_index, training_mode in enumerate(batch_modes):
            rows = shuffled_by_mode[training_mode]
            pair = rows[(step + batch_index) % len(rows)]
            full_hidden, delta = _pair_forward(backend, model, context_encoder, pair, max_length, "full")
            targets = torch.tensor([pair.expected_full_option_id, pair.expected_counterfactual_option_id], dtype=torch.long, device=backend.device)
            centroids = fixed_centroids_by_mode[training_mode]
            projected_centroids = projected_centroids_by_mode[training_mode]
            full_scores = _centroid_scores(full_hidden, centroids)
            projected_full = full_hidden_projector(full_hidden)
            projected_scores = _centroid_scores(projected_full, projected_centroids)
            projected_delta = projector(delta)
            delta_scores = _centroid_scores(projected_delta, projected_centroids)
            projected_answer_scores = _answer_scores(projected_delta, option_vectors)
            full_answer_scores = _answer_scores(full_hidden, option_vectors)
            for index, target in enumerate(targets.tolist()):
                full_loss, _full_margin = _margin_loss(full_scores[index : index + 1], target, 0.20)
                projected_loss, _projected_margin = _margin_loss(projected_scores[index : index + 1], target, 0.20)
                delta_loss, _delta_margin = _margin_loss(delta_scores[index : index + 1], target, 0.20)
                answer_retention_loss, _answer_margin = _margin_loss(projected_answer_scores[index : index + 1], target, 0.20)
                full_answer_loss, _full_answer_margin = _margin_loss(full_answer_scores[index : index + 1], target, 0.20)
                full_losses.append(4.0 * full_loss if training_mode == "combined_binary_curriculum" else full_loss)
                projected_losses.append(projected_loss)
                delta_losses.append(delta_loss)
                answer_retention_losses.append(answer_retention_loss)
                full_answer_losses.append(full_answer_loss)
            pair_flip_loss = (
                torch.relu(
                    (F.normalize(full_hidden[0:1], dim=-1) * F.normalize(full_hidden[1:2], dim=-1)).sum(dim=-1) - 0.50
                ).mean()
            )
            pair_flip_losses.append(4.0 * pair_flip_loss if training_mode == "combined_binary_curriculum" else pair_flip_loss)
            batch_hidden_parts.append(full_hidden)
            batch_target_parts.append(targets)
        batch_hidden = torch.cat(batch_hidden_parts, dim=0)
        batch_targets = torch.cat(batch_target_parts, dim=0)
        dynamic_centroids = []
        for option_id in (0, 1):
            rows = batch_hidden[batch_targets == option_id]
            dynamic_centroids.append(rows.mean(dim=0))
        dynamic_scores = _centroid_scores(batch_hidden, torch.stack(dynamic_centroids, dim=0))
        dynamic_losses = []
        for index, target in enumerate(batch_targets.tolist()):
            dynamic_loss, _dynamic_margin = _margin_loss(dynamic_scores[index : index + 1], target, 0.25)
            dynamic_losses.append(dynamic_loss)
        full_hidden_centroid_margin_loss = torch.stack(full_losses).mean()
        full_hidden_answer_margin_loss = torch.stack(full_answer_losses).mean()
        projected_full_hidden_alignment_loss = torch.stack(projected_losses).mean()
        projected_delta_retention_loss = torch.stack(delta_losses).mean()
        projected_answer_retention_loss = torch.stack(answer_retention_losses).mean()
        full_hidden_pair_flip_loss = torch.stack(pair_flip_losses).mean()
        dynamic_centroid_separation_loss = torch.stack(dynamic_losses).mean()
        total = (
            2.00 * full_hidden_centroid_margin_loss
            + 2.00 * full_hidden_answer_margin_loss
            + 0.25 * projected_full_hidden_alignment_loss
            + 0.25 * projected_delta_retention_loss
            + 6.00 * projected_answer_retention_loss
            + 2.00 * full_hidden_pair_flip_loss
            + 10.00 * dynamic_centroid_separation_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()) + list(full_hidden_projector.parameters()), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": float(step),
                "total_loss": float(total.detach().cpu()),
                "full_hidden_centroid_margin_loss": float(full_hidden_centroid_margin_loss.detach().cpu()),
                "full_hidden_answer_margin_loss": float(full_hidden_answer_margin_loss.detach().cpu()),
                "projected_full_hidden_alignment_loss": float(projected_full_hidden_alignment_loss.detach().cpu()),
                "projected_delta_retention_loss": float(projected_delta_retention_loss.detach().cpu()),
                "projected_answer_retention_loss": float(projected_answer_retention_loss.detach().cpu()),
                "full_hidden_pair_flip_loss": float(full_hidden_pair_flip_loss.detach().cpu()),
                "dynamic_centroid_separation_loss": float(dynamic_centroid_separation_loss.detach().cpu()),
            }
        )
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"full_hidden_centroid_alignment_seed_{seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {
                "seed": seed,
                "target_layers": target_layers,
                "alignment_steps": alignment_steps,
            },
        },
        checkpoint_path,
    )
    result = FullHiddenAlignmentResult(
        seed=seed,
        target_layers=target_layers,
        alignment_steps=alignment_steps,
        losses=losses,
        loss_decreased={
            "full_hidden_centroid_margin_loss": _window_decreased(losses, "full_hidden_centroid_margin_loss"),
            "projected_full_hidden_alignment_loss": _window_decreased(losses, "projected_full_hidden_alignment_loss"),
            "full_hidden_pair_flip_loss": _window_decreased(losses, "full_hidden_pair_flip_loss"),
        },
        qwen_trainable_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        qwen_gradients_present=sum(parameter.grad is not None for parameter in backend.model.parameters()),
        optimizer_contains_qwen_parameters=any(
            id(parameter) in qwen_ids
            for parameter in list(model.parameters()) + list(projector.parameters()) + list(full_hidden_projector.parameters())
        ),
        checkpoint_path=str(checkpoint_path),
    )
    return full_hidden_projector, result


def _evaluate_full_hidden_projected(
    backend: Qwen3Backend,
    model,
    full_hidden_projector: FullHiddenCentroidProjector,
    train_pairs: list[BinaryDiagnosticPair],
    test_pairs: list[BinaryDiagnosticPair],
    mode: str,
    stage: str,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    train_vectors = []
    train_targets = _targets_for_pairs(train_pairs)
    with torch.no_grad():
        for pair in train_pairs:
            full_hidden, _delta = _pair_forward(backend, model, context_encoder, pair, max_length, "full")
            train_vectors.append(full_hidden_projector(full_hidden).float().cpu())
    train_matrix = torch.cat(train_vectors, dim=0).numpy()
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    test_targets = _targets_for_pairs(test_pairs)
    eval_samples = _samples_for_pairs(test_pairs)
    with torch.no_grad():
        full_hidden, _delta, _traces = _forward_vectors(backend, model, context_encoder, eval_samples, max_length, "full")
        projected_full = full_hidden_projector(full_hidden).float().cpu().numpy()
    predictions = _centroid_predictions(train_matrix, train_targets, projected_full)
    accuracy = sum(pred == target for pred, target in zip(predictions, test_targets, strict=True)) / len(test_targets)
    metric_rows.append({"diagnostic_mode": mode, "stage": stage, "readout": "projected_full_hidden_centroid", "accuracy": accuracy})
    for pair_index, pair in enumerate(test_pairs):
        first = pair_index * 2
        second = first + 1
        pair_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "pair_id": pair.pair_id,
                "projected_full_hidden_pair_success": predictions[first] == test_targets[first] and predictions[second] == test_targets[second],
            }
        )
    return metric_rows, pair_rows


def run_qwen3_full_hidden_centroid_alignment(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/full_hidden_centroid_alignment",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    pairs_per_mode: int = 120,
    train_pairs: int = 80,
    held_out_pairs: int = 40,
    alignment_steps: int = 80,
    memory_steps: int | None = None,
    rule_steps: int | None = None,
    conflict_steps: int | None = None,
    combined_steps: int | None = None,
    centroid_steps: int | None = None,
    max_length: int = 64,
    learning_rate: float = 3e-4,
    target_layers: tuple[int, ...] = (16, 24),
    preferred_device: str | None = None,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
    active_modes: tuple[str, ...] = CURRICULUM_MODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    train_by_mode, test_by_mode = _build_stage36_pairs(pairs_per_mode, train_pairs, held_out_pairs, seed, max_length)
    model, projector, curriculum = _train_curriculum(
        backend=backend,
        train_by_mode=train_by_mode,
        output_dir=output_path,
        seed=seed,
        target_layers=target_layers,
        memory_steps=memory_steps if memory_steps is not None else (40 if pairs_per_mode >= 120 else 2),
        rule_steps=rule_steps if rule_steps is not None else (40 if pairs_per_mode >= 120 else 2),
        conflict_steps=conflict_steps if conflict_steps is not None else (60 if pairs_per_mode >= 120 else 2),
        combined_steps=combined_steps if combined_steps is not None else (80 if pairs_per_mode >= 120 else 2),
        centroid_steps=centroid_steps if centroid_steps is not None else (20 if pairs_per_mode >= 120 else 2),
        max_length=max_length,
        learning_rate=learning_rate,
    )
    _load_projected_checkpoint(curriculum.checkpoint_paths["centroid_regularization"], model, projector)
    before_projected_rows: list[dict[str, Any]] = []
    before_fixed_rows: list[dict[str, Any]] = []
    before_pair_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    for mode in active_modes:
        projected_rows, _raw_rows, fixed_rows, pair_rows, _trace_rows, failures = _evaluate_projected_matrix(
            backend,
            model,
            projector,
            train_by_mode[mode],
            test_by_mode[mode],
            mode,
            "before_full_hidden_alignment",
            max_length,
            evaluation_modes,
        )
        before_projected_rows.extend(projected_rows)
        before_fixed_rows.extend(fixed_rows)
        before_pair_rows.extend(pair_rows)
        failure_cases.extend(failures)
    full_hidden_projector, alignment = _train_full_hidden_alignment(
        backend=backend,
        model=model,
        projector=projector,
        train_by_mode=train_by_mode,
        output_dir=output_path,
        seed=seed,
        target_layers=target_layers,
        alignment_steps=alignment_steps,
        max_length=max_length,
        learning_rate=learning_rate,
    )
    projected_rows_all: list[dict[str, Any]] = []
    raw_rows_all: list[dict[str, Any]] = []
    fixed_rows_all: list[dict[str, Any]] = []
    pair_rows_all: list[dict[str, Any]] = []
    trace_rows_all: list[dict[str, Any]] = []
    projected_full_rows: list[dict[str, Any]] = []
    projected_full_pair_rows: list[dict[str, Any]] = []
    for mode in active_modes:
        projected_rows, raw_rows, fixed_rows, pair_rows, trace_rows, failures = _evaluate_projected_matrix(
            backend,
            model,
            projector,
            train_by_mode[mode],
            test_by_mode[mode],
            mode,
            "after_full_hidden_alignment",
            max_length,
            evaluation_modes,
        )
        projected_rows_all.extend(projected_rows)
        raw_rows_all.extend(raw_rows)
        fixed_rows_all.extend(fixed_rows)
        pair_rows_all.extend(pair_rows)
        trace_rows_all.extend(trace_rows)
        failure_cases.extend(failures)
        projected_full, projected_full_pairs = _evaluate_full_hidden_projected(
            backend,
            model,
            full_hidden_projector,
            train_by_mode[mode],
            test_by_mode[mode],
            mode,
            "after_full_hidden_alignment",
            max_length,
        )
        projected_full_rows.extend(projected_full)
        projected_full_pair_rows.extend(projected_full_pairs)
    before_fixed_average = sum(
        _mean(before_fixed_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "full")
        for mode in active_modes
    ) / len(active_modes)
    after_fixed_average = sum(
        _mean(fixed_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "full")
        for mode in active_modes
    ) / len(active_modes)
    final_projected = {
        mode: _mean(projected_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "full")
        for mode in active_modes
    }
    final_fixed = {
        mode: _mean(fixed_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "full")
        for mode in active_modes
    }
    projected_full = {
        mode: _mean(projected_full_rows, lambda row, current=mode: row["diagnostic_mode"] == current)
        for mode in active_modes
    }
    drops = {}
    ablation_rows: list[dict[str, Any]] = []
    for mode in active_modes:
        full = final_projected[mode]
        for eval_mode in ("no_memory_path", "no_rule_path", "no_state_path"):
            ablated = _mean(projected_rows_all, lambda row, current=mode, current_eval=eval_mode: row["diagnostic_mode"] == current and row["eval_mode"] == current_eval)
            drop = full - ablated
            drops[(mode, eval_mode)] = drop
            ablation_rows.append({"diagnostic_mode": mode, "eval_mode": eval_mode, "full_accuracy": full, "ablated_accuracy": ablated, "absolute_drop": drop})
    final_pair = {
        mode: _mean(pair_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current, field="projected_pair_success")
        for mode in active_modes
    }
    full_hidden_pair = {
        mode: _mean(projected_full_pair_rows, lambda row, current=mode: row["diagnostic_mode"] == current, field="projected_full_hidden_pair_success")
        for mode in active_modes
    }
    max_hidden_norm = max((float(row["hidden_norm_ratio"]) for row in trace_rows_all if row["hidden_norm_ratio"] is not None), default=1.0)
    stage_gates = {
        "qwen_frozen": alignment.qwen_trainable_parameter_count == 0 and alignment.qwen_gradients_present == 0 and not alignment.optimizer_contains_qwen_parameters,
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(
            abs(
                _mean(projected_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "adapter_disabled")
                - _mean(projected_rows_all, lambda row, current=mode: row["diagnostic_mode"] == current and row["eval_mode"] == "zero_scale")
            )
            <= 1e-9
            for mode in CURRICULUM_MODES
        ),
        "hidden_norm_ratio": max_hidden_norm <= 2.0,
        "projected_memory_retained": "memory_only_diagnostic" not in active_modes or final_projected["memory_only_diagnostic"] >= 0.95,
        "projected_rule_retained": "rule_only_diagnostic" not in active_modes or final_projected["rule_only_diagnostic"] >= 0.95,
        "projected_conflict_retained": RULE_CONDITIONED_CONFLICT_MODE not in active_modes or final_projected[RULE_CONDITIONED_CONFLICT_MODE] >= 0.95,
        "projected_combined_retained": "combined_binary_curriculum" not in active_modes or final_projected["combined_binary_curriculum"] >= 0.95,
        "memory_path_drop": "memory_only_diagnostic" not in active_modes or drops.get(("memory_only_diagnostic", "no_memory_path"), 0.0) >= 0.25,
        "rule_path_drop": "rule_only_diagnostic" not in active_modes or drops.get(("rule_only_diagnostic", "no_rule_path"), 0.0) >= 0.25,
        "conflict_rule_path_drop": RULE_CONDITIONED_CONFLICT_MODE not in active_modes or drops.get((RULE_CONDITIONED_CONFLICT_MODE, "no_rule_path"), 0.0) >= 0.25,
        "fixed_centroid_improvement": after_fixed_average >= before_fixed_average + 0.05,
        "fixed_centroid_average": after_fixed_average >= 0.80,
        "fixed_centroid_per_mode": all(value >= 0.75 for value in final_fixed.values()),
        "combined_fixed_centroid": "combined_binary_curriculum" not in active_modes or final_fixed["combined_binary_curriculum"] >= 0.75,
        "full_hidden_pair_flip": all(value >= 0.75 for value in full_hidden_pair.values()),
    }
    training_rows = [asdict(curriculum), asdict(alignment)]
    training_rows[0].pop("losses", None)
    loss_rows = [{"source": "curriculum", **row} for row in curriculum.losses] + [{"source": "full_hidden_alignment", **row} for row in alignment.losses]
    centroid_stability_rows = [
        {"metric": "before_fixed_average", "value": before_fixed_average},
        {"metric": "after_fixed_average", "value": after_fixed_average},
        {"metric": "improvement", "value": after_fixed_average - before_fixed_average},
    ]
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "final_projected_accuracy": final_projected,
        "final_fixed_centroid_accuracy": final_fixed,
        "projected_full_hidden_accuracy": projected_full,
        "fixed_centroid_average_before": before_fixed_average,
        "fixed_centroid_average_after": after_fixed_average,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_multiclass_repair_planning": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "full_hidden_centroid_metrics.csv", fixed_rows_all)
    _write_csv(output_path / "projected_full_hidden_metrics.csv", projected_full_rows)
    _write_csv(output_path / "projected_readout_metrics.csv", projected_rows_all)
    _write_csv(output_path / "raw_readout_metrics.csv", raw_rows_all)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "pair_flip_metrics.csv", pair_rows_all)
    _write_csv(output_path / "centroid_stability.csv", centroid_stability_rows)
    _write_csv(output_path / "rehearsal_retention.csv", [{"diagnostic_mode": mode, "projected_accuracy": final_projected[mode]} for mode in active_modes])
    _write_csv(output_path / "trace_contribution.csv", trace_rows_all)
    _json_dump(output_path / "resource_usage.json", [{"rss": psutil.Process().memory_info().rss, "seconds": time.perf_counter() - started}])
    _json_dump(output_path / "failure_cases.json", failure_cases)
    return summary
