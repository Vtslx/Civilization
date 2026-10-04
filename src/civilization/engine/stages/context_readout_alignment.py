from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
import torch.nn.functional as F

from ..adapter import CivilizationAdapterConfig, PathSpecificCivilizationAdapter, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .binary_path_diagnostic_data import (
    BINARY_BASE_DIAGNOSTIC_MODES,
    BinaryDiagnosticPair,
    assert_binary_pairs_valid,
    build_binary_path_diagnostic_pairs,
    split_binary_pairs,
)
from .binary_path_diagnostic_training import _deterministic_binary_pair_sequence, _residual_parameters
from .evidence_answer_training import _answer_scores, _capture_updates, _margin_loss
from .hidden_states import last_non_padding_pool


PROBE_LAYER = 28


@dataclass
class ReadoutAlignmentResult:
    mode: str
    seed: int
    target_layers: tuple[int, ...]
    projector_steps: int
    adapter_steps: int
    centroid_steps: int
    losses: list[dict[str, float | str]]
    loss_decreased: dict[str, bool]
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    adapter_checkpoint_path: str
    projector_checkpoint_path: str


class PathReadoutProjector(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(vector.float()))


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _samples_for_pairs(pairs: list[BinaryDiagnosticPair]):
    return [sample for pair in pairs for sample in (pair.full_sample, pair.counterfactual_sample)]


def _targets_for_pairs(pairs: list[BinaryDiagnosticPair]) -> list[int]:
    result: list[int] = []
    for pair in pairs:
        result.extend((pair.expected_full_option_id, pair.expected_counterfactual_option_id))
    return result


def _make_model(backend: Qwen3Backend, target_layers: tuple[int, ...]) -> Qwen3MultiAdapterModel:
    return Qwen3MultiAdapterModel(
        backend,
        {
            layer: PathSpecificCivilizationAdapter(CivilizationAdapterConfig(target_layer=layer))
            for layer in target_layers
        },
    )


def _encode_pair(
    backend: Qwen3Backend,
    pair: BinaryDiagnosticPair,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list]:
    samples = [pair.full_sample, pair.counterfactual_sample]
    encoded, truncations = backend.encode(
        [qwen_text_for_sample(sample) for sample in samples],
        max_length=max_length,
    )
    if truncations:
        raise ValueError("context/readout diagnostic sample was truncated")
    return {name: tensor.to(backend.device) for name, tensor in encoded.items()}, samples


def _context_vectors(
    context_encoder: FrozenQwenContextEncoder,
    samples: list,
    attention_mask: torch.Tensor,
):
    context = context_encoder.build_context(samples, attention_mask, context_mode="full")
    memory = context.memory_vectors.float()
    rule = context.rule_vectors.float()
    memory_mask = context.memory_mask
    rule_mask = context.rule_mask
    memory_vector = (
        (memory * memory_mask.to(memory.dtype).unsqueeze(-1)).sum(dim=1)
        / memory_mask.to(memory.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        if memory.shape[1]
        else memory.new_zeros((memory.shape[0], memory.shape[-1]))
    )
    rule_vector = (
        (rule * rule_mask.to(rule.dtype).unsqueeze(-1)).sum(dim=1)
        / rule_mask.to(rule.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        if rule.shape[1]
        else rule.new_zeros((rule.shape[0], rule.shape[-1]))
    )
    return context, memory_vector, rule_vector


def _pair_forward(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    context_encoder: FrozenQwenContextEncoder,
    pair: BinaryDiagnosticPair,
    max_length: int,
):
    encoded, samples = _encode_pair(backend, pair, max_length)
    with torch.inference_mode():
        baseline = backend.inference_forward(encoded)
    context, memory_vector, rule_vector = _context_vectors(context_encoder, samples, encoded["attention_mask"])
    output = model(encoded, context)
    disabled_context = context_encoder.build_context(
        samples,
        encoded["attention_mask"],
        adapter_enabled=False,
        context_mode="full",
    )
    with torch.no_grad():
        disabled = model(encoded, disabled_context)
    full_hidden = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    disabled_hidden = last_non_padding_pool(disabled.hidden_states[-1], disabled.attention_mask).float()
    baseline_hidden = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
    delta = full_hidden - disabled_hidden
    raw_update = _capture_updates(model, output.attention_mask)[-1]
    return {
        "encoded": encoded,
        "samples": samples,
        "output": output,
        "baseline": baseline,
        "full_hidden": full_hidden,
        "baseline_hidden": baseline_hidden,
        "disabled_hidden": disabled_hidden,
        "delta": delta,
        "raw_update": raw_update,
        "memory_vector": memory_vector,
        "rule_vector": rule_vector,
    }


def _context_accuracy(
    context_vectors: torch.Tensor,
    targets: torch.Tensor,
    option_vectors: torch.Tensor,
) -> tuple[float, float]:
    scores = _answer_scores(context_vectors, option_vectors)
    predictions = scores.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean()
    margins = []
    for index, target in enumerate(targets.tolist()):
        _loss, margin = _margin_loss(scores[index : index + 1], target, 0.0)
        margins.append(margin)
    return float(accuracy.detach().cpu()), float(torch.stack(margins).mean().detach().cpu())


def _centroid_predictions(train_vectors: np.ndarray, train_targets: list[int], test_vectors: np.ndarray) -> list[int]:
    centroids = []
    labels = np.array(train_targets)
    for option_id in (0, 1):
        rows = train_vectors[labels == option_id]
        if len(rows) == 0:
            raise ValueError("missing centroid rows")
        centroids.append(rows.mean(axis=0))
    matrix = np.stack(centroids)
    norm_test = test_vectors / np.clip(np.linalg.norm(test_vectors, axis=1, keepdims=True), 1e-8, None)
    norm_centroids = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    return np.argmax(norm_test @ norm_centroids.T, axis=1).tolist()


def _evaluate_readouts(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    train_pairs: list[BinaryDiagnosticPair],
    test_pairs: list[BinaryDiagnosticPair],
    mode: str,
    stage: str,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context_encoder = FrozenQwenContextEncoder(backend)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    option_vectors_np = build_answer_option_vectors(backend, test_pairs[0].answer_options)
    option_vectors = torch.tensor(option_vectors_np, dtype=torch.float32, device=backend.device)
    train_full_vectors = []
    train_targets = _targets_for_pairs(train_pairs)
    with torch.no_grad():
        for pair in train_pairs:
            result = _pair_forward(backend, model, context_encoder, pair, max_length)
            train_full_vectors.append(result["full_hidden"].float().cpu())
    train_matrix = torch.cat(train_full_vectors, dim=0).numpy()

    raw_vectors = []
    projected_vectors = []
    full_vectors = []
    samples = []
    targets = []
    context_memory_vectors = []
    context_rule_vectors = []
    with torch.no_grad():
        for pair in test_pairs:
            result = _pair_forward(backend, model, context_encoder, pair, max_length)
            raw_vectors.append(result["delta"].float().cpu())
            projected_vectors.append(projector(result["delta"]).float().cpu())
            full_vectors.append(result["full_hidden"].float().cpu())
            context_memory_vectors.append(result["memory_vector"].float().cpu())
            context_rule_vectors.append(result["rule_vector"].float().cpu())
            samples.extend(result["samples"])
            targets.extend((pair.expected_full_option_id, pair.expected_counterfactual_option_id))
    target_tensor = torch.tensor(targets, dtype=torch.long, device=backend.device)
    raw_matrix = torch.cat(raw_vectors, dim=0).numpy()
    projected_matrix = torch.cat(projected_vectors, dim=0).numpy()
    full_matrix = torch.cat(full_vectors, dim=0).numpy()
    memory_context = torch.cat(context_memory_vectors, dim=0).to(backend.device)
    rule_context = torch.cat(context_rule_vectors, dim=0).to(backend.device)
    if not np.isfinite(raw_matrix).all() or not np.isfinite(projected_matrix).all() or not np.isfinite(full_matrix).all():
        failures.append({"mode": mode, "stage": stage, "type": "nan_inf"})
    raw = score_answer_options(raw_matrix, samples, option_vectors_np, test_pairs[0].answer_options)
    projected = score_answer_options(projected_matrix, samples, option_vectors_np, test_pairs[0].answer_options)
    fixed_predictions = _centroid_predictions(train_matrix, train_targets, full_matrix)
    fixed_accuracy = sum(pred == target for pred, target in zip(fixed_predictions, targets, strict=True)) / len(targets)
    memory_context_accuracy, memory_context_margin = _context_accuracy(memory_context, target_tensor, option_vectors)
    rule_context_accuracy, rule_context_margin = _context_accuracy(rule_context, target_tensor, option_vectors)
    for readout_name, metric in (("raw_delta", raw), ("projected_delta", projected)):
        rows.append(
            {
                "mode": mode,
                "stage": stage,
                "readout": readout_name,
                "accuracy": metric.accuracy,
                "macro_accuracy": metric.macro_accuracy,
                "mean_margin": metric.mean_margin,
            }
        )
    rows.append(
        {
            "mode": mode,
            "stage": stage,
            "readout": "fixed_centroid",
            "accuracy": fixed_accuracy,
            "macro_accuracy": "",
            "mean_margin": "",
        }
    )
    rows.append(
        {
            "mode": mode,
            "stage": stage,
            "readout": "memory_context_to_answer",
            "accuracy": memory_context_accuracy,
            "macro_accuracy": "",
            "mean_margin": memory_context_margin,
        }
    )
    rows.append(
        {
            "mode": mode,
            "stage": stage,
            "readout": "rule_context_to_answer",
            "accuracy": rule_context_accuracy,
            "macro_accuracy": "",
            "mean_margin": rule_context_margin,
        }
    )
    for pair_index, pair in enumerate(test_pairs):
        first = pair_index * 2
        second = first + 1
        pair_rows.append(
            {
                "mode": mode,
                "stage": stage,
                "pair_id": pair.pair_id,
                "raw_pair_success": raw.rows[first]["correct"] and raw.rows[second]["correct"],
                "projected_pair_success": projected.rows[first]["correct"] and projected.rows[second]["correct"],
                "fixed_pair_success": fixed_predictions[first] == targets[first] and fixed_predictions[second] == targets[second],
            }
        )
    return rows, pair_rows, failures


def _window_decreased(rows: list[dict[str, Any]], key: str) -> bool:
    values = [float(row[key]) for row in rows if key in row]
    if len(values) < 2:
        return False
    window = max(1, min(10, len(values) // 3))
    return sum(values[-window:]) / window < sum(values[:window]) / window


def _train_alignment(
    backend: Qwen3Backend,
    train_pairs: list[BinaryDiagnosticPair],
    mode: str,
    output_dir: Path,
    seed: int,
    target_layers: tuple[int, ...],
    projector_steps: int,
    adapter_steps: int,
    centroid_steps: int,
    max_length: int,
    learning_rate: float,
) -> tuple[Qwen3MultiAdapterModel, PathReadoutProjector, ReadoutAlignmentResult]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_model(backend, target_layers)
    projector = PathReadoutProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors = torch.tensor(build_answer_option_vectors(backend, train_pairs[0].answer_options), dtype=torch.float32, device=backend.device)
    sequence = _deterministic_binary_pair_sequence(train_pairs, projector_steps + adapter_steps + centroid_steps, seed)
    residual_parameters = _residual_parameters(model)
    adapter_parameters = [
        parameter
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
        if all(parameter is not residual for residual in residual_parameters)
    ]
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=0.01)
    joint_optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    losses: list[dict[str, float | str]] = []
    for step, pair in enumerate(sequence):
        if step < projector_steps:
            stage = "stage_a_projector_only"
            optimizer = projector_optimizer
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        elif step < projector_steps + adapter_steps:
            stage = "stage_b_adapter_projector"
            optimizer = joint_optimizer
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        else:
            stage = "stage_c_centroid_regularization"
            optimizer = joint_optimizer
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        result = _pair_forward(backend, model, context_encoder, pair, max_length)
        targets = torch.tensor([pair.expected_full_option_id, pair.expected_counterfactual_option_id], dtype=torch.long, device=backend.device)
        projected = projector(result["delta"])
        projected_scores = _answer_scores(projected, option_vectors)
        raw_scores = _answer_scores(result["delta"], option_vectors)
        projected_losses = []
        projected_margins = []
        raw_margins = []
        for index, target in enumerate(targets.tolist()):
            loss, margin = _margin_loss(projected_scores[index : index + 1], target, 0.2)
            _raw_loss, raw_margin = _margin_loss(raw_scores[index : index + 1], target, 0.0)
            projected_losses.append(loss)
            projected_margins.append(margin)
            raw_margins.append(raw_margin)
        answer_loss = torch.stack(projected_losses).mean()
        pair_loss = torch.relu(
            (
                F.normalize(projected[0:1], dim=-1)
                * F.normalize(projected[1:2], dim=-1)
            ).sum(dim=-1)
            - 0.80
        ).mean()
        centroid_loss = torch.zeros((), device=backend.device)
        if stage == "stage_c_centroid_regularization":
            full_scores = _answer_scores(result["full_hidden"], option_vectors)
            centroid_losses = []
            for index, target in enumerate(targets.tolist()):
                loss, _margin = _margin_loss(full_scores[index : index + 1], target, 0.1)
                centroid_losses.append(loss)
            centroid_loss = torch.stack(centroid_losses).mean()
        total = answer_loss + pair_loss + 0.05 * centroid_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + list(model.parameters()), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": float(step),
                "stage": stage,
                "total_loss": float(total.detach().cpu()),
                "projected_answer_loss": float(answer_loss.detach().cpu()),
                "pair_separation_loss": float(pair_loss.detach().cpu()),
                "centroid_loss": float(centroid_loss.detach().cpu()),
                "projected_margin": float(torch.stack(projected_margins).mean().detach().cpu()),
                "raw_margin": float(torch.stack(raw_margins).mean().detach().cpu()),
                "rss": float(psutil.Process().memory_info().rss),
            }
        )
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_checkpoint = checkpoint_dir / f"context_readout_{mode}_adapter_seed_{seed}.pt"
    projector_checkpoint = checkpoint_dir / f"context_readout_{mode}_projector_seed_{seed}.pt"
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "adapter_variant": "path_specific_v2",
            "target_layers": target_layers,
            "mode": mode,
        },
        adapter_checkpoint,
    )
    torch.save(
        {
            "projector_state_dict": projector.state_dict(),
            "mode": mode,
            "seed": seed,
        },
        projector_checkpoint,
    )
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    trainable_qwen = sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad)
    result = ReadoutAlignmentResult(
        mode=mode,
        seed=seed,
        target_layers=target_layers,
        projector_steps=projector_steps,
        adapter_steps=adapter_steps,
        centroid_steps=centroid_steps,
        losses=losses,
        loss_decreased={
            "projected_answer_loss": _window_decreased(losses, "projected_answer_loss"),
            "pair_separation_loss": _window_decreased(losses, "pair_separation_loss"),
            "centroid_loss": _window_decreased([row for row in losses if row["stage"] == "stage_c_centroid_regularization"], "centroid_loss"),
        },
        qwen_trainable_parameter_count=trainable_qwen,
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=any(
            id(parameter) in qwen_ids
            for parameter in list(model.parameters()) + list(projector.parameters())
        ),
        adapter_checkpoint_path=str(adapter_checkpoint),
        projector_checkpoint_path=str(projector_checkpoint),
    )
    return model, projector, result


def run_qwen3_context_readout_alignment(
    output_dir: str | Path = "artifacts/civilization/context_readout_alignment",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    modes: tuple[str, ...] = BINARY_BASE_DIAGNOSTIC_MODES,
    pairs_per_mode: int = 120,
    train_pairs: int = 80,
    held_out_pairs: int = 40,
    projector_steps: int = 40,
    adapter_steps: int = 80,
    centroid_steps: int = 20,
    max_length: int = 64,
    learning_rate: float = 3e-4,
    target_layers: tuple[int, ...] = (16, 24),
    preferred_device: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    grouped = build_binary_path_diagnostic_pairs(pairs_per_mode=pairs_per_mode, seed=seed, max_seq_len=max_length)
    assert_binary_pairs_valid([pair for mode_pairs in grouped.values() for pair in mode_pairs])
    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for mode in modes:
        started_rss = psutil.Process().memory_info().rss
        train, test = split_binary_pairs(grouped[mode], train_pairs, held_out_pairs)
        model, projector, training = _train_alignment(
            backend=backend,
            train_pairs=train,
            mode=mode,
            output_dir=output_path,
            seed=seed,
            target_layers=target_layers,
            projector_steps=projector_steps,
            adapter_steps=adapter_steps,
            centroid_steps=centroid_steps,
            max_length=max_length,
            learning_rate=learning_rate,
        )
        row = asdict(training)
        losses = row.pop("losses")
        training_rows.append(row)
        loss_rows.extend({"mode": mode, **loss} for loss in losses)
        metrics, pairs, failures = _evaluate_readouts(
            backend=backend,
            model=model,
            projector=projector,
            train_pairs=train,
            test_pairs=test,
            mode=mode,
            stage="final",
            max_length=max_length,
        )
        metric_rows.extend(metrics)
        pair_rows.extend(pairs)
        failure_cases.extend(failures)
        resource_rows.append(
            {
                "mode": mode,
                "rss_start": started_rss,
                "rss_end": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        del model, projector
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    def metric(mode: str, readout: str) -> float:
        rows = [row for row in metric_rows if row["mode"] == mode and row["readout"] == readout]
        return float(rows[0]["accuracy"]) if rows else 0.0

    projected = {mode: metric(mode, "projected_delta") for mode in modes}
    raw = {mode: metric(mode, "raw_delta") for mode in modes}
    fixed = {mode: metric(mode, "fixed_centroid") for mode in modes}
    context_accuracy = {
        mode: {
            "memory_context_to_answer": metric(mode, "memory_context_to_answer"),
            "rule_context_to_answer": metric(mode, "rule_context_to_answer"),
        }
        for mode in modes
    }
    conflict_pair_success = (
        sum(row["projected_pair_success"] for row in pair_rows if row["mode"] == "memory_rule_conflict_diagnostic")
        / max(1, sum(1 for row in pair_rows if row["mode"] == "memory_rule_conflict_diagnostic"))
    )
    stage_gates = {
        "qwen_frozen": all(row["qwen_trainable_parameter_count"] == 0 and row["qwen_gradients_present"] == 0 and not row["optimizer_contains_qwen_parameters"] for row in training_rows),
        "no_engineering_failures": not failure_cases,
        "memory_context_separable": context_accuracy.get("memory_only_diagnostic", {}).get("memory_context_to_answer", 0.0) >= 0.90,
        "rule_context_separable": context_accuracy.get("rule_only_diagnostic", {}).get("rule_context_to_answer", 0.0) >= 0.90,
        "memory_projected_readout": projected.get("memory_only_diagnostic", 0.0) >= 0.80,
        "rule_projected_readout": projected.get("rule_only_diagnostic", 0.0) >= 0.80,
        "conflict_projected_pair_success": conflict_pair_success >= 0.75,
        "fixed_centroid_recorded": all(value >= 0.0 for value in fixed.values()),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "modes": list(modes),
        "projected_accuracy": projected,
        "raw_accuracy": raw,
        "fixed_centroid_accuracy": fixed,
        "context_accuracy": context_accuracy,
        "conflict_projected_pair_success": conflict_pair_success,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_stage32_rerun": all(stage_gates.values()),
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "readout_metrics.csv", metric_rows)
    _write_csv(output_path / "pair_metrics.csv", pair_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "dataset_mode": "context_readout_alignment_v1",
            "modes": list(modes),
            "pairs_per_mode": pairs_per_mode,
            "train_pairs": train_pairs,
            "held_out_pairs": held_out_pairs,
        },
    )
    return summary
