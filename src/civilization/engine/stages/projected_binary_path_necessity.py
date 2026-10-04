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
import torch.nn.functional as F

from civilization.research.torch_line.model import CivilizationAblationConfig

from ..adapter import CivilizationAdapterConfig, PathSpecificCivilizationAdapter, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors, score_answer_options
from .binary_path_diagnostic_data import (
    BINARY_BASE_DIAGNOSTIC_MODES,
    BINARY_DIAGNOSTIC_MODES,
    BinaryDiagnosticPair,
    assert_binary_pairs_valid,
    build_binary_path_diagnostic_pairs,
    build_combined_binary_path_diagnostic_pairs,
    split_binary_pairs,
)
from .binary_path_diagnostic_training import _deterministic_binary_pair_sequence, _residual_parameters
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_training import _answer_scores, _capture_updates, _margin_loss
from .hidden_states import last_non_padding_pool


PROBE_LAYER = 28
EVALUATION_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_state_path",
    "no_rule_path",
    "empty_context",
    "wrong_context",
    "counterfactual_context",
)


@dataclass
class ProjectedNecessityTrainingResult:
    mode: str
    seed: int
    target_layers: tuple[int, ...]
    stage_a_steps: int
    stage_b_steps: int
    stage_c_steps: int
    losses: list[dict[str, float | str]]
    loss_decreased: dict[str, bool]
    adapter_parameter_count: int
    projector_parameter_count: int
    qwen_parameter_count: int
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    stage_a_checkpoint_path: str
    stage_b_checkpoint_path: str
    stage_c_checkpoint_path: str


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


def _centroid_predictions(train_vectors: np.ndarray, train_targets: list[int], test_vectors: np.ndarray) -> list[int]:
    labels = np.array(train_targets)
    centroids = []
    for option_id in (0, 1):
        rows = train_vectors[labels == option_id]
        if len(rows) == 0:
            raise ValueError("missing train centroid for binary option")
        centroids.append(rows.mean(axis=0))
    matrix = np.stack(centroids)
    normalized_test = test_vectors / np.clip(np.linalg.norm(test_vectors, axis=1, keepdims=True), 1e-8, None)
    normalized_centroids = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    return np.argmax(normalized_test @ normalized_centroids.T, axis=1).tolist()


def _window_decreased(rows: list[dict[str, Any]], key: str, stage: str | None = None) -> bool:
    values = [
        float(row[key])
        for row in rows
        if key in row and (stage is None or row.get("stage") == stage)
    ]
    if len(values) < 2:
        return False
    window = max(1, min(10, len(values) // 3))
    return sum(values[-window:]) / window < sum(values[:window]) / window


def _encode_samples(backend: Qwen3Backend, samples: list, max_length: int) -> dict[str, torch.Tensor]:
    encoded, truncations = backend.encode(
        [qwen_text_for_sample(sample) for sample in samples],
        max_length=max_length,
    )
    if truncations:
        raise ValueError("projected binary path diagnostic sample was truncated")
    return {name: tensor.to(backend.device) for name, tensor in encoded.items()}


def _context_for_eval(
    context_encoder: FrozenQwenContextEncoder,
    samples: list,
    attention_mask: torch.Tensor,
    eval_mode: str,
):
    ablation = None
    adapter_enabled = True
    force_zero = False
    context_mode = "full"
    if eval_mode == "adapter_disabled":
        adapter_enabled = False
    elif eval_mode == "zero_scale":
        force_zero = True
    elif eval_mode == "no_memory_path":
        ablation = CivilizationAblationConfig(use_memory_path=False)
    elif eval_mode == "no_state_path":
        ablation = CivilizationAblationConfig(use_state_path=False)
    elif eval_mode == "no_rule_path":
        ablation = CivilizationAblationConfig(use_rule_path=False)
    elif eval_mode == "empty_context":
        context_mode = "empty"
    elif eval_mode == "wrong_context":
        context_mode = "wrong"
    elif eval_mode in {"full", "counterfactual_context"}:
        context_mode = "full"
    else:
        raise ValueError(f"unsupported eval_mode: {eval_mode}")
    return context_encoder.build_context(
        samples,
        attention_mask,
        ablation_config=ablation,
        adapter_enabled=adapter_enabled,
        force_zero_scale=force_zero,
        context_mode=context_mode,
    )


def _forward_vectors(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    context_encoder: FrozenQwenContextEncoder,
    samples: list,
    max_length: int,
    eval_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    encoded = _encode_samples(backend, samples, max_length)
    disabled_context = context_encoder.build_context(
        samples,
        encoded["attention_mask"],
        adapter_enabled=False,
        context_mode="full",
    )
    with torch.no_grad():
        disabled = model(encoded, disabled_context)
    context = _context_for_eval(context_encoder, samples, encoded["attention_mask"], eval_mode)
    output = model(encoded, context)
    full_hidden = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    disabled_hidden = last_non_padding_pool(disabled.hidden_states[-1], disabled.attention_mask).float()
    delta = full_hidden - disabled_hidden
    trace_rows: list[dict[str, Any]] = []
    for layer, trace in output.traces.items():
        trace_rows.append(
            {
                "layer": layer,
                "memory_delta_norm": trace.memory_delta_norm,
                "rule_delta_norm": trace.rule_delta_norm,
                "state_delta_norm": trace.state_delta_norm,
                "base_delta_norm": trace.base_delta_norm,
                "memory_residual_scale": trace.memory_residual_scale,
                "rule_residual_scale": trace.rule_residual_scale,
                "path_dominance_ratio": trace.path_dominance_ratio,
                "hidden_norm_ratio": trace.hidden_norm_ratio,
                "path_specific_adapter_version": trace.path_specific_adapter_version,
            }
        )
    return full_hidden, delta, trace_rows


def _pair_forward(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    context_encoder: FrozenQwenContextEncoder,
    pair: BinaryDiagnosticPair,
    max_length: int,
    eval_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [pair.full_sample, pair.counterfactual_sample]
    full_hidden, delta, _trace_rows = _forward_vectors(
        backend,
        model,
        context_encoder,
        samples,
        max_length,
        eval_mode,
    )
    return full_hidden, delta


def _save_projected_checkpoint(
    path: Path,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "training_metadata": metadata,
        },
        path,
    )


def _load_projected_checkpoint(
    path: str,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.adapters.load_state_dict(payload["adapter_state_dict"])
    projector.load_state_dict(payload["projector_state_dict"])


def _train_projected_necessity(
    backend: Qwen3Backend,
    train_pairs: list[BinaryDiagnosticPair],
    mode: str,
    output_dir: Path,
    seed: int,
    target_layers: tuple[int, ...],
    stage_a_steps: int,
    stage_b_steps: int,
    stage_c_steps: int,
    max_length: int,
    learning_rate: float,
) -> tuple[Qwen3MultiAdapterModel, PathReadoutProjector, ProjectedNecessityTrainingResult]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_model(backend, target_layers)
    projector = PathReadoutProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors = torch.tensor(
        build_answer_option_vectors(backend, train_pairs[0].answer_options),
        dtype=torch.float32,
        device=backend.device,
    )
    residual_parameters = _residual_parameters(model)
    residual_ids = {id(parameter) for parameter in residual_parameters}
    adapter_parameters = [
        parameter
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
        if id(parameter) not in residual_ids
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
    total_steps = stage_a_steps + stage_b_steps + stage_c_steps
    sequence = _deterministic_binary_pair_sequence(train_pairs, total_steps, seed)
    losses: list[dict[str, float | str]] = []
    checkpoint_dir = output_dir / "checkpoints" / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a_path = checkpoint_dir / f"projected_{mode}_stage_a_seed_{seed}.pt"
    stage_b_path = checkpoint_dir / f"projected_{mode}_stage_b_seed_{seed}.pt"
    stage_c_path = checkpoint_dir / f"projected_{mode}_stage_c_seed_{seed}.pt"

    for step, pair in enumerate(sequence):
        if step < stage_a_steps:
            stage = "stage_a_projector_only"
            optimizer = projector_optimizer
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        elif step < stage_a_steps + stage_b_steps:
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
        full_hidden, delta = _pair_forward(backend, model, context_encoder, pair, max_length, "full")
        targets = torch.tensor(
            [pair.expected_full_option_id, pair.expected_counterfactual_option_id],
            dtype=torch.long,
            device=backend.device,
        )
        projected = projector(delta)
        projected_scores = _answer_scores(projected, option_vectors)
        answer_losses = []
        margins = []
        for index, target in enumerate(targets.tolist()):
            loss, margin = _margin_loss(projected_scores[index : index + 1], target, 0.20)
            answer_losses.append(loss)
            margins.append(margin)
        answer_loss = torch.stack(answer_losses).mean()
        pair_flip_loss = torch.relu(
            (
                F.normalize(projected[0:1], dim=-1)
                * F.normalize(projected[1:2], dim=-1)
            ).sum(dim=-1)
            - 0.80
        ).mean()
        centroid_loss = torch.zeros((), device=backend.device)
        if stage == "stage_c_centroid_regularization":
            full_scores = _answer_scores(full_hidden, option_vectors)
            centroid_losses = []
            for index, target in enumerate(targets.tolist()):
                loss, _margin = _margin_loss(full_scores[index : index + 1], target, 0.10)
                centroid_losses.append(loss)
            centroid_loss = torch.stack(centroid_losses).mean()
        total = answer_loss + pair_flip_loss + 0.05 * centroid_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + list(model.parameters()), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": float(step),
                "stage": stage,
                "total_loss": float(total.detach().cpu()),
                "projected_answer_loss": float(answer_loss.detach().cpu()),
                "pair_flip_loss": float(pair_flip_loss.detach().cpu()),
                "fixed_centroid_loss": float(centroid_loss.detach().cpu()),
                "projected_margin": float(torch.stack(margins).mean().detach().cpu()),
            }
        )
        if step + 1 == stage_a_steps:
            _save_projected_checkpoint(stage_a_path, model, projector, {"stage": stage, "mode": mode, "seed": seed})
        if step + 1 == stage_a_steps + stage_b_steps:
            _save_projected_checkpoint(stage_b_path, model, projector, {"stage": stage, "mode": mode, "seed": seed})
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    _save_projected_checkpoint(stage_c_path, model, projector, {"stage": "stage_c_centroid_regularization", "mode": mode, "seed": seed})
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    trainable_qwen = sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad)
    result = ProjectedNecessityTrainingResult(
        mode=mode,
        seed=seed,
        target_layers=target_layers,
        stage_a_steps=stage_a_steps,
        stage_b_steps=stage_b_steps,
        stage_c_steps=stage_c_steps,
        losses=losses,
        loss_decreased={
            "projected_answer_loss": _window_decreased(losses, "projected_answer_loss"),
            "pair_flip_loss": _window_decreased(losses, "pair_flip_loss"),
            "fixed_centroid_loss": _window_decreased(losses, "fixed_centroid_loss", "stage_c_centroid_regularization"),
        },
        adapter_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        projector_parameter_count=sum(parameter.numel() for parameter in projector.parameters()),
        qwen_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters()),
        qwen_trainable_parameter_count=trainable_qwen,
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=any(
            id(parameter) in qwen_ids
            for parameter in list(model.parameters()) + list(projector.parameters())
        ),
        stage_a_checkpoint_path=str(stage_a_path),
        stage_b_checkpoint_path=str(stage_b_path),
        stage_c_checkpoint_path=str(stage_c_path),
    )
    return model, projector, result


def _evaluate_projected_matrix(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    projector: PathReadoutProjector,
    train_pairs: list[BinaryDiagnosticPair],
    test_pairs: list[BinaryDiagnosticPair],
    mode: str,
    stage: str,
    max_length: int,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors_np = build_answer_option_vectors(backend, test_pairs[0].answer_options)
    train_targets = _targets_for_pairs(train_pairs)
    test_targets = _targets_for_pairs(test_pairs)
    train_vectors = []
    with torch.no_grad():
        for pair in train_pairs:
            samples = [pair.full_sample, pair.counterfactual_sample]
            full_hidden, _delta, _trace = _forward_vectors(backend, model, context_encoder, samples, max_length, "full")
            train_vectors.append(full_hidden.float().cpu())
    train_matrix = torch.cat(train_vectors, dim=0).numpy()

    projected_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    full_projected_accuracy = 0.0
    wrong_projected_accuracy = 0.0
    for eval_mode in evaluation_modes:
        eval_pairs = test_pairs
        eval_samples = _samples_for_pairs(eval_pairs)
        if eval_mode == "counterfactual_context":
            eval_samples = [sample for pair in eval_pairs for sample in (pair.counterfactual_sample, pair.full_sample)]
        with torch.no_grad():
            full_hidden, delta, traces = _forward_vectors(
                backend,
                model,
                context_encoder,
                eval_samples,
                max_length,
                "full" if eval_mode == "counterfactual_context" else eval_mode,
            )
            projected = projector(delta)
        raw_matrix = delta.float().cpu().numpy()
        projected_matrix = projected.float().cpu().numpy()
        full_matrix = full_hidden.float().cpu().numpy()
        if not np.isfinite(raw_matrix).all() or not np.isfinite(projected_matrix).all() or not np.isfinite(full_matrix).all():
            failure_cases.append({"mode": mode, "stage": stage, "eval_mode": eval_mode, "type": "nan_inf"})
        raw = score_answer_options(raw_matrix, eval_samples, option_vectors_np, test_pairs[0].answer_options)
        projected_score = score_answer_options(projected_matrix, eval_samples, option_vectors_np, test_pairs[0].answer_options)
        fixed_predictions = _centroid_predictions(train_matrix, train_targets, full_matrix)
        fixed_accuracy = sum(
            prediction == target
            for prediction, target in zip(fixed_predictions, test_targets, strict=True)
        ) / len(test_targets)
        if eval_mode == "full":
            full_projected_accuracy = projected_score.accuracy
        if eval_mode == "wrong_context":
            wrong_projected_accuracy = projected_score.accuracy
        raw_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                "accuracy": raw.accuracy,
                "macro_accuracy": raw.macro_accuracy,
                "mean_margin": raw.mean_margin,
            }
        )
        projected_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                "accuracy": projected_score.accuracy,
                "macro_accuracy": projected_score.macro_accuracy,
                "mean_margin": projected_score.mean_margin,
            }
        )
        fixed_rows.append(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                "accuracy": fixed_accuracy,
            }
        )
        trace_rows.extend(
            {
                "diagnostic_mode": mode,
                "stage": stage,
                "eval_mode": eval_mode,
                **trace,
            }
            for trace in traces
        )
        if eval_mode == "full":
            for pair_index, pair in enumerate(test_pairs):
                first = pair_index * 2
                second = first + 1
                pair_rows.append(
                    {
                        "diagnostic_mode": mode,
                        "stage": stage,
                        "pair_type": pair.pair_type,
                        "pair_id": pair.pair_id,
                        "projected_pair_success": bool(
                            projected_score.rows[first]["correct"] and projected_score.rows[second]["correct"]
                        ),
                        "raw_pair_success": bool(raw.rows[first]["correct"] and raw.rows[second]["correct"]),
                        "fixed_pair_success": bool(
                            fixed_predictions[first] == test_targets[first]
                            and fixed_predictions[second] == test_targets[second]
                        ),
                    }
                )
    projected_rows.append(
        {
            "diagnostic_mode": mode,
            "stage": stage,
            "eval_mode": "wrong_context_drop",
            "accuracy": full_projected_accuracy - wrong_projected_accuracy,
            "macro_accuracy": "",
            "mean_margin": "",
        }
    )
    return projected_rows, raw_rows, fixed_rows, pair_rows, trace_rows, failure_cases


def run_qwen3_projected_binary_path_necessity(
    output_dir: str | Path = "artifacts/civilization/projected_binary_path_necessity",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    modes: tuple[str, ...] = BINARY_DIAGNOSTIC_MODES,
    pairs_per_mode: int = 120,
    train_pairs: int = 80,
    held_out_pairs: int = 40,
    stage_a_steps: int = 40,
    stage_b_steps: int = 80,
    stage_c_steps: int = 20,
    max_length: int = 64,
    learning_rate: float = 3e-4,
    target_layers: tuple[int, ...] = (16, 24),
    preferred_device: str | None = None,
    evaluation_modes: tuple[str, ...] = EVALUATION_MODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    grouped = build_binary_path_diagnostic_pairs(pairs_per_mode=pairs_per_mode, seed=seed, max_seq_len=max_length)
    grouped["combined_binary_diagnostic"] = build_combined_binary_path_diagnostic_pairs(grouped, pairs_per_mode)
    assert_binary_pairs_valid([pair for mode_pairs in grouped.values() for pair in mode_pairs if pair.mode in BINARY_BASE_DIAGNOSTIC_MODES])

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []

    for mode in modes:
        print(f"projected_binary_path_necessity_mode_start mode={mode}", flush=True)
        mode_started = time.perf_counter()
        train, test = split_binary_pairs(grouped[mode], train_pairs, held_out_pairs)
        model, projector, training = _train_projected_necessity(
            backend=backend,
            train_pairs=train,
            mode=mode,
            output_dir=output_path,
            seed=seed,
            target_layers=target_layers,
            stage_a_steps=stage_a_steps,
            stage_b_steps=stage_b_steps,
            stage_c_steps=stage_c_steps,
            max_length=max_length,
            learning_rate=learning_rate,
        )
        training_row = asdict(training)
        losses = training_row.pop("losses")
        training_rows.append(training_row)
        loss_rows.extend({"diagnostic_mode": mode, **loss} for loss in losses)
        mode_stage_metrics: dict[str, dict[str, float]] = {}
        stage_checkpoints = {
            "stage_b": training.stage_b_checkpoint_path,
            "stage_c": training.stage_c_checkpoint_path,
        }
        for stage_name, checkpoint_path in stage_checkpoints.items():
            _load_projected_checkpoint(checkpoint_path, model, projector)
            stage_projected, stage_raw, stage_fixed, stage_pairs, stage_traces, stage_failures = _evaluate_projected_matrix(
                backend=backend,
                model=model,
                projector=projector,
                train_pairs=train,
                test_pairs=test,
                mode=mode,
                stage=stage_name,
                max_length=max_length,
                evaluation_modes=evaluation_modes,
            )
            projected_rows.extend(stage_projected)
            raw_rows.extend(stage_raw)
            fixed_rows.extend(stage_fixed)
            pair_rows.extend(stage_pairs)
            trace_rows.extend(stage_traces)
            failure_cases.extend(stage_failures)
            mode_stage_metrics[stage_name] = {
                "projected_accuracy": _mean(stage_projected, lambda row: row["eval_mode"] == "full"),
                "fixed_accuracy": _mean(stage_fixed, lambda row: row["eval_mode"] == "full"),
            }
        stage_rows.append(
            {
                "diagnostic_mode": mode,
                "stage_b_projected_accuracy": mode_stage_metrics["stage_b"]["projected_accuracy"],
                "stage_c_projected_accuracy": mode_stage_metrics["stage_c"]["projected_accuracy"],
                "projected_regression": mode_stage_metrics["stage_b"]["projected_accuracy"] - mode_stage_metrics["stage_c"]["projected_accuracy"],
                "stage_b_fixed_accuracy": mode_stage_metrics["stage_b"]["fixed_accuracy"],
                "stage_c_fixed_accuracy": mode_stage_metrics["stage_c"]["fixed_accuracy"],
                "fixed_improvement": mode_stage_metrics["stage_c"]["fixed_accuracy"] - mode_stage_metrics["stage_b"]["fixed_accuracy"],
            }
        )
        for eval_mode in ("no_memory_path", "no_state_path", "no_rule_path", "empty_context", "counterfactual_context"):
            full = _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "full")
            ablated = _mean(projected_rows, lambda row, current=mode, current_eval=eval_mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == current_eval)
            ablation_rows.append(
                {
                    "diagnostic_mode": mode,
                    "eval_mode": eval_mode,
                    "readout": "projected_delta",
                    "full_accuracy": full,
                    "ablated_accuracy": ablated,
                    "absolute_drop": full - ablated,
                }
            )
        full = _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "full")
        wrong = _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "wrong_context")
        wrong_rows.append(
            {
                "diagnostic_mode": mode,
                "readout": "projected_delta",
                "full_accuracy": full,
                "wrong_context_accuracy": wrong,
                "wrong_context_drop": full - wrong,
            }
        )
        resource_rows.append(
            {
                "diagnostic_mode": mode,
                "seconds": time.perf_counter() - mode_started,
                "rss": psutil.Process().memory_info().rss,
                "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
            }
        )
        print(f"projected_binary_path_necessity_mode_complete mode={mode} seconds={time.perf_counter() - mode_started:.2f}", flush=True)
        del model, projector
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    drops = {(row["diagnostic_mode"], row["eval_mode"]): row["absolute_drop"] for row in ablation_rows}
    stage_c_projected = {
        mode: _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "full")
        for mode in modes
    }
    stage_c_raw = {
        mode: _mean(raw_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "full")
        for mode in modes
    }
    stage_c_fixed = {
        mode: _mean(fixed_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "full")
        for mode in modes
    }
    projected_pair_success = {
        mode: _mean(pair_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c", field="projected_pair_success")
        for mode in modes
    }
    stage_metrics = {row["diagnostic_mode"]: row for row in stage_rows}
    max_hidden_norm = max((float(row["hidden_norm_ratio"]) for row in trace_rows if row["hidden_norm_ratio"] is not None), default=1.0)
    stage_gates = {
        "qwen_frozen": all(
            row["qwen_trainable_parameter_count"] == 0
            and row["qwen_gradients_present"] == 0
            and not row["optimizer_contains_qwen_parameters"]
            for row in training_rows
        ),
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(
            abs(
                _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "adapter_disabled")
                - _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == "stage_c" and row["eval_mode"] == "zero_scale")
            )
            <= 1e-9
            for mode in modes
        ),
        "hidden_norm_ratio": max_hidden_norm <= 2.0,
        "memory_projected_answer_accuracy": stage_c_projected.get("memory_only_diagnostic", 0.0) >= 0.85,
        "memory_pair_flip": projected_pair_success.get("memory_only_diagnostic", 0.0) >= 0.80,
        "memory_wrong_context_drop": next((row["wrong_context_drop"] for row in wrong_rows if row["diagnostic_mode"] == "memory_only_diagnostic"), 0.0) >= 0.20,
        "memory_path_drop": drops.get(("memory_only_diagnostic", "no_memory_path"), 0.0) >= 0.25,
        "memory_rule_control": drops.get(("memory_only_diagnostic", "no_rule_path"), 0.0) <= 0.10,
        "rule_projected_answer_accuracy": stage_c_projected.get("rule_only_diagnostic", 0.0) >= 0.85,
        "rule_pair_flip": projected_pair_success.get("rule_only_diagnostic", 0.0) >= 0.80,
        "rule_wrong_context_drop": next((row["wrong_context_drop"] for row in wrong_rows if row["diagnostic_mode"] == "rule_only_diagnostic"), 0.0) >= 0.20,
        "rule_path_drop": drops.get(("rule_only_diagnostic", "no_rule_path"), 0.0) >= 0.25,
        "rule_memory_control": drops.get(("rule_only_diagnostic", "no_memory_path"), 0.0) <= 0.10,
        "conflict_projected_resolution": stage_c_projected.get("memory_rule_conflict_diagnostic", 0.0) >= 0.80,
        "conflict_wrong_context_drop": next((row["wrong_context_drop"] for row in wrong_rows if row["diagnostic_mode"] == "memory_rule_conflict_diagnostic"), 0.0) >= 0.20,
        "conflict_rule_path_drop": drops.get(("memory_rule_conflict_diagnostic", "no_rule_path"), 0.0) >= 0.20,
        "stage_c_fixed_improvement": all(row["fixed_improvement"] >= 0.05 for row in stage_rows),
        "stage_c_projected_retention": all(row["projected_regression"] <= 0.05 for row in stage_rows),
    }
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "readout": "projected_delta",
        "modes": list(modes),
        "pairs_per_mode": pairs_per_mode,
        "train_pairs": train_pairs,
        "held_out_pairs": held_out_pairs,
        "stage_a_steps": stage_a_steps,
        "stage_b_steps": stage_b_steps,
        "stage_c_steps": stage_c_steps,
        "projected_answer_accuracy": stage_c_projected,
        "raw_answer_accuracy": stage_c_raw,
        "fixed_centroid_accuracy": stage_c_fixed,
        "projected_pair_success": projected_pair_success,
        "stage_a_b_c_comparison": stage_metrics,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_multiclass_repair_planning": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", training_rows)
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "projected_readout_metrics.csv", projected_rows)
    _write_csv(output_path / "raw_readout_metrics.csv", raw_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "pair_flip_metrics.csv", pair_rows)
    _write_csv(output_path / "stage_a_b_c_comparison.csv", stage_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "dataset_mode": "projected_binary_path_necessity_v1",
            "modes": list(modes),
            "pairs_per_mode": pairs_per_mode,
            "train_pairs": train_pairs,
            "held_out_pairs": held_out_pairs,
            "evaluation_modes": list(evaluation_modes),
        },
    )
    return summary
