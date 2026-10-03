from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from experiments.civilization_transformer_torch.model import CivilizationAblationConfig

from ..adapter import (
    CivilizationAdapter,
    CivilizationAdapterConfig,
    PathSpecificCivilizationAdapter,
    Qwen3MultiAdapterModel,
)
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import AdapterDiagnosticHeads, _kl_preservation
from .answer_option_readout import build_answer_option_vectors
from .binary_path_diagnostic_data import BinaryDiagnosticPair, BINARY_DIAGNOSTIC_MODES
from .evidence_answer_training import _answer_scores, _capture_updates, _margin_loss
from .hidden_states import last_non_padding_pool


@dataclass(frozen=True)
class BinaryPathDiagnosticLossWeights:
    binary_answer_margin: float = 1.50
    path_counterfactual_flip: float = 1.00
    path_ablation_drop: float = 1.00
    wrong_context_separation: float = 1.00
    logit_preservation: float = 0.10
    residual_budget: float = 0.10
    full_hidden_centroid_regularization: float = 0.10


@dataclass
class BinaryPathDiagnosticTrainingResult:
    mode: str
    seed: int
    target_layers: tuple[int, ...]
    stage_a_steps: int
    stage_b_steps: int
    losses: list[dict[str, float | str]]
    loss_decreased: dict[str, bool]
    adapter_parameter_count: int
    qwen_parameter_count: int
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    initial_fingerprint: dict[str, float | int]
    final_fingerprint: dict[str, float | int]
    fingerprint_unchanged: bool
    stage_a_checkpoint_path: str
    stage_b_checkpoint_path: str
    max_mps_allocated: int
    adapter_variant: str = "baseline_v1"


def _make_adapter(config: CivilizationAdapterConfig, adapter_variant: str):
    if adapter_variant == "baseline_v1":
        return CivilizationAdapter(config)
    if adapter_variant == "path_specific_v2":
        return PathSpecificCivilizationAdapter(config)
    raise ValueError(f"unsupported adapter_variant: {adapter_variant}")


def _residual_parameters(model: Qwen3MultiAdapterModel) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    for adapter in model.adapters.values():
        for name in (
            "base_residual_scale",
            "memory_residual_scale",
            "rule_residual_scale",
            "state_residual_scale",
            "residual_scale",
        ):
            parameter = getattr(adapter, name, None)
            if isinstance(parameter, torch.nn.Parameter) and all(parameter is not existing for existing in result):
                result.append(parameter)
    return result


def _window_mean(rows: list[dict[str, float | str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _checkpoint_payload(
    model: Qwen3MultiAdapterModel,
    heads: AdapterDiagnosticHeads,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "adapter_state_dict": model.adapters.state_dict(),
        "diagnostic_heads_state_dict": heads.state_dict(),
        "adapter_configs": {
            str(layer): model.adapters[str(layer)].config.to_dict()
            for layer in model.target_layers
        },
        "training_metadata": metadata,
    }


def _deterministic_binary_pair_sequence(
    pairs: list[BinaryDiagnosticPair],
    count: int,
    seed: int,
) -> list[BinaryDiagnosticPair]:
    if not pairs:
        raise ValueError("binary diagnostic pairs cannot be empty")
    rng = random.Random(seed)
    ordered = sorted(pairs, key=lambda pair: (pair.mode, pair.pair_type, pair.pair_id))
    result: list[BinaryDiagnosticPair] = []
    while len(result) < count:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:count]


def _path_ablation_for_pair(pair: BinaryDiagnosticPair) -> CivilizationAblationConfig:
    if pair.mode == "memory_only_diagnostic":
        return CivilizationAblationConfig(use_memory_path=False)
    if pair.mode in {"rule_only_diagnostic", "memory_rule_conflict_diagnostic"}:
        return CivilizationAblationConfig(use_rule_path=False)
    raise ValueError(f"unsupported diagnostic pair mode: {pair.mode}")


def _path_drop_loss(
    full_scores: torch.Tensor,
    ablated_scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    full_target = full_scores.gather(1, targets[:, None]).squeeze(1)
    ablated_target = ablated_scores.gather(1, targets[:, None]).squeeze(1)
    return torch.relu(margin + ablated_target - full_target).mean()


def _run_pair_forward(
    backend: Qwen3Backend,
    model: Qwen3MultiAdapterModel,
    context_encoder: FrozenQwenContextEncoder,
    pair: BinaryDiagnosticPair,
    max_length: int,
    ablation_config: CivilizationAblationConfig | None = None,
):
    samples = [pair.full_sample, pair.counterfactual_sample]
    encoded, truncations = backend.encode(
        [qwen_text_for_sample(sample) for sample in samples],
        max_length=max_length,
    )
    if truncations:
        raise ValueError("binary diagnostic sample was truncated")
    encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        baseline = backend.inference_forward(encoded)
    context = context_encoder.build_context(
        samples,
        encoded["attention_mask"],
        ablation_config=ablation_config,
        context_mode="full",
    )
    output = model(encoded, context)
    pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
    delta = pooled - baseline_pooled
    updates = _capture_updates(model, output.attention_mask)
    return encoded, baseline, output, pooled, delta, updates


def run_binary_path_diagnostic_training(
    backend: Qwen3Backend,
    train_pairs: list[BinaryDiagnosticPair],
    output_dir: str | Path,
    mode: str,
    seed: int = 202,
    target_layers: tuple[int, ...] = (16, 24),
    stage_a_steps: int = 80,
    stage_b_steps: int = 40,
    gradient_accumulation: int = 4,
    max_length: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    answer_margin: float = 0.20,
    wrong_separation_margin: float = 0.20,
    adapter_variant: str = "baseline_v1",
    weights: BinaryPathDiagnosticLossWeights | None = None,
) -> tuple[Qwen3MultiAdapterModel, AdapterDiagnosticHeads, BinaryPathDiagnosticTrainingResult]:
    if mode not in BINARY_DIAGNOSTIC_MODES:
        raise ValueError(f"unsupported diagnostic mode: {mode}")
    if mode != "combined_binary_diagnostic" and any(pair.mode != mode for pair in train_pairs):
        raise ValueError("diagnostic training mode received pairs from another mode")
    if mode == "combined_binary_diagnostic" and not {
        pair.mode for pair in train_pairs
    }.issubset({"memory_only_diagnostic", "rule_only_diagnostic", "memory_rule_conflict_diagnostic"}):
        raise ValueError("combined diagnostic received unsupported pair modes")
    if stage_a_steps <= 0 or stage_b_steps <= 0 or gradient_accumulation <= 0:
        raise ValueError("stage steps and gradient_accumulation must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    adapters = {
        layer: _make_adapter(CivilizationAdapterConfig(target_layer=layer), adapter_variant)
        for layer in target_layers
    }
    model = Qwen3MultiAdapterModel(backend, adapters)
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    context_encoder = FrozenQwenContextEncoder(backend)
    residual_parameters = _residual_parameters(model)
    residual_ids = {id(parameter) for parameter in residual_parameters}
    adapter_parameters = [
        parameter
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
        if id(parameter) not in residual_ids
    ]
    head_parameters = list(heads.parameters())
    trainable = adapter_parameters + residual_parameters + head_parameters
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    optimizer_contains_qwen = any(id(parameter) in qwen_ids for parameter in trainable)
    if optimizer_contains_qwen:
        raise RuntimeError("optimizer contains frozen Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": head_parameters, "lr": learning_rate, "weight_decay": weight_decay},
        ]
    )
    loss_weights = weights or BinaryPathDiagnosticLossWeights()
    total_steps = stage_a_steps + stage_b_steps
    sequence = _deterministic_binary_pair_sequence(train_pairs, total_steps * gradient_accumulation, seed)
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float | str]] = []
    max_mps_allocated = 0
    cursor = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage_a_checkpoint_path = output_path / f"binary_path_{mode}_stage_a_seed_{seed}.pt"
    stage_b_checkpoint_path = output_path / f"binary_path_{mode}_stage_b_seed_{seed}.pt"

    for step in range(total_steps):
        stage = "stage_a_answer_first" if step < stage_a_steps else "stage_b_centroid_regularization"
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        for _ in range(gradient_accumulation):
            pair = sequence[cursor]
            cursor += 1
            encoded, baseline, output, pooled, delta, updates = _run_pair_forward(
                backend,
                model,
                context_encoder,
                pair,
                max_length,
            )
            _encoded_ablation, _baseline_ablation, ablated_output, _ablated_pooled, ablated_delta, _ablated_updates = _run_pair_forward(
                backend,
                model,
                context_encoder,
                pair,
                max_length,
                ablation_config=_path_ablation_for_pair(pair),
            )
            option_vectors = option_cache.get(pair.answer_options)
            if option_vectors is None:
                option_vectors = torch.tensor(
                    build_answer_option_vectors(backend, pair.answer_options),
                    dtype=torch.float32,
                    device=backend.device,
                )
                option_cache[pair.answer_options] = option_vectors
            targets = torch.tensor(
                [pair.expected_full_option_id, pair.expected_counterfactual_option_id],
                dtype=torch.long,
                device=backend.device,
            )
            scores = _answer_scores(delta, option_vectors)
            update_scores = _answer_scores(updates[-1], option_vectors)
            ablated_scores = _answer_scores(ablated_delta, option_vectors)
            margin_losses = []
            margins = []
            for index, target in enumerate(targets.tolist()):
                final_loss, final_margin = _margin_loss(scores[index : index + 1], target, answer_margin)
                update_loss, update_margin = _margin_loss(update_scores[index : index + 1], target, answer_margin)
                margin_losses.append(0.5 * (final_loss + update_loss))
                margins.append(0.5 * (final_margin + update_margin))
            binary_answer_margin_loss = torch.stack(margin_losses).mean()
            correct_option_margin = torch.stack(margins).mean()
            path_ablation_drop_loss = _path_drop_loss(scores, ablated_scores, targets, answer_margin)
            pair_similarity = (
                F.normalize(delta[0:1], dim=-1)
                * F.normalize(delta[1:2], dim=-1)
            ).sum(dim=-1)
            path_counterfactual_flip_loss = torch.relu(
                pair_similarity - (1.0 - wrong_separation_margin)
            ).mean()
            wrong_context_separation_loss = path_counterfactual_flip_loss
            hidden_scores = _answer_scores(pooled.float(), option_vectors)
            hidden_losses = []
            for index, target in enumerate(targets.tolist()):
                loss, _margin = _margin_loss(hidden_scores[index : index + 1], target, answer_margin)
                hidden_losses.append(loss)
            full_hidden_centroid_regularization_loss = torch.stack(hidden_losses).mean()
            preservation = _kl_preservation(output.logits[:, -1, :], baseline.logits[:, -1, :])
            residual_budget = torch.stack(
                [
                    torch.tensor(trace.delta_norm, device=backend.device, dtype=torch.float32)
                    for trace in output.traces.values()
                ]
            ).sum()
            residual_budget_loss = torch.relu(residual_budget - 0.75).square()
            hidden_norm_ratio = max(trace.hidden_norm_ratio for trace in output.traces.values())
            total = (
                loss_weights.binary_answer_margin * binary_answer_margin_loss
                + loss_weights.path_counterfactual_flip * path_counterfactual_flip_loss
                + loss_weights.path_ablation_drop * path_ablation_drop_loss
                + loss_weights.wrong_context_separation * wrong_context_separation_loss
                + loss_weights.logit_preservation * preservation
                + loss_weights.residual_budget * residual_budget_loss
            )
            if stage == "stage_b_centroid_regularization":
                total = total + (
                    loss_weights.full_hidden_centroid_regularization
                    * full_hidden_centroid_regularization_loss
                )
            (total / gradient_accumulation).backward()
            values = {
                "total_loss": total,
                "binary_answer_margin_loss": binary_answer_margin_loss,
                "path_counterfactual_flip_loss": path_counterfactual_flip_loss,
                "path_ablation_drop_loss": path_ablation_drop_loss,
                "wrong_context_separation_loss": wrong_context_separation_loss,
                "full_hidden_centroid_regularization_loss": full_hidden_centroid_regularization_loss,
                "correct_option_margin": correct_option_margin,
                "logit_preservation_kl": preservation,
                "residual_budget_loss": residual_budget_loss,
            }
            for name, value in values.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu()) / gradient_accumulation
            aggregate["hidden_norm_ratio"] = max(aggregate.get("hidden_norm_ratio", 0.0), hidden_norm_ratio)

        adapter_gradient_norm = torch.nn.utils.clip_grad_norm_(adapter_parameters + residual_parameters, 1.0)
        head_gradient_norm = torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
        optimizer.step()
        mps_allocated = torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0
        row: dict[str, float | str] = {
            "step": float(step),
            "stage": stage,
            **aggregate,
            "adapter_gradient_norm": float(adapter_gradient_norm.detach().cpu()),
            "head_gradient_norm": float(head_gradient_norm.detach().cpu()),
            "rss": float(psutil.Process().memory_info().rss),
            "mps_allocated": float(mps_allocated),
        }
        for layer in model.target_layers:
            adapter = model.adapters[str(layer)]
            row[f"residual_scale_{layer}"] = float(adapter.residual_scale.detach().cpu())
            for scale_name in ("memory_residual_scale", "rule_residual_scale", "state_residual_scale", "base_residual_scale"):
                scale = getattr(adapter, scale_name, None)
                if isinstance(scale, torch.nn.Parameter):
                    row[f"{scale_name}_{layer}"] = float(scale.detach().cpu())
        losses.append(row)
        max_mps_allocated = max(max_mps_allocated, int(mps_allocated))
        if step + 1 == stage_a_steps:
            torch.save(
                _checkpoint_payload(
                    model,
                    heads,
                    {
                        "mode": mode,
                        "seed": seed,
                        "stage": "stage_a_answer_first",
                        "base_model_sha256": backend.initial_sha256,
                        "loss_weights": asdict(loss_weights),
                        "adapter_variant": adapter_variant,
                    },
                ),
                stage_a_checkpoint_path,
            )
        if step == 0 or (step + 1) % 10 == 0 or step + 1 in {stage_a_steps, total_steps}:
            print(
                f"binary_path_training mode={mode} stage={stage} step={step + 1}/{total_steps} "
                f"total_loss={float(row['total_loss']):.6f} answer_margin={float(row['correct_option_margin']):.6f}",
                flush=True,
            )
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    torch.save(
        _checkpoint_payload(
            model,
            heads,
            {
                "mode": mode,
                "seed": seed,
                "stage": "stage_b_centroid_regularization",
                "base_model_sha256": backend.initial_sha256,
                "loss_weights": asdict(loss_weights),
                "adapter_variant": adapter_variant,
            },
        ),
        stage_b_checkpoint_path,
    )
    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    window = max(1, min(10, len(losses) // 3))
    tracked = (
        "total_loss",
        "binary_answer_margin_loss",
        "path_ablation_drop_loss",
        "full_hidden_centroid_regularization_loss",
    )
    result = BinaryPathDiagnosticTrainingResult(
        mode=mode,
        seed=seed,
        target_layers=model.target_layers,
        stage_a_steps=stage_a_steps,
        stage_b_steps=stage_b_steps,
        losses=losses,
        loss_decreased={
            name: _window_mean(losses[-window:], name) < _window_mean(losses[:window], name)
            for name in tracked
        },
        adapter_parameter_count=sum(adapter.trainable_parameter_count for adapter in model.adapters.values()),
        qwen_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters()),
        qwen_trainable_parameter_count=sum(
            parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad
        ),
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=optimizer_contains_qwen,
        initial_fingerprint=initial_fingerprint,
        final_fingerprint=final_fingerprint,
        fingerprint_unchanged=initial_fingerprint == final_fingerprint,
        stage_a_checkpoint_path=str(stage_a_checkpoint_path),
        stage_b_checkpoint_path=str(stage_b_checkpoint_path),
        max_mps_allocated=max_mps_allocated,
        adapter_variant=adapter_variant,
    )
    return model, heads, result
