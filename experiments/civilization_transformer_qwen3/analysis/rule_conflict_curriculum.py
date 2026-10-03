from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH, _mean
from .answer_option_readout import build_answer_option_vectors
from .binary_path_diagnostic_data import (
    RULE_CONDITIONED_CONFLICT_MODE,
    BinaryDiagnosticPair,
    assert_binary_pairs_valid,
    assert_rule_conditioned_conflict_pairs_valid,
    build_binary_path_diagnostic_pairs,
    build_rule_conditioned_conflict_pairs,
    split_binary_pairs,
)
from .binary_path_diagnostic_training import _deterministic_binary_pair_sequence, _residual_parameters
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_training import _answer_scores, _margin_loss
from .projected_binary_path_necessity import (
    EVALUATION_MODES,
    _evaluate_projected_matrix,
    _make_model,
    _pair_forward,
    _save_projected_checkpoint,
)


CURRICULUM_MODES = (
    "memory_only_diagnostic",
    "rule_only_diagnostic",
    RULE_CONDITIONED_CONFLICT_MODE,
    "combined_binary_curriculum",
)


@dataclass
class CurriculumTrainingResult:
    seed: int
    target_layers: tuple[int, ...]
    losses: list[dict[str, float | str]]
    loss_decreased: dict[str, bool]
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    checkpoint_paths: dict[str, str]


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


def _window_decreased(rows: list[dict[str, Any]], key: str, stage: str | None = None) -> bool:
    values = [
        float(row[key])
        for row in rows
        if key in row and (stage is None or row.get("curriculum_stage") == stage)
    ]
    if len(values) < 2:
        return False
    window = max(1, min(10, len(values) // 3))
    return sum(values[-window:]) / window < sum(values[:window]) / window


def _balanced_combined_pairs(*groups: list[BinaryDiagnosticPair], pairs_per_mode: int) -> list[BinaryDiagnosticPair]:
    per_group = max(1, pairs_per_mode // len(groups))
    result: list[BinaryDiagnosticPair] = []
    for group in groups:
        result.extend(sorted(group, key=lambda pair: pair.pair_id)[:per_group])
    remaining = pairs_per_mode - len(result)
    if remaining > 0:
        tail = [pair for group in groups for pair in sorted(group, key=lambda item: item.pair_id)[per_group:]]
        result.extend(tail[:remaining])
    return sorted(result, key=lambda pair: (pair.mode, pair.pair_id))


def _stage_sequence(
    primary: list[BinaryDiagnosticPair],
    steps: int,
    seed: int,
    rehearsal: tuple[list[BinaryDiagnosticPair], ...] = (),
) -> list[BinaryDiagnosticPair]:
    if steps <= 0:
        return []
    primary_sequence = _deterministic_binary_pair_sequence(primary, steps, seed)
    if not rehearsal:
        return primary_sequence
    rehearsal_pool = [pair for group in rehearsal for pair in group]
    rehearsal_sequence = _deterministic_binary_pair_sequence(rehearsal_pool, steps, seed + 17)
    result: list[BinaryDiagnosticPair] = []
    for index in range(steps):
        result.append(rehearsal_sequence[index] if index % 4 == 3 else primary_sequence[index])
    return result


def _train_curriculum(
    backend: Qwen3Backend,
    train_by_mode: dict[str, list[BinaryDiagnosticPair]],
    output_dir: Path,
    seed: int,
    target_layers: tuple[int, ...],
    memory_steps: int,
    rule_steps: int,
    conflict_steps: int,
    combined_steps: int,
    centroid_steps: int,
    max_length: int,
    learning_rate: float,
) -> tuple[Any, PathReadoutProjector, CurriculumTrainingResult]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_model(backend, target_layers)
    projector = PathReadoutProjector().to(backend.device)
    context_encoder = FrozenQwenContextEncoder(backend)
    option_vectors = torch.tensor(
        build_answer_option_vectors(backend, train_by_mode["memory_only_diagnostic"][0].answer_options),
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
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": 0.01},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": projector.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
    )
    stages: list[tuple[str, list[BinaryDiagnosticPair], int]] = [
        ("memory_alignment", _stage_sequence(train_by_mode["memory_only_diagnostic"], memory_steps, seed), memory_steps),
        (
            "rule_alignment",
            _stage_sequence(train_by_mode["rule_only_diagnostic"], rule_steps, seed + 1, (train_by_mode["memory_only_diagnostic"],)),
            rule_steps,
        ),
        (
            "rule_conflict_alignment",
            _stage_sequence(
                train_by_mode[RULE_CONDITIONED_CONFLICT_MODE],
                conflict_steps,
                seed + 2,
                (train_by_mode["memory_only_diagnostic"], train_by_mode["rule_only_diagnostic"]),
            ),
            conflict_steps,
        ),
        (
            "combined_curriculum",
            _stage_sequence(train_by_mode["combined_binary_curriculum"], combined_steps, seed + 3),
            combined_steps,
        ),
        (
            "centroid_regularization",
            _stage_sequence(train_by_mode["combined_binary_curriculum"], centroid_steps, seed + 4),
            centroid_steps,
        ),
    ]
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses: list[dict[str, float | str]] = []
    checkpoint_paths: dict[str, str] = {}
    for stage_name, sequence, _steps in stages:
        for pair in sequence:
            optimizer.zero_grad(set_to_none=True)
            full_hidden, delta = _pair_forward(backend, model, context_encoder, pair, max_length, "full")
            projected = projector(delta)
            targets = torch.tensor(
                [pair.expected_full_option_id, pair.expected_counterfactual_option_id],
                dtype=torch.long,
                device=backend.device,
            )
            scores = _answer_scores(projected, option_vectors)
            answer_losses = []
            margins = []
            for index, target in enumerate(targets.tolist()):
                loss, margin = _margin_loss(scores[index : index + 1], target, 0.20)
                answer_losses.append(loss)
                margins.append(margin)
            answer_loss = torch.stack(answer_losses).mean()
            pair_loss = torch.relu(
                (
                    F.normalize(projected[0:1], dim=-1)
                    * F.normalize(projected[1:2], dim=-1)
                ).sum(dim=-1)
                - 0.80
            ).mean()
            centroid_loss = torch.zeros((), device=backend.device)
            if stage_name == "centroid_regularization":
                full_scores = _answer_scores(full_hidden, option_vectors)
                centroid_losses = []
                for index, target in enumerate(targets.tolist()):
                    loss, _margin = _margin_loss(full_scores[index : index + 1], target, 0.10)
                    centroid_losses.append(loss)
                centroid_loss = torch.stack(centroid_losses).mean()
            total = answer_loss + pair_loss + 0.05 * centroid_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), 1.0)
            optimizer.step()
            losses.append(
                {
                    "curriculum_stage": stage_name,
                    "pair_mode": pair.mode,
                    "total_loss": float(total.detach().cpu()),
                    "projected_answer_loss": float(answer_loss.detach().cpu()),
                    "pair_flip_loss": float(pair_loss.detach().cpu()),
                    "fixed_centroid_loss": float(centroid_loss.detach().cpu()),
                    "projected_margin": float(torch.stack(margins).mean().detach().cpu()),
                }
            )
        checkpoint_path = checkpoint_dir / f"rule_conflict_curriculum_{stage_name}_seed_{seed}.pt"
        _save_projected_checkpoint(
            checkpoint_path,
            model,
            projector,
            {"stage": stage_name, "seed": seed, "target_layers": target_layers},
        )
        checkpoint_paths[stage_name] = str(checkpoint_path)
        if backend.device.type == "mps":
            torch.mps.empty_cache()
    result = CurriculumTrainingResult(
        seed=seed,
        target_layers=target_layers,
        losses=losses,
        loss_decreased={
            "projected_answer_loss": _window_decreased(losses, "projected_answer_loss"),
            "pair_flip_loss": _window_decreased(losses, "pair_flip_loss"),
            "fixed_centroid_loss": _window_decreased(losses, "fixed_centroid_loss", "centroid_regularization"),
        },
        qwen_trainable_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        qwen_gradients_present=sum(parameter.grad is not None for parameter in backend.model.parameters()),
        optimizer_contains_qwen_parameters=any(
            id(parameter) in qwen_ids
            for parameter in list(model.parameters()) + list(projector.parameters())
        ),
        checkpoint_paths=checkpoint_paths,
    )
    return model, projector, result


def run_qwen3_rule_conflict_curriculum(
    output_dir: str | Path = "experiments/civilization_transformer_qwen3/artifacts/rule_conflict_curriculum",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 202,
    pairs_per_mode: int = 120,
    train_pairs: int = 80,
    held_out_pairs: int = 40,
    memory_steps: int = 40,
    rule_steps: int = 40,
    conflict_steps: int = 60,
    combined_steps: int = 80,
    centroid_steps: int = 20,
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
    mode_started = time.perf_counter()
    model, projector, training = _train_curriculum(
        backend=backend,
        train_by_mode=train_by_mode,
        output_dir=output_path,
        seed=seed,
        target_layers=target_layers,
        memory_steps=memory_steps,
        rule_steps=rule_steps,
        conflict_steps=conflict_steps,
        combined_steps=combined_steps,
        centroid_steps=centroid_steps,
        max_length=max_length,
        learning_rate=learning_rate,
    )
    training_row = asdict(training)
    loss_rows = training_row.pop("losses")
    projected_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    stage_names = tuple(training.checkpoint_paths)
    from .projected_binary_path_necessity import _load_projected_checkpoint

    for stage_name in stage_names:
        _load_projected_checkpoint(training.checkpoint_paths[stage_name], model, projector)
        for mode in CURRICULUM_MODES:
            stage_projected, stage_raw, stage_fixed, stage_pairs, stage_traces, stage_failures = _evaluate_projected_matrix(
                backend=backend,
                model=model,
                projector=projector,
                train_pairs=train_by_mode[mode],
                test_pairs=test_by_mode[mode],
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
            stage_rows.append(
                {
                    "curriculum_stage": stage_name,
                    "diagnostic_mode": mode,
                    "projected_accuracy": _mean(stage_projected, lambda row: row["eval_mode"] == "full"),
                    "fixed_accuracy": _mean(stage_fixed, lambda row: row["eval_mode"] == "full"),
                    "projected_pair_success": _mean(stage_pairs, lambda row: row["stage"] == stage_name, field="projected_pair_success"),
                }
            )
    final_stage = "centroid_regularization"
    for mode in CURRICULUM_MODES:
        full = _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == final_stage and row["eval_mode"] == "full")
        wrong = _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == final_stage and row["eval_mode"] == "wrong_context")
        wrong_rows.append(
            {
                "diagnostic_mode": mode,
                "full_accuracy": full,
                "wrong_context_accuracy": wrong,
                "wrong_context_drop": full - wrong,
            }
        )
        for eval_mode in ("no_memory_path", "no_state_path", "no_rule_path", "empty_context", "counterfactual_context"):
            ablated = _mean(projected_rows, lambda row, current=mode, current_eval=eval_mode: row["diagnostic_mode"] == current and row["stage"] == final_stage and row["eval_mode"] == current_eval)
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
    def stage_metric(stage: str, mode: str, field: str) -> float:
        return _mean(stage_rows, lambda row: row["curriculum_stage"] == stage and row["diagnostic_mode"] == mode, field=field)

    drops = {(row["diagnostic_mode"], row["eval_mode"]): row["absolute_drop"] for row in ablation_rows}
    final_projected = {
        mode: stage_metric(final_stage, mode, "projected_accuracy")
        for mode in CURRICULUM_MODES
    }
    final_pair = {
        mode: stage_metric(final_stage, mode, "projected_pair_success")
        for mode in CURRICULUM_MODES
    }
    memory_best = max(stage_metric(stage, "memory_only_diagnostic", "projected_accuracy") for stage in stage_names)
    rule_best = max(stage_metric(stage, "rule_only_diagnostic", "projected_accuracy") for stage in stage_names)
    conflict_best = max(stage_metric(stage, RULE_CONDITIONED_CONFLICT_MODE, "projected_accuracy") for stage in stage_names)
    fixed_before = sum(stage_metric("combined_curriculum", mode, "fixed_accuracy") for mode in CURRICULUM_MODES) / len(CURRICULUM_MODES)
    fixed_after = sum(stage_metric(final_stage, mode, "fixed_accuracy") for mode in CURRICULUM_MODES) / len(CURRICULUM_MODES)
    max_hidden_norm = max((float(row["hidden_norm_ratio"]) for row in trace_rows if row["hidden_norm_ratio"] is not None), default=1.0)
    stage_gates = {
        "qwen_frozen": training.qwen_trainable_parameter_count == 0 and training.qwen_gradients_present == 0 and not training.optimizer_contains_qwen_parameters,
        "no_engineering_failures": not failure_cases,
        "disabled_zero_equivalence": all(
            abs(
                _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == final_stage and row["eval_mode"] == "adapter_disabled")
                - _mean(projected_rows, lambda row, current=mode: row["diagnostic_mode"] == current and row["stage"] == final_stage and row["eval_mode"] == "zero_scale")
            )
            <= 1e-9
            for mode in CURRICULUM_MODES
        ),
        "hidden_norm_ratio": max_hidden_norm <= 2.0,
        "memory_projected_accuracy": final_projected["memory_only_diagnostic"] >= 0.90,
        "memory_pair_flip": final_pair["memory_only_diagnostic"] >= 0.85,
        "memory_path_drop": drops.get(("memory_only_diagnostic", "no_memory_path"), 0.0) >= 0.25,
        "memory_rule_control": drops.get(("memory_only_diagnostic", "no_rule_path"), 0.0) <= 0.10,
        "rule_projected_accuracy": final_projected["rule_only_diagnostic"] >= 0.90,
        "rule_pair_flip": final_pair["rule_only_diagnostic"] >= 0.85,
        "rule_path_drop": drops.get(("rule_only_diagnostic", "no_rule_path"), 0.0) >= 0.25,
        "rule_memory_control": drops.get(("rule_only_diagnostic", "no_memory_path"), 0.0) <= 0.10,
        "conflict_projected_accuracy": final_projected[RULE_CONDITIONED_CONFLICT_MODE] >= 0.85,
        "conflict_pair_flip": final_pair[RULE_CONDITIONED_CONFLICT_MODE] >= 0.80,
        "conflict_rule_path_drop": drops.get((RULE_CONDITIONED_CONFLICT_MODE, "no_rule_path"), 0.0) >= 0.25,
        "conflict_memory_control": drops.get((RULE_CONDITIONED_CONFLICT_MODE, "no_memory_path"), 0.0) <= 0.15,
        "conflict_wrong_context_drop": next((row["wrong_context_drop"] for row in wrong_rows if row["diagnostic_mode"] == RULE_CONDITIONED_CONFLICT_MODE), 0.0) >= 0.20,
        "combined_projected_accuracy": final_projected["combined_binary_curriculum"] >= 0.80,
        "combined_pair_flip": final_pair["combined_binary_curriculum"] >= 0.75,
        "memory_rehearsal_retention": memory_best - final_projected["memory_only_diagnostic"] <= 0.05,
        "rule_rehearsal_retention": rule_best - final_projected["rule_only_diagnostic"] <= 0.05,
        "conflict_rehearsal_retention": conflict_best - final_projected[RULE_CONDITIONED_CONFLICT_MODE] <= 0.08,
        "fixed_centroid_improvement": fixed_after - fixed_before >= 0.05,
    }
    resource_rows = [
        {
            "seconds": time.perf_counter() - mode_started,
            "rss": psutil.Process().memory_info().rss,
            "mps_allocated": torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0,
        }
    ]
    rehearsal_rows = [
        {
            "diagnostic_mode": "memory_only_diagnostic",
            "best_accuracy": memory_best,
            "final_accuracy": final_projected["memory_only_diagnostic"],
            "regression": memory_best - final_projected["memory_only_diagnostic"],
        },
        {
            "diagnostic_mode": "rule_only_diagnostic",
            "best_accuracy": rule_best,
            "final_accuracy": final_projected["rule_only_diagnostic"],
            "regression": rule_best - final_projected["rule_only_diagnostic"],
        },
        {
            "diagnostic_mode": RULE_CONDITIONED_CONFLICT_MODE,
            "best_accuracy": conflict_best,
            "final_accuracy": final_projected[RULE_CONDITIONED_CONFLICT_MODE],
            "regression": conflict_best - final_projected[RULE_CONDITIONED_CONFLICT_MODE],
        },
    ]
    summary = {
        "model_path": str(Path(model_path).resolve()),
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "seed": seed,
        "adapter_variant": "path_specific_v2",
        "readout": "projected_delta",
        "final_projected_accuracy": final_projected,
        "final_pair_success": final_pair,
        "fixed_centroid_average_before": fixed_before,
        "fixed_centroid_average_after": fixed_after,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
        "allows_multiclass_repair_planning": all(stage_gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "weights_unchanged": backend.verify_weights_unchanged(),
    }
    _json_dump(output_path / "summary.json", summary)
    _json_dump(output_path / "training_runs.json", [training_row])
    _write_csv(output_path / "loss_curves.csv", loss_rows)
    _write_csv(output_path / "curriculum_stage_metrics.csv", stage_rows)
    _write_csv(output_path / "projected_readout_metrics.csv", projected_rows)
    _write_csv(output_path / "raw_readout_metrics.csv", raw_rows)
    _write_csv(output_path / "fixed_centroid_metrics.csv", fixed_rows)
    _write_csv(output_path / "path_ablation_drop.csv", ablation_rows)
    _write_csv(output_path / "wrong_context_metrics.csv", wrong_rows)
    _write_csv(output_path / "pair_flip_metrics.csv", pair_rows)
    _write_csv(output_path / "rehearsal_retention.csv", rehearsal_rows)
    _write_csv(output_path / "trace_contribution.csv", trace_rows)
    _json_dump(output_path / "resource_usage.json", resource_rows)
    _json_dump(output_path / "failure_cases.json", failure_cases)
    _json_dump(
        output_path / "dataset_manifest.json",
        {
            "dataset_mode": "rule_conditioned_conflict_curriculum_v1",
            "pairs_per_mode": pairs_per_mode,
            "train_pairs": train_pairs,
            "held_out_pairs": held_out_pairs,
            "modes": list(CURRICULUM_MODES),
        },
    )
    return summary
