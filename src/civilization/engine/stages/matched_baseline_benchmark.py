from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import psutil
import torch
from torch import nn
import torch.nn.functional as F

from ..adapter.context_encoder import context_items_for_sample, qwen_text_for_sample
from ..backend import Qwen3Backend
from .answer_option_readout import build_answer_option_vectors
from .evidence_answer_training import _answer_scores
from .group_full_hidden_centroid_integration import _centroid_scores
from .hidden_states import last_non_padding_pool
from .multiclass_group_curriculum_repair import GROUP_TYPES, SurfaceGroupCandidateBatch
from .rule_conflict_group_recovery import _build_group_splits


METHOD_MLP = "mlp_adapter"
METHOD_CONTEXT = "context_token_only"
METHOD_PROMPT = "prompt_task_text_only"
SUPPORTED_METHODS = (METHOD_MLP, METHOD_CONTEXT, METHOD_PROMPT)
EVAL_MODES = ("full", "no_memory", "no_rule", "no_state", "empty_context", "wrong_context")
TARGET_LAYERS = (16, 24)
MLP_BOTTLENECK = 912
CIVILIZATION_OPTIMIZED_PARAMETER_COUNT = 3_740_164
DEFAULT_OPTIMIZATION_STEPS = 440


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_mode(mode: str) -> str:
    if mode not in EVAL_MODES:
        raise ValueError(f"unsupported token-context mode: {mode}")
    return "empty" if mode == "empty_context" else "wrong" if mode == "wrong_context" else "full"


def _serialize_sample_context(sample, mode: str, *, include_context: bool) -> str:
    task = qwen_text_for_sample(sample).strip()
    if not include_context:
        return f"[TASK]\n{task}"
    context = context_items_for_sample(sample, context_mode=_normalize_mode(mode))
    memories = [f"{item.summary} {item.content} {item.relation_type}" for item in context.memories]
    rules = [f"{item.type} {item.condition} {item.effect} {item.source}" for item in context.rules]
    if mode == "no_memory":
        memories = []
    if mode == "no_rule":
        rules = []
    state = context.state_values if mode != "no_state" else (0.5, 0.5, 0.5)
    memory_text = "\n".join(memories) if memories else "<none>"
    rule_text = "\n".join(rules) if rules else "<none>"
    return (
        f"[TASK]\n{task}\n"
        f"[MEMORY]\n{memory_text}\n"
        f"[RULE]\n{rule_text}\n"
        f"[STATE]\nrigor={state[0]:.4f}; creativity={state[1]:.4f}; defensiveness={state[2]:.4f}"
    )


def _group_texts(group: SurfaceGroupCandidateBatch, mode: str, *, include_context: bool) -> list[str]:
    return [
        _serialize_sample_context(pair.full_sample, mode, include_context=include_context)
        for pair in group.pairs
    ]


def _targets(group: SurfaceGroupCandidateBatch, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [pair.expected_full_option_id for pair in group.pairs],
        dtype=torch.long,
        device=device,
    )


class PlainBottleneckAdapter(nn.Module):
    """A conventional residual MLP adapter without Memory/Rule/State branches."""

    def __init__(self, hidden_size: int = 1024, bottleneck_size: int = MLP_BOTTLENECK) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down_projection = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.up_projection = nn.Linear(bottleneck_size, hidden_size, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = self.up_projection(F.gelu(self.down_projection(self.norm(hidden))))
        return hidden + self.residual_scale * update


class Qwen3PlainAdapterModel(nn.Module):
    def __init__(self, backend: Qwen3Backend, target_layers: tuple[int, ...] = TARGET_LAYERS) -> None:
        super().__init__()
        self.backend = backend
        self.adapters = nn.ModuleDict(
            {
                str(layer): PlainBottleneckAdapter().to(backend.device, dtype=backend.dtype)
                for layer in target_layers
            }
        )
        self.target_layers = target_layers
        self.active_hook_count = 0

    def forward(self, encoded: dict[str, torch.Tensor]):
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        handles = []

        def make_hook(layer: int):
            def hook(_module, _args, output):
                return self.adapters[str(layer)](output)
            return hook

        try:
            for layer in self.target_layers:
                handles.append(self.backend.model.model.layers[layer].register_forward_hook(make_hook(layer)))
                self.active_hook_count += 1
            output = self.backend.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
                self.active_hook_count -= 1
        if self.active_hook_count:
            raise RuntimeError("plain adapter hook leaked")
        return output, attention_mask


def plain_adapter_parameter_count(bottleneck_size: int = MLP_BOTTLENECK) -> int:
    # Per layer: LayerNorm weight+bias, two bias-free projections, residual scalar.
    return len(TARGET_LAYERS) * (2 * 1024 + 2 * 1024 * bottleneck_size + 1)


def _encode_texts(backend: Qwen3Backend, texts: list[str], max_length: int) -> tuple[dict[str, torch.Tensor], list[int]]:
    tokenized = backend.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    token_counts = [len(row) for row in tokenized["input_ids"]]
    if max(token_counts, default=0) > max_length:
        raise ValueError(f"matched baseline input truncation: max_tokens={max(token_counts)} max_length={max_length}")
    encoded, truncations = backend.encode(texts, max_length=max_length)
    if truncations:
        raise ValueError("matched baseline input was truncated")
    return {key: value.to(backend.device) for key, value in encoded.items()}, token_counts


def _pooled_forward(
    backend: Qwen3Backend,
    model: Qwen3PlainAdapterModel | None,
    texts: list[str],
    max_length: int,
) -> tuple[torch.Tensor, list[int]]:
    encoded, counts = _encode_texts(backend, texts, max_length)
    if model is None:
        output = backend.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        mask = encoded["attention_mask"]
    else:
        output, mask = model(encoded)
    return last_non_padding_pool(output.hidden_states[-1], mask).float(), counts


def _option_vectors(backend: Qwen3Backend, group: SurfaceGroupCandidateBatch) -> torch.Tensor:
    return torch.tensor(
        build_answer_option_vectors(backend, group.pairs[0].base_record.answer_options),
        dtype=torch.float32,
        device=backend.device,
    )


def _balanced_step_groups(groups: list[SurfaceGroupCandidateBatch], step: int, seed: int) -> list[SurfaceGroupCandidateBatch]:
    selected = []
    for type_index, group_type in enumerate(GROUP_TYPES):
        rows = sorted((row for row in groups if row.group_type == group_type), key=lambda row: row.surface_group_id)
        if not rows:
            raise ValueError(f"missing training group type: {group_type}")
        selected.append(rows[(step + seed + type_index) % len(rows)])
    return selected


def _train_mlp(
    *,
    backend: Qwen3Backend,
    model: Qwen3PlainAdapterModel,
    train_groups: list[SurfaceGroupCandidateBatch],
    steps: int,
    seed: int,
    max_length: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    losses: list[dict[str, Any]] = []
    for step in range(steps):
        groups = _balanced_step_groups(train_groups, step, seed)
        full_texts = [text for group in groups for text in _group_texts(group, "full", include_context=True)]
        wrong_texts = [text for group in groups for text in _group_texts(group, "wrong_context", include_context=True)]
        targets = torch.cat([_targets(group, backend.device) for group in groups])
        optimizer.zero_grad(set_to_none=True)
        pooled, counts = _pooled_forward(backend, model, full_texts + wrong_texts, max_length)
        full = pooled[: targets.numel()]
        wrong = pooled[targets.numel() :]
        option_vectors = torch.cat([_option_vectors(backend, group) for group in groups], dim=0)
        # All local groups use the same five candidate answers.
        option_vectors = option_vectors[:5]
        full_scores = _answer_scores(full, option_vectors)
        wrong_scores = _answer_scores(wrong, option_vectors)
        dynamic_centroids = torch.stack([full[targets == option].mean(dim=0) for option in range(5)])
        dynamic_scores = _centroid_scores(full, dynamic_centroids)
        correct = full_scores.gather(1, targets[:, None]).squeeze(1)
        wrong_correct = wrong_scores.gather(1, targets[:, None]).squeeze(1)
        loss = (
            F.cross_entropy(full_scores / 0.05, targets)
            + 2.0 * F.cross_entropy(dynamic_scores / 0.05, targets)
            + torch.relu(0.20 - (correct - wrong_correct)).mean()
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        optimizer.step()
        accuracy = float((full_scores.argmax(dim=-1) == targets).float().mean().detach().cpu())
        row = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "train_accuracy": accuracy,
            "gradient_norm": grad_norm,
            "max_input_tokens": max(counts),
        }
        losses.append(row)
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(
                f"matched_baseline method={METHOD_MLP} seed={seed} step={step + 1}/{steps} "
                f"loss={row['loss']:.6f} accuracy={accuracy:.3f}",
                flush=True,
            )
    return losses


def _build_centroids(
    *,
    backend: Qwen3Backend,
    model: Qwen3PlainAdapterModel | None,
    groups: list[SurfaceGroupCandidateBatch],
    include_context: bool,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    chunks: dict[str, list[torch.Tensor]] = {kind: [] for kind in GROUP_TYPES}
    labels: dict[str, list[torch.Tensor]] = {kind: [] for kind in GROUP_TYPES}
    audits = []
    with torch.no_grad():
        for group in groups:
            pooled, counts = _pooled_forward(
                backend, model, _group_texts(group, "full", include_context=include_context), max_length
            )
            chunks[group.group_type].append(pooled.detach())
            labels[group.group_type].append(_targets(group, backend.device))
            audits.append({
                "group_type": group.group_type,
                "surface_group_id": group.surface_group_id,
                "max_input_tokens": max(counts),
                "included_in_centroid": True,
            })
    centroids = {}
    for kind in GROUP_TYPES:
        matrix = torch.cat(chunks[kind])
        targets = torch.cat(labels[kind])
        centroids[kind] = torch.stack([matrix[targets == option].mean(dim=0) for option in range(5)]).detach()
    return centroids, audits


def _evaluate(
    *,
    backend: Qwen3Backend,
    model: Qwen3PlainAdapterModel | None,
    groups: list[SurfaceGroupCandidateBatch],
    centroids: dict[str, torch.Tensor],
    include_context: bool,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    prompt_rows = []
    with torch.no_grad():
        for group in groups:
            targets = _targets(group, backend.device)
            for mode in EVAL_MODES:
                texts = _group_texts(group, mode, include_context=include_context)
                pooled, counts = _pooled_forward(backend, model, texts, max_length)
                predictions = _centroid_scores(pooled, centroids[group.group_type]).argmax(dim=-1)
                rows.append({
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "mode": mode,
                    "accuracy": float((predictions == targets).float().mean().cpu()),
                    "group_success": bool(torch.all(predictions == targets).cpu()),
                    "max_input_tokens": max(counts),
                    "mean_input_tokens": sum(counts) / len(counts),
                })
                for text, count in zip(texts, counts, strict=True):
                    prompt_rows.append({
                        "group_type": group.group_type,
                        "surface_group_id": group.surface_group_id,
                        "mode": mode,
                        "prompt_sha256": _sha(text),
                        "input_tokens": count,
                    })
    return rows, prompt_rows


def _mean(rows: Iterable[dict[str, Any]], *, kind: str | None = None, mode: str = "full") -> float:
    values = [
        float(row["accuracy"])
        for row in rows
        if row["mode"] == mode and (kind is None or row["group_type"] == kind)
    ]
    return sum(values) / len(values) if values else 0.0


def _resource_start(backend: Qwen3Backend) -> tuple[float, int]:
    if backend.device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.device)
    return time.perf_counter(), psutil.Process().memory_info().rss


def _resource_end(backend: Qwen3Backend, started: float, rss_before: int) -> dict[str, Any]:
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
    return {
        "runtime_seconds": time.perf_counter() - started,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": psutil.Process().memory_info().rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(backend.device) if backend.device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(backend.device) if backend.device.type == "cuda" else 0,
    }


def run_matched_baseline_seed(
    *,
    backend: Qwen3Backend,
    output_dir: str | Path,
    method: str,
    seed: int,
    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    max_length: int = 160,
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported matched baseline method: {method}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    train, heldout, manifest = _build_group_splits(
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        seed=seed,
        max_length=64,
    )
    mapping = [
        {
            "split": split,
            "group_type": group.group_type,
            "surface_group_id": group.surface_group_id,
            "targets": [pair.expected_full_option_id for pair in group.pairs],
            "task_text_sha256": _sha(qwen_text_for_sample(group.pairs[0].full_sample)),
        }
        for split, groups in (("train", train), ("heldout", heldout))
        for group in groups
    ]
    _json_dump(output / "dataset_manifest.json", manifest)
    _write_csv(output / "sample_contract.csv", mapping)
    initial_fingerprint = backend.parameter_fingerprint()
    started, rss_before = _resource_start(backend)
    model = Qwen3PlainAdapterModel(backend) if method == METHOD_MLP else None
    include_context = method != METHOD_PROMPT
    loss_rows: list[dict[str, Any]] = []
    if model is not None:
        loss_rows = _train_mlp(
            backend=backend,
            model=model,
            train_groups=train,
            steps=optimization_steps,
            seed=seed,
            max_length=max_length,
        )
    centroids, centroid_rows = _build_centroids(
        backend=backend,
        model=model,
        groups=train,
        include_context=include_context,
        max_length=max_length,
    )
    metric_rows, prompt_rows = _evaluate(
        backend=backend,
        model=model,
        groups=heldout,
        centroids=centroids,
        include_context=include_context,
        max_length=max_length,
    )
    resource = _resource_end(backend, started, rss_before)
    trainable = sum(parameter.numel() for parameter in model.parameters()) if model is not None else 0
    qwen_gradients = sum(
        1 for parameter in backend.model.parameters()
        if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu()) > 0
    )
    full = _mean(metric_rows)
    wrong = _mean(metric_rows, mode="wrong_context")
    summary = {
        "method": method,
        "seed": seed,
        "model": "Qwen3-0.6B",
        "model_sha256": backend.initial_sha256,
        "optimization_steps": optimization_steps if model is not None else 0,
        "reference_optimization_steps": optimization_steps,
        "trainable_parameter_count": trainable,
        "civilization_optimized_parameter_count": CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
        "parameter_match_relative_error": (
            abs(trainable - CIVILIZATION_OPTIMIZED_PARAMETER_COUNT) / CIVILIZATION_OPTIMIZED_PARAMETER_COUNT
            if model is not None else None
        ),
        "accuracy": full,
        "wrong_context_accuracy": wrong,
        "wrong_context_drop": full - wrong,
        "memory_ablation_drop": full - _mean(metric_rows, mode="no_memory"),
        "rule_ablation_drop": full - _mean(metric_rows, mode="no_rule"),
        "state_ablation_drop": full - _mean(metric_rows, mode="no_state"),
        "empty_context_drop": full - _mean(metric_rows, mode="empty_context"),
        "per_group_accuracy": {kind: _mean(metric_rows, kind=kind) for kind in GROUP_TYPES},
        "ablation_semantics": "information_channel" if include_context else "not_applicable_task_only",
        "qwen_trainable_parameters": sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
        "qwen_gradients": qwen_gradients,
        "qwen_weights_unchanged": initial_fingerprint == backend.parameter_fingerprint() and backend.verify_weights_unchanged(),
        **resource,
    }
    _write_csv(output / "loss_curves.csv", loss_rows)
    _write_csv(output / "centroid_build_audit.csv", centroid_rows)
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "prompt_audit.csv", prompt_rows)
    _json_dump(output / "resource_usage.json", resource)
    _json_dump(output / "summary.json", summary)
    if model is not None:
        torch.save(
            {
                "adapter_state_dict": model.adapters.state_dict(),
                "training_metadata": {
                    "method": method,
                    "seed": seed,
                    "target_layers": TARGET_LAYERS,
                    "bottleneck_size": MLP_BOTTLENECK,
                    "optimization_steps": optimization_steps,
                },
            },
            output / f"{method}_seed_{seed}.pt",
        )
    return summary
