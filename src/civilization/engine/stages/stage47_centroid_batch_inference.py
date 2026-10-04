from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .adapter_training import AdapterDiagnosticHeads
from .evidence_answer_data import LOCAL_ANSWER_OPTIONS
from .real_task_data import build_local_semireal_task_records, records_to_logic_datasets
from .real_task_data import load_external_task_records
from .stage44a_local_multiclass_integration import (
    LOCAL_REAL_TASK_TYPES,
    _build_centroids,
    build_stage44a_groups,
    split_stage44a_groups,
)
from ..adapter.context_encoder import context_items_for_sample, qwen_text_for_sample
from .stage45_adapter_package import (
    Stage45InferenceRequest,
    Stage45InferenceResponse,
    load_stage45_package_manifest,
)
from .stage46_runtime_inference import (
    DEFAULT_PACKAGE_MANIFEST,
    Stage46QwenRuntimeBackend,
)
from .stage44b_external_recovery_audit import deterministic_external_split, project_external_source_records
from .stage44b_external_recovery_training import (
    DEFAULT_DATASET_FIELDS_CACHE_DIR,
    EXTERNAL_ANSWER_OPTIONS,
    STAGE44B4_CONTEXT_SCALE,
    _forward_record_stage44b1,
    _token_safe_context_text,
    build_external_recovery_records,
)
from ..adapter.context_encoder import FrozenQwenContextEncoder


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage47_centroid_batch_inference")
CENTROID_BUNDLE_VERSION = "stage47_raw_full_hidden_centroids_v1"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_centroid_bundle(
    path: str | Path,
    *,
    centroids: dict[tuple[int, str], torch.Tensor],
    options: dict[str, tuple[str, ...]],
    source_audit: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for (seed, task_name), matrix in sorted(centroids.items()):
        if matrix.ndim != 2 or matrix.shape[1] != 1024:
            raise ValueError(f"invalid centroid shape for {seed}/{task_name}: {tuple(matrix.shape)}")
        task_options = options[task_name]
        if matrix.shape[0] != len(task_options):
            raise ValueError(f"centroid/option count mismatch for {seed}/{task_name}")
        entries[f"{seed}:{task_name}"] = {
            "seed": seed,
            "task_name": task_name,
            "options": list(task_options),
            "centroids": matrix.detach().cpu().float(),
        }
    torch.save(
        {
            "bundle_version": CENTROID_BUNDLE_VERSION,
            "entries": entries,
            "source_audit": source_audit,
            "metadata": metadata,
        },
        destination,
    )
    return {
        "bundle_version": CENTROID_BUNDLE_VERSION,
        "path": str(destination),
        "sha256": _sha256(destination),
        "entry_count": len(entries),
        "tasks": sorted({row["task_name"] for row in entries.values()}),
        "seeds": sorted({row["seed"] for row in entries.values()}),
        "contains_qwen_weights": False,
        "source_split": "train_full_context_only",
    }


class Stage47CentroidProvider:
    def __init__(self, bundle_path: str | Path):
        self.path = Path(bundle_path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("bundle_version") != CENTROID_BUNDLE_VERSION:
            raise ValueError("unsupported Stage47 centroid bundle version")
        if any(key in payload for key in ("qwen_state_dict", "model_state_dict", "adapter_state_dict")):
            raise ValueError("centroid bundle contains forbidden model weights")
        self.payload = payload

    def get(self, seed: int, task_name: str, option_count: int) -> torch.Tensor | None:
        entry = self.payload["entries"].get(f"{seed}:{task_name}")
        if entry is None or len(entry["options"]) != option_count:
            return None
        matrix = entry["centroids"].float()
        if matrix.shape != (option_count, 1024) or not torch.isfinite(matrix).all():
            raise ValueError(f"invalid centroid matrix for {seed}/{task_name}")
        return matrix

    def options(self, seed: int, task_name: str) -> tuple[str, ...] | None:
        entry = self.payload["entries"].get(f"{seed}:{task_name}")
        return tuple(entry["options"]) if entry is not None else None


def build_stage47_local_centroid_bundle(
    runtime: Stage46QwenRuntimeBackend,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 202,
    samples_per_label: int = 24,
    train_groups: int = 16,
    max_length: int = 128,
) -> tuple[dict[str, Any], list[Any]]:
    model, projector, full_hidden_projector = runtime._load_for(seed, "custom")
    records = build_local_semireal_task_records(
        samples_per_label, seed=seed, context_grounding_mode="stage44a_path_grounded_v2"
    )
    datasets, _tokenizer = records_to_logic_datasets(
        records, max_seq_len=max_length, context_grounding_mode="stage44a_path_grounded_v2"
    )
    train, heldout = split_stage44a_groups(build_stage44a_groups(datasets), train_groups, seed)
    heads = AdapterDiagnosticHeads().to(runtime.backend.device, dtype=torch.float32)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    centroids_by_task, audit = _build_centroids(
        backend=runtime.backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=heads,
        groups=train,
        max_length=max_length,
        projected=False,
    )
    if any(row.get("split") != "train" or row.get("mode") != "full" for row in audit):
        raise ValueError("Stage47 centroid audit includes non-train or non-full rows")
    output = Path(output_dir)
    bundle_path = output / f"centroid_bundle_seed_{seed}.pt"
    manifest = save_centroid_bundle(
        bundle_path,
        centroids={(seed, task): matrix for task, matrix in centroids_by_task.items()},
        options={task: LOCAL_ANSWER_OPTIONS for task in LOCAL_REAL_TASK_TYPES},
        source_audit=audit,
        metadata={
            "seed": seed,
            "samples_per_label": samples_per_label,
            "train_groups": train_groups,
            "max_length": max_length,
            "source": "stage44a_train_full_context_raw_hidden",
            "qwen_frozen": True,
        },
    )
    _json_dump(output / "centroid_bundle_manifest.json", manifest)
    _json_dump(output / "centroid_build_audit.json", audit)
    return manifest, heldout


def extend_stage47_external_centroids(
    runtime: Stage46QwenRuntimeBackend,
    *,
    bundle_path: str | Path,
    cache_dir: str | Path = DEFAULT_DATASET_FIELDS_CACHE_DIR,
    seed: int = 202,
    train_per_label: int = 12,
    heldout_per_label: int = 12,
    max_length: int = 384,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    destination = Path(bundle_path)
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    records_by_task, cache_manifest = load_external_task_records(
        cache_dir=cache_dir,
        task_names=tuple(EXTERNAL_ANSWER_OPTIONS),
        allow_download=False,
        context_grounding_mode="real_task_v1",
    )
    projected_sources = project_external_source_records(records_by_task)
    heldout_by_task: dict[str, list[Any]] = {}
    audit = list(payload.get("source_audit", []))
    for task_name, sources in projected_sources.items():
        train_sources, heldout_sources = deterministic_external_split(
            sources,
            seed=seed,
            train_per_label=train_per_label,
            heldout_per_label=heldout_per_label,
        )
        train_records = build_external_recovery_records(
            train_sources, grounding_mode="stage44b4_structured_semantic_v5"
        )
        heldout_by_task[task_name] = build_external_recovery_records(
            heldout_sources, grounding_mode="stage44b4_structured_semantic_v5"
        )
        model, projector, _full_hidden_projector = runtime._load_for(seed, task_name)
        context_encoder = FrozenQwenContextEncoder(runtime.backend)
        class _SingleTaskProjector:
            def __call__(self, vector: torch.Tensor, _task_name: str) -> torch.Tensor:
                return projector(vector)

        task_projector = _SingleTaskProjector()
        by_target: dict[int, list[torch.Tensor]] = {
            option_id: [] for option_id in range(len(EXTERNAL_ANSWER_OPTIONS[task_name]))
        }
        with torch.no_grad():
            for record in train_records:
                _output, _projected, pooled, _truncated = _forward_record_stage44b1(
                    backend=runtime.backend,
                    model=model,
                    task_projector=task_projector,
                    context_encoder=context_encoder,
                    record=record,
                    mode="full",
                    max_length=max_length,
                    context_scale=STAGE44B4_CONTEXT_SCALE,
                )
                target = record.source.correct_option_id
                by_target[target].append(pooled.squeeze(0).detach().cpu().float())
                audit.append(
                    {
                        "split": "train",
                        "mode": "full",
                        "centroid_scope": task_name,
                        "task_type": task_name,
                        "source_id": record.source.source_id,
                        "target": target,
                        "used": True,
                    }
                )
        matrix = torch.stack(
            [torch.stack(by_target[option_id]).mean(dim=0) for option_id in range(len(by_target))]
        )
        payload["entries"][f"{seed}:{task_name}"] = {
            "seed": seed,
            "task_name": task_name,
            "options": list(EXTERNAL_ANSWER_OPTIONS[task_name]),
            "centroids": matrix,
        }
    payload["source_audit"] = audit
    payload.setdefault("metadata", {})["external_cache_manifest"] = cache_manifest
    torch.save(payload, destination)
    manifest = {
        "bundle_version": CENTROID_BUNDLE_VERSION,
        "path": str(destination),
        "sha256": _sha256(destination),
        "entry_count": len(payload["entries"]),
        "tasks": sorted({row["task_name"] for row in payload["entries"].values()}),
        "seeds": sorted({row["seed"] for row in payload["entries"].values()}),
        "contains_qwen_weights": False,
        "source_split": "train_full_context_only",
        "external_cache_manifest": cache_manifest,
    }
    _json_dump(destination.parent / "centroid_bundle_manifest.json", manifest)
    _json_dump(destination.parent / "centroid_build_audit.json", audit)
    return manifest, heldout_by_task


class Stage47BatchInferenceEngine:
    def __init__(self, runtime: Stage46QwenRuntimeBackend, provider: Stage47CentroidProvider):
        self.runtime = runtime
        self.provider = provider
        self.runtime.raw_centroid_provider = provider

    def predict_batch(self, requests: list[Stage45InferenceRequest]) -> tuple[list[Stage45InferenceResponse], dict[str, Any]]:
        if not requests:
            raise ValueError("Stage47 batch inference requires at least one request")
        started = time.perf_counter()
        responses = [self.runtime.predict(request) for request in requests]
        elapsed = time.perf_counter() - started
        return responses, {
            "batch_size": len(requests),
            "elapsed_seconds": elapsed,
            "seconds_per_sample": elapsed / len(requests),
            "cuda_peak_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "all_ok": all(response.status == "ok" for response in responses),
            "raw_centroid_available": all(
                response.trace["raw_full_hidden_fixed_centroid"]["available"] for response in responses
            ),
        }


def _request_for_group(group, seed: int) -> Stage45InferenceRequest:
    sample = group.pairs[0].full_sample
    items = context_items_for_sample(sample)
    memory_items = tuple(
        f"{item.summary} {item.content} {item.relation_type}" for item in items.memories
    )
    rule_items = tuple(
        f"{item.type} {item.condition} {item.effect} {item.source}" for item in items.rules
    )
    return Stage45InferenceRequest(
        text=qwen_text_for_sample(sample),
        memory_items=memory_items,
        rule_items=rule_items,
        state_values=items.state_values,
        answer_options=LOCAL_ANSWER_OPTIONS,
        task_name=group.group_type,
        seed=seed,
        readouts=("projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"),
    )


def run_stage47_centroid_batch_smoke(
    *,
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path,
    preferred_device: str = "cuda",
    seed: int = 202,
    smoke: bool = False,
    include_external: bool = False,
) -> dict[str, Any]:
    manifest = load_stage45_package_manifest(package_manifest)
    runtime = Stage46QwenRuntimeBackend(
        manifest,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=384 if include_external else 64 if smoke else 128,
    )
    bundle_manifest, heldout = build_stage47_local_centroid_bundle(
        runtime,
        output_dir=output_dir,
        seed=seed,
        samples_per_label=4 if smoke else 24,
        train_groups=2 if smoke else 16,
        max_length=64 if smoke else 128,
    )
    external_heldout = {}
    if include_external:
        bundle_manifest, external_heldout = extend_stage47_external_centroids(
            runtime,
            bundle_path=bundle_manifest["path"],
            seed=seed,
            train_per_label=2 if smoke else 12,
            heldout_per_label=2 if smoke else 12,
            max_length=128 if smoke else 384,
        )
    provider = Stage47CentroidProvider(bundle_manifest["path"])
    engine = Stage47BatchInferenceEngine(runtime, provider)
    selected = []
    for task_name in LOCAL_REAL_TASK_TYPES:
        selected.append(next(group for group in heldout if group.group_type == task_name))
    requests = [_request_for_group(group, seed) for group in selected]
    if include_external:
        for task_name, records in external_heldout.items():
            record = records[0]
            sample = replace(
                record.sample,
                memory_target=_token_safe_context_text(runtime.backend, record.sample.memory_target, max_tokens=20),
                rule_target=_token_safe_context_text(runtime.backend, record.sample.rule_target, max_tokens=20),
            )
            items = context_items_for_sample(sample)
            requests.append(
                Stage45InferenceRequest(
                    text=sample.text,
                    memory_items=tuple(
                        f"{item.summary} {item.content} {item.relation_type}" for item in items.memories
                    ),
                    rule_items=tuple(
                        f"{item.type} {item.condition} {item.effect} {item.source}" for item in items.rules
                    ),
                    state_values=items.state_values,
                    answer_options=EXTERNAL_ANSWER_OPTIONS[task_name],
                    task_name=task_name,
                    seed=seed,
                    readouts=("projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"),
                )
            )
    responses, resource_usage = engine.predict_batch(requests)
    raw_available = all(response.trace["raw_full_hidden_fixed_centroid"]["available"] for response in responses)
    score_shapes_ok = all(
        len(response.scores["raw_full_hidden_fixed_centroid"]) == len(request.answer_options)
        for request, response in zip(requests, responses, strict=True)
    )
    summary = {
        "stage": "stage47_raw_centroid_batch_inference",
        "seed": seed,
        "smoke": smoke,
        "centroid_bundle": bundle_manifest,
        "responses": [asdict(response) for response in responses],
        "resource_usage": resource_usage,
        "stage_gates": {
            "bundle_entries": bundle_manifest["entry_count"] == len(LOCAL_REAL_TASK_TYPES) + (len(EXTERNAL_ANSWER_OPTIONS) if include_external else 0),
            "bundle_excludes_qwen": not bundle_manifest["contains_qwen_weights"],
            "train_full_context_only": bundle_manifest["source_split"] == "train_full_context_only",
            "batch_all_ok": resource_usage["all_ok"],
            "raw_centroid_available": raw_available,
            "raw_score_shapes": score_shapes_ok,
            "qwen_frozen": all(response.trace["qwen_trainable_parameters"] == 0 for response in responses),
            "qwen_gradients": all(response.trace["qwen_gradients"] == 0 for response in responses),
            "weights_unchanged": all(response.trace["weights_unchanged"] for response in responses),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    _json_dump(Path(output_dir) / "summary.json", summary)
    _json_dump(Path(output_dir) / "resource_usage.json", resource_usage)
    return summary
