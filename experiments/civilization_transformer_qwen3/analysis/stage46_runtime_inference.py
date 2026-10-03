from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from experiments.civilization_transformer_torch.model import CivilizationAblationConfig

from ..adapter.civilization_adapter import CivilizationAdapterContext
from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .answer_option_readout import build_answer_option_vectors
from .group_full_hidden_centroid_integration import _load_stage41_checkpoint
from .hidden_states import last_non_padding_pool
from .context_readout_alignment import PathReadoutProjector
from .full_hidden_centroid_alignment import FullHiddenCentroidProjector
from .multiclass_necessity_repair import _make_path_specific_model
from .stage45_adapter_package import (
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE45_OUTPUT_DIR,
    Stage45InferenceRequest,
    Stage45InferenceResponse,
    Stage45PackageManifest,
    load_stage45_package_manifest,
    validate_inference_request,
    validate_stage45_package,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage46_runtime_inference")
DEFAULT_PACKAGE_MANIFEST = DEFAULT_STAGE45_OUTPUT_DIR / "package_manifest.json"


class RawCentroidProvider(Protocol):
    def get(self, seed: int, task_name: str, option_count: int) -> torch.Tensor | None:
        ...


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_checkpoint(manifest: Stage45PackageManifest, seed: int, task_name: str) -> Path:
    preferred_roles = [task_name, "combined"] if task_name in {"glue_rte", "super_glue_cb", "boolq"} else ["full_hidden_alignment"]
    for role in preferred_roles:
        for ref in manifest.checkpoints:
            if ref.seed == seed and ref.role == role:
                return Path(ref.path)
    raise FileNotFoundError(f"no packaged checkpoint for seed={seed} task={task_name}")


def _cosine_scores(vector: torch.Tensor, option_vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.float(), dim=-1) @ F.normalize(option_vectors.float(), dim=-1).T


def _last_token_projected_delta(output, projector) -> torch.Tensor:
    tensors = []
    for trace in output.traces.values():
        path_tensors = [
            trace.memory_delta_tensor,
            trace.rule_delta_tensor,
            trace.state_delta_tensor,
            trace.base_delta_tensor,
        ]
        for tensor in path_tensors:
            if tensor is None:
                continue
            if tensor.shape[0] == output.attention_mask.shape[0] and tensor.shape[1] == 1:
                tensor = tensor.expand(-1, output.attention_mask.shape[1], -1)
            if tensor.shape[:2] != output.attention_mask.shape:
                raise RuntimeError(
                    f"delta tensor shape mismatch: {tuple(tensor.shape[:2])} != {tuple(output.attention_mask.shape)}"
                )
            tensors.append(tensor)
    if not tensors:
        raise RuntimeError("adapter traces did not expose differentiable delta tensors")
    combined = torch.stack(tensors, dim=0).sum(dim=0)
    attention_mask = output.attention_mask
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if torch.any(indices < 0):
        raise RuntimeError("attention mask contains an empty sequence")
    delta = combined[torch.arange(combined.shape[0], device=combined.device), indices]
    return projector(delta)


def _runtime_context(
    backend: Qwen3Backend,
    request: Stage45InferenceRequest,
    attention_mask: torch.Tensor,
    *,
    adapter_enabled: bool = True,
    force_zero_scale: bool = False,
    ablation_config: CivilizationAblationConfig | None = None,
    context_scale: float = 1.0,
) -> CivilizationAdapterContext:
    encoder = FrozenQwenContextEncoder(backend)
    memory_vectors, memory_mask = encoder.encode_text_groups([list(request.memory_items)])
    rule_vectors, rule_mask = encoder.encode_text_groups([list(request.rule_items)])
    state_values = torch.tensor([request.state_values], dtype=backend.dtype, device=backend.device)
    return CivilizationAdapterContext(
        memory_vectors=memory_vectors * context_scale,
        memory_mask=memory_mask,
        state_values=state_values,
        rule_vectors=rule_vectors * context_scale,
        rule_mask=rule_mask,
        attention_mask=attention_mask.to(backend.device),
        ablation_config=ablation_config,
        adapter_enabled=adapter_enabled,
        force_zero_scale=force_zero_scale,
    )


def _trace_summary(output) -> dict[str, Any]:
    rows = {}
    for layer, trace in output.traces.items():
        rows[str(layer)] = {
            "hidden_norm_ratio": trace.hidden_norm_ratio,
            "delta_norm": trace.delta_norm,
            "memory_delta_norm": trace.memory_delta_norm,
            "rule_delta_norm": trace.rule_delta_norm,
            "state_delta_norm": trace.state_delta_norm,
            "base_delta_norm": trace.base_delta_norm,
            "memory_residual_scale": trace.memory_residual_scale,
            "rule_residual_scale": trace.rule_residual_scale,
            "path_dominance_ratio": trace.path_dominance_ratio,
            "adapter_enabled": trace.adapter_enabled,
            "path_specific_adapter_version": trace.path_specific_adapter_version,
        }
    return rows


class Stage46QwenRuntimeBackend:
    def __init__(
        self,
        manifest: Stage45PackageManifest,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        preferred_device: str = "cuda",
        max_length: int = 128,
        raw_centroid_provider: RawCentroidProvider | None = None,
        backend: Qwen3Backend | None = None,
    ) -> None:
        validation = validate_stage45_package(manifest)
        if not validation["passes_stage_gate"]:
            raise ValueError(f"invalid Stage45 package: {validation}")
        self.manifest = manifest
        self.backend = backend if backend is not None else Qwen3Backend(model_path, preferred_device=preferred_device)
        self.max_length = max_length
        self.raw_centroid_provider = raw_centroid_provider
        self._loaded_key: tuple[int, str] | None = None
        self._model = None
        self._projector = None
        self._full_hidden_projector = None
        self._payload: dict[str, Any] | None = None

    def _load_for(self, seed: int, task_name: str):
        key = (seed, task_name if task_name in {"glue_rte", "super_glue_cb", "boolq"} else "local")
        if self._loaded_key == key:
            return self._model, self._projector, self._full_hidden_projector
        checkpoint = _select_checkpoint(self.manifest, seed, task_name)
        payload = torch.load(checkpoint, map_location=self.backend.device, weights_only=True)
        if "task_projector_state_dict" in payload:
            metadata = payload.get("training_metadata", {})
            target_layers = tuple(int(layer) for layer in metadata.get("target_layers", [16, 24]))
            model = _make_path_specific_model(self.backend, target_layers)
            model.adapters.load_state_dict(payload["adapter_state_dict"])
            projector = PathReadoutProjector().to(self.backend.device)
            prefix = f"projectors.{task_name}."
            task_state = {
                name[len(prefix) :]: value
                for name, value in payload["task_projector_state_dict"].items()
                if name.startswith(prefix)
            }
            if not task_state:
                raise ValueError(f"task-specific projector state is missing for {task_name}")
            projector.load_state_dict(task_state)
            full_hidden_projector = FullHiddenCentroidProjector().to(self.backend.device)
            full_hidden_projector.load_state_dict(payload["full_hidden_projector_state_dict"])
        else:
            model, projector, full_hidden_projector, payload = _load_stage41_checkpoint(self.backend, checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in projector.parameters():
            parameter.requires_grad_(False)
        for parameter in full_hidden_projector.parameters():
            parameter.requires_grad_(False)
        self._loaded_key = key
        self._model = model
        self._projector = projector
        self._full_hidden_projector = full_hidden_projector
        self._payload = payload
        return model, projector, full_hidden_projector

    def predict(self, request: Stage45InferenceRequest, manifest: Stage45PackageManifest | None = None) -> Stage45InferenceResponse:
        validate_inference_request(request, self.manifest)
        model, projector, full_hidden_projector = self._load_for(request.seed, request.task_name)
        encoded, truncations = self.backend.encode([request.text], max_length=self.max_length)
        encoded = {name: tensor.to(self.backend.device) for name, tensor in encoded.items()}
        external_task = request.task_name in {"glue_rte", "super_glue_cb", "boolq"}
        use_memory = request.control_mode != "no_memory_path"
        use_rule = request.control_mode != "no_rule_path"
        use_state = not external_task and request.control_mode != "no_state_path"
        ablation = CivilizationAblationConfig(
            use_memory_path=use_memory,
            use_rule_path=use_rule,
            use_state_path=use_state,
        )
        context = _runtime_context(
            self.backend,
            request,
            encoded["attention_mask"],
            adapter_enabled=request.control_mode != "adapter_disabled",
            force_zero_scale=request.control_mode == "zero_scale",
            ablation_config=ablation,
            context_scale=0.85 if external_task else 1.0,
        )
        with torch.no_grad():
            output = model(encoded, context)
        if truncations:
            raise ValueError(f"Stage46 inference text was truncated: {truncations}")
        option_vectors = torch.tensor(
            build_answer_option_vectors(self.backend, request.answer_options),
            dtype=torch.float32,
            device=self.backend.device,
        )
        scores: dict[str, list[float]] = {}
        if "projected_delta" in request.readouts:
            projected = _last_token_projected_delta(output, projector)
            scores["projected_delta"] = [float(value) for value in _cosine_scores(projected, option_vectors)[0].detach().cpu()]
        raw_centroids = None
        if "raw_full_hidden_fixed_centroid" in request.readouts:
            raw_centroids = (
                self.raw_centroid_provider.get(request.seed, request.task_name, len(request.answer_options))
                if self.raw_centroid_provider is not None
                else None
            )
            if raw_centroids is None:
                scores["raw_full_hidden_fixed_centroid"] = []
            else:
                full_hidden = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
                scores["raw_full_hidden_fixed_centroid"] = [
                    float(value)
                    for value in _cosine_scores(full_hidden, raw_centroids.to(self.backend.device))[0].detach().cpu()
                ]
        if "projected_full_hidden" in request.readouts:
            full_hidden = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
            projected_full_hidden = full_hidden_projector(full_hidden)
            scores["projected_full_hidden"] = [
                float(value) for value in _cosine_scores(projected_full_hidden, option_vectors)[0].detach().cpu()
            ]
        primary_scores = scores.get("projected_delta") or scores.get("projected_full_hidden") or []
        predicted = int(max(range(len(primary_scores)), key=lambda index: primary_scores[index])) if primary_scores else None
        qwen_gradients = sum(1 for parameter in self.backend.model.parameters() if parameter.grad is not None)
        trace = {
            "checkpoint_seed": request.seed,
            "task_name": request.task_name,
            "control_mode": request.control_mode,
            "device": str(self.backend.device),
            "dtype": str(self.backend.dtype),
            "qwen_trainable_parameters": sum(parameter.numel() for parameter in self.backend.model.parameters() if parameter.requires_grad),
            "qwen_gradients": qwen_gradients,
            "weights_unchanged": self.backend.verify_weights_unchanged(),
            "traces": _trace_summary(output),
            "raw_full_hidden_fixed_centroid": {
                "available": raw_centroids is not None,
                "reason": None if raw_centroids is not None else "No compatible Stage47 train centroid bundle was provided.",
            },
        }
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=predicted,
            scores=scores,
            trace=trace,
        )


def run_stage46_runtime_smoke(
    *,
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preferred_device: str = "cuda",
    seed: int = 202,
) -> dict[str, Any]:
    manifest = load_stage45_package_manifest(package_manifest)
    runtime = Stage46QwenRuntimeBackend(
        manifest,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=128,
    )
    request = Stage45InferenceRequest(
        text="A controller must decide whether to approve the operation after reviewing the supplied evidence.",
        memory_items=("The operational evidence supports approving the action.",),
        rule_items=("Use grounded memory evidence unless a stronger rule overrides it.",),
        state_values=(0.9, 0.1, 0.8),
        answer_options=("Approve the action.", "Reject the action."),
        task_name="custom",
        seed=seed,
        readouts=("projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"),
    )
    response = runtime.predict(request)
    hidden_norm_ratios = [row["hidden_norm_ratio"] for row in response.trace["traces"].values()]
    summary = {
        "stage": "stage46_runtime_inference_smoke",
        "request": asdict(request),
        "response": asdict(response),
        "stage_gates": {
            "status_ok": response.status == "ok",
            "projected_scores_present": bool(response.scores.get("projected_delta")),
            "qwen_frozen": response.trace["qwen_trainable_parameters"] == 0,
            "qwen_gradients": response.trace["qwen_gradients"] == 0,
            "weights_unchanged": response.trace["weights_unchanged"],
            "hidden_norm_ratio": max(hidden_norm_ratios) <= 2.0 if hidden_norm_ratios else False,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir)
    _json_dump(output / "runtime_smoke_summary.json", summary)
    return summary
