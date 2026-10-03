from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package")
DEFAULT_STAGE44C_SUMMARY = Path("experiments/civilization_transformer_qwen3/artifacts/stage44c_real_task_runner/summary.json")
DEFAULT_STAGE44A_ROOT = Path("experiments/civilization_transformer_qwen3/artifacts/stage44a_local_multiclass_integration")
DEFAULT_STAGE44B6_ROOT = Path("experiments/civilization_transformer_qwen3/artifacts/stage44b6_external_recovery_training")
PACKAGE_VERSION = "stage45_v1"
QWEN_MODEL_FAMILY = "Qwen3-0.6B"
ADAPTER_ARCHITECTURE = "dual_16_24_path_specific_v2"
RAW_FULL_HIDDEN_RESIDUAL_SCALE = 200.0
EXTERNAL_TASKS = ("glue_rte", "super_glue_cb", "boolq")
INFERENCE_CONTROL_MODES = (
    "full",
    "no_memory_path",
    "no_rule_path",
    "no_state_path",
    "adapter_disabled",
    "zero_scale",
)


@dataclass(frozen=True)
class Stage45CheckpointRef:
    seed: int
    scope: str
    role: str
    path: str
    sha256: str
    bytes: int
    excludes_qwen_weights: bool


@dataclass(frozen=True)
class Stage45PackageManifest:
    package_version: str
    model_family: str
    architecture: str
    target_layers: tuple[int, int]
    raw_full_hidden_residual_scale: float
    seeds: tuple[int, ...]
    source_stage: str
    stage44c_summary_path: str
    stage44c_summary_sha256: str
    local_stage_passed: bool
    external_stage_passed: bool
    dataset_fields_cache_ok: bool
    checkpoints: tuple[Stage45CheckpointRef, ...]
    external_dataset_manifest: tuple[dict[str, Any], ...]
    inference_contract: dict[str, Any]


@dataclass(frozen=True)
class Stage45InferenceRequest:
    text: str
    memory_items: tuple[str, ...]
    rule_items: tuple[str, ...]
    state_values: tuple[float, float, float]
    answer_options: tuple[str, ...]
    task_name: str = "custom"
    seed: int = 202
    readouts: tuple[str, ...] = ("projected_delta", "raw_full_hidden_fixed_centroid")
    control_mode: str = "full"


@dataclass(frozen=True)
class Stage45InferenceResponse:
    status: str
    task_name: str
    seed: int
    predicted_option_id: int | None
    scores: dict[str, list[float]]
    trace: dict[str, Any]
    reason: str | None = None


class Stage45RuntimeBackend(Protocol):
    def predict(self, request: Stage45InferenceRequest, manifest: Stage45PackageManifest) -> Stage45InferenceResponse:
        ...


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_contains_qwen_weights(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    forbidden_top_level = {"qwen_state_dict", "model_state_dict", "base_model_state_dict"}
    if forbidden_top_level & set(payload):
        return True
    for key in payload:
        lowered = str(key).lower()
        if lowered.startswith("qwen") or "qwen_state" in lowered:
            return True
    for value in payload.values():
        if isinstance(value, dict):
            for nested_key in value:
                lowered = str(nested_key).lower()
                if lowered.startswith("model.") or lowered.startswith("qwen") or "embed_tokens" in lowered:
                    return True
    return False


def _checkpoint_ref(path: Path, seed: int, scope: str, role: str) -> Stage45CheckpointRef:
    if not path.exists():
        raise FileNotFoundError(f"missing Stage45 checkpoint dependency: {path}")
    contains_qwen = checkpoint_contains_qwen_weights(path)
    return Stage45CheckpointRef(
        seed=seed,
        scope=scope,
        role=role,
        path=str(path),
        sha256=_sha256(path),
        bytes=path.stat().st_size,
        excludes_qwen_weights=not contains_qwen,
    )


def _local_checkpoint(stage44a_root: Path, seed: int) -> Path:
    return stage44a_root / f"seed_{seed}" / "checkpoints" / "full_hidden_alignment" / f"stage44a_full_hidden_alignment_seed_{seed}.pt"


def _external_checkpoint(stage44b6_root: Path, seed: int, role: str) -> Path:
    filename = f"stage44b4_{role}_seed_{seed}.pt"
    return stage44b6_root / f"seed_{seed}" / "checkpoints" / role / filename


def _combined_external_checkpoint(stage44b6_root: Path, seed: int) -> Path:
    return stage44b6_root / f"seed_{seed}" / "checkpoints" / "combined" / f"stage44b4_combined_seed_{seed}.pt"


def _dataset_fields_cache_ok(rows: Iterable[dict[str, Any]]) -> bool:
    return all(
        row.get("dataset_fields_structured_ok") is True and row.get("structure_sources") == ["dataset_fields"]
        for row in rows
    )


def _inference_contract() -> dict[str, Any]:
    return {
        "request_fields": [
            "text",
            "memory_items",
            "rule_items",
            "state_values",
            "answer_options",
            "task_name",
            "seed",
            "readouts",
            "control_mode",
        ],
        "required_readouts": ["projected_delta", "raw_full_hidden_fixed_centroid"],
        "diagnostic_readouts": ["projected_full_hidden"],
        "supported_controls": [
            "full_context",
            "adapter_disabled",
            "zero_scale",
            "no_memory_path",
            "no_rule_path",
            "no_state_path",
            "wrong_context",
            "counterfactual_context",
        ],
        "frozen_base_model": True,
        "forbidden_package_contents": ["Qwen3 weights", "tokenizer files", "dataset cache payloads"],
    }


def build_stage45_adapter_package(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stage44c_summary_path: str | Path = DEFAULT_STAGE44C_SUMMARY,
    stage44a_root: str | Path = DEFAULT_STAGE44A_ROOT,
    stage44b6_root: str | Path = DEFAULT_STAGE44B6_ROOT,
    seeds: tuple[int, ...] = (202, 303, 404),
    include_task_checkpoints: bool = True,
) -> Stage45PackageManifest:
    output = Path(output_dir)
    stage44c_path = Path(stage44c_summary_path)
    stage44c = _json_load(stage44c_path)
    gates = stage44c.get("stage_gates", {})
    if not stage44c.get("passes_stage_gate"):
        raise ValueError("Stage44C summary does not pass stage gate")
    external_manifest = tuple(stage44c.get("external", {}).get("dataset_fields_cache_manifest", []))
    dataset_fields_ok = _dataset_fields_cache_ok(external_manifest)
    if not dataset_fields_ok:
        raise ValueError("Stage44C external dataset-fields cache gate is not satisfied")

    stage44a = Path(stage44a_root)
    stage44b = Path(stage44b6_root)
    refs: list[Stage45CheckpointRef] = []
    for seed in seeds:
        refs.append(_checkpoint_ref(_local_checkpoint(stage44a, seed), seed, "local", "full_hidden_alignment"))
        refs.append(_checkpoint_ref(_combined_external_checkpoint(stage44b, seed), seed, "external", "combined"))
        if include_task_checkpoints:
            for task_name in EXTERNAL_TASKS:
                refs.append(_checkpoint_ref(_external_checkpoint(stage44b, seed, task_name), seed, "external", task_name))
    if any(not ref.excludes_qwen_weights for ref in refs):
        raise ValueError("Stage45 package dependency contains Qwen weights")

    manifest = Stage45PackageManifest(
        package_version=PACKAGE_VERSION,
        model_family=QWEN_MODEL_FAMILY,
        architecture=ADAPTER_ARCHITECTURE,
        target_layers=(16, 24),
        raw_full_hidden_residual_scale=RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        seeds=seeds,
        source_stage="stage44c_real_task_runner",
        stage44c_summary_path=str(stage44c_path),
        stage44c_summary_sha256=_sha256(stage44c_path),
        local_stage_passed=bool(gates.get("local_stage44a_passed")),
        external_stage_passed=bool(gates.get("external_stage44b6_passed")),
        dataset_fields_cache_ok=dataset_fields_ok,
        checkpoints=tuple(refs),
        external_dataset_manifest=external_manifest,
        inference_contract=_inference_contract(),
    )
    _json_dump(output / "package_manifest.json", _manifest_to_json(manifest))
    _json_dump(output / "inference_contract.json", manifest.inference_contract)
    _json_dump(output / "dataset_manifest.json", list(external_manifest))
    return manifest


def _manifest_to_json(manifest: Stage45PackageManifest) -> dict[str, Any]:
    value = asdict(manifest)
    value["checkpoints"] = [asdict(ref) for ref in manifest.checkpoints]
    value["external_dataset_manifest"] = list(manifest.external_dataset_manifest)
    return value


def load_stage45_package_manifest(path: str | Path) -> Stage45PackageManifest:
    payload = _json_load(Path(path))
    checkpoints = tuple(Stage45CheckpointRef(**row) for row in payload["checkpoints"])
    return Stage45PackageManifest(
        package_version=payload["package_version"],
        model_family=payload["model_family"],
        architecture=payload["architecture"],
        target_layers=tuple(payload["target_layers"]),
        raw_full_hidden_residual_scale=float(payload["raw_full_hidden_residual_scale"]),
        seeds=tuple(payload["seeds"]),
        source_stage=payload["source_stage"],
        stage44c_summary_path=payload["stage44c_summary_path"],
        stage44c_summary_sha256=payload["stage44c_summary_sha256"],
        local_stage_passed=bool(payload["local_stage_passed"]),
        external_stage_passed=bool(payload["external_stage_passed"]),
        dataset_fields_cache_ok=bool(payload["dataset_fields_cache_ok"]),
        checkpoints=checkpoints,
        external_dataset_manifest=tuple(payload["external_dataset_manifest"]),
        inference_contract=payload["inference_contract"],
    )


def validate_stage45_package(manifest: Stage45PackageManifest) -> dict[str, Any]:
    checks = {
        "package_version": manifest.package_version == PACKAGE_VERSION,
        "architecture": manifest.architecture == ADAPTER_ARCHITECTURE,
        "target_layers": tuple(manifest.target_layers) == (16, 24),
        "residual_scale": manifest.raw_full_hidden_residual_scale == RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        "local_stage_passed": manifest.local_stage_passed,
        "external_stage_passed": manifest.external_stage_passed,
        "dataset_fields_cache_ok": manifest.dataset_fields_cache_ok,
        "checkpoints_exist": all(Path(ref.path).exists() for ref in manifest.checkpoints),
        "checkpoints_exclude_qwen": all(ref.excludes_qwen_weights for ref in manifest.checkpoints),
        "external_tasks_present": {row.get("task_name") for row in manifest.external_dataset_manifest} >= set(EXTERNAL_TASKS),
    }
    return {
        "checks": checks,
        "passes_stage_gate": all(checks.values()),
        "checkpoint_count": len(manifest.checkpoints),
        "seeds": list(manifest.seeds),
    }


def validate_inference_request(request: Stage45InferenceRequest, manifest: Stage45PackageManifest) -> None:
    if not request.text.strip():
        raise ValueError("inference request text is empty")
    if len(request.state_values) != 3:
        raise ValueError("state_values must contain exactly three floats")
    if len(request.answer_options) < 2:
        raise ValueError("at least two answer options are required")
    if request.seed not in manifest.seeds:
        raise ValueError(f"seed {request.seed} is not packaged")
    supported = set(manifest.inference_contract["required_readouts"]) | set(
        manifest.inference_contract.get("diagnostic_readouts", [])
    )
    unsupported = set(request.readouts) - supported
    if unsupported:
        raise ValueError(f"unsupported readout(s): {sorted(unsupported)}")
    if request.control_mode not in INFERENCE_CONTROL_MODES:
        raise ValueError(f"unsupported control_mode: {request.control_mode}")


class Stage45InferenceEngine:
    def __init__(self, manifest: Stage45PackageManifest, runtime_backend: Stage45RuntimeBackend | None = None):
        self.manifest = manifest
        self.runtime_backend = runtime_backend

    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        validate_inference_request(request, self.manifest)
        if self.runtime_backend is None:
            return Stage45InferenceResponse(
                status="requires_runtime_backend",
                task_name=request.task_name,
                seed=request.seed,
                predicted_option_id=None,
                scores={readout: [] for readout in request.readouts},
                trace={
                    "architecture": self.manifest.architecture,
                    "target_layers": list(self.manifest.target_layers),
                    "raw_full_hidden_residual_scale": self.manifest.raw_full_hidden_residual_scale,
                    "checkpoint_seed_available": request.seed in self.manifest.seeds,
                },
                reason="Stage45 package and request are valid; attach a Qwen3 runtime backend to execute real forward passes.",
            )
        return self.runtime_backend.predict(request, self.manifest)


def run_stage45_eval_harness(
    *,
    manifest_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_stage45_package_manifest(manifest_path)
    validation = validate_stage45_package(manifest)
    response = Stage45InferenceEngine(manifest).predict(
        Stage45InferenceRequest(
            text="A local operation record is evaluated with supplied memory and rules.",
            memory_items=("The memory evidence supports the first candidate.",),
            rule_items=("Use the supplied evidence unless an explicit rule overrides it.",),
            state_values=(1.0, 0.0, 0.5),
            answer_options=("Approve the action.", "Reject the action."),
            task_name="stage45_contract_smoke",
            seed=manifest.seeds[0],
        )
    )
    summary = {
        "stage": "stage45_adapter_package_eval_harness",
        "package_validation": validation,
        "contract_smoke_response": asdict(response),
        "passes_stage_gate": validation["passes_stage_gate"] and response.status == "requires_runtime_backend",
        "allows_runtime_integration": validation["passes_stage_gate"],
    }
    output = Path(output_dir)
    _json_dump(output / "eval_harness_summary.json", summary)
    return summary
