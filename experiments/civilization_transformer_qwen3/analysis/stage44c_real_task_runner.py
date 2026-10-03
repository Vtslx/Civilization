from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage44a_local_multiclass_integration import run_qwen3_stage44a_local_multiclass_integration
from .stage44b_external_recovery_training import run_qwen3_stage44b6_external_recovery_training


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage44c_real_task_runner")
DEFAULT_STAGE44A_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage44a_local_multiclass_integration")
DEFAULT_STAGE44B6_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage44b6_external_recovery_training")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_stage44a_summary(stage44a_dir: Path) -> dict[str, Any]:
    path = stage44a_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Stage44A summary: {path}")
    return _read_json(path)


def _existing_stage44b6_summary(stage44b6_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    per_seed = []
    for seed in seeds:
        path = stage44b6_dir / f"seed_{seed}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing Stage44B.6 seed summary: {path}")
        per_seed.append(_read_json(path))
    return {
        "stage": "stage44b6_dataset_fields_external_recovery_existing",
        "seeds": list(seeds),
        "completed_seed_count": len(per_seed),
        "passed_seed_count": sum(1 for row in per_seed if row.get("passes_stage_gate")),
        "per_seed": per_seed,
        "passes_stage_gate": len(per_seed) == len(seeds) and all(row.get("passes_stage_gate") for row in per_seed),
        "dataset_fields_cache_manifest": per_seed[-1].get("dataset_fields_cache_manifest", []) if per_seed else [],
    }


def _stage44b6_multiseed_summary(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "stage": "stage44b6_dataset_fields_external_recovery",
        "seeds": list(seeds),
        "completed_seed_count": len(rows),
        "passed_seed_count": sum(1 for row in rows if row.get("passes_stage_gate")),
        "per_seed": rows,
        "passes_stage_gate": len(rows) == len(seeds) and all(row.get("passes_stage_gate") for row in rows),
        "dataset_fields_cache_manifest": rows[-1].get("dataset_fields_cache_manifest", []) if rows else [],
    }


def _gate_summary(local: dict[str, Any], external: dict[str, Any]) -> dict[str, bool]:
    local_pass = bool(local.get("passes_stage_gate"))
    external_pass = bool(external.get("passes_stage_gate"))
    external_dataset_fields = all(
        row.get("dataset_fields_structured_ok") and row.get("structure_sources") == ["dataset_fields"]
        for row in external.get("dataset_fields_cache_manifest", [])
    )
    return {
        "local_stage44a_passed": local_pass,
        "external_stage44b6_passed": external_pass,
        "external_dataset_fields_cache": external_dataset_fields,
        "stage44c_passed": local_pass and external_pass and external_dataset_fields,
    }


def run_qwen3_stage44c_real_task_runner(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] = (202, 303, 404),
    preferred_device: str = "cuda",
    smoke: bool = False,
    reuse_existing: bool = False,
    allow_dataset_download: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        seeds = (seeds[0],)
    if reuse_existing:
        local_summary = _existing_stage44a_summary(DEFAULT_STAGE44A_DIR)
        external_summary = _existing_stage44b6_summary(DEFAULT_STAGE44B6_DIR, seeds)
    else:
        local_summary = run_qwen3_stage44a_local_multiclass_integration(
            output_dir=output / "stage44a_local",
            seeds=seeds,
            preferred_device=preferred_device,
            samples_per_label=4 if smoke else 24,
            train_groups=2 if smoke else 16,
            max_length=64 if smoke else 128,
            strict_stage_gates=not smoke,
            memory_steps=2 if smoke else 40,
            rule_steps=2 if smoke else 40,
            state_steps=2 if smoke else 40,
            conflict_steps=2 if smoke else 60,
            combined_steps=2 if smoke else 80,
            full_hidden_steps=2 if smoke else 80,
        )
        external_rows = []
        for seed in seeds:
            external_rows.append(
                run_qwen3_stage44b6_external_recovery_training(
                    output_dir=output / "stage44b6_external" / f"seed_{seed}",
                    seed=seed,
                    preferred_device=preferred_device,
                    task_steps=2 if smoke else 100,
                    combined_steps=2 if smoke else 160,
                    train_per_label=2 if smoke else 12,
                    heldout_per_label=2 if smoke else 12,
                    strict_stage_gates=not smoke,
                    allow_dataset_download=allow_dataset_download,
                )
            )
        external_summary = _stage44b6_multiseed_summary(external_rows, seeds)
    gates = _gate_summary(local_summary, external_summary)
    summary = {
        "stage": "stage44c_real_task_runner",
        "seeds": list(seeds),
        "mode": "reuse_existing" if reuse_existing else "execute",
        "smoke": smoke,
        "local": local_summary,
        "external": external_summary,
        "stage_gates": gates,
        "passes_stage_gate": gates["stage44c_passed"],
        "allows_next_stage": gates["stage44c_passed"],
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "local_summary.json", local_summary)
    _json_dump(output / "external_summary.json", external_summary)
    return summary
