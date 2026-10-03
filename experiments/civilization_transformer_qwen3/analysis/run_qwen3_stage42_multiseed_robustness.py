from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from .group_full_hidden_centroid_integration import run_qwen3_group_full_hidden_centroid_integration
from .memory_delta_gradient_diagnostic import run_qwen3_memory_delta_gradient_diagnostic
from .rule_conflict_group_recovery import run_qwen3_rule_conflict_group_recovery


DEFAULT_OUTPUT_DIR = Path(
    "experiments/civilization_transformer_qwen3/artifacts/stage42_multiseed_robustness"
)
STAGE40_CHECKPOINT_NAME = (
    "multiclass_necessity_local_only_transfer_group_memory_delta_path_only_seed_{seed}.pt"
)
STAGE41_CHECKPOINT_NAME = "stage41_combined_seed_{seed}.pt"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _seed_stage40_dir(output: Path, seed: int) -> Path:
    return output / "stage40" / f"seed_{seed}"


def _seed_stage41_dir(output: Path, seed: int) -> Path:
    return output / "stage41" / f"seed_{seed}"


def _seed_stage42_dir(output: Path, seed: int) -> Path:
    return output / "stage42" / f"seed_{seed}"


def _stage40_checkpoint(output: Path, seed: int) -> Path:
    return _seed_stage40_dir(output, seed) / "checkpoints" / STAGE40_CHECKPOINT_NAME.format(seed=seed)


def _stage41_checkpoint(output: Path, seed: int) -> Path:
    return _seed_stage41_dir(output, seed) / "checkpoints" / "stage_combined" / STAGE41_CHECKPOINT_NAME.format(seed=seed)


def _existing_stage40_checkpoint(seed: int) -> Path:
    return Path(
        "experiments/civilization_transformer_qwen3/artifacts/memory_delta_gradient_diagnostic/checkpoints"
    ) / STAGE40_CHECKPOINT_NAME.format(seed=seed)


def _existing_stage41_checkpoint(seed: int) -> Path:
    return Path(
        "experiments/civilization_transformer_qwen3/artifacts/rule_conflict_group_recovery/checkpoints/stage_combined"
    ) / STAGE41_CHECKPOINT_NAME.format(seed=seed)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _metric_row(seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    final_fixed = summary.get("final_fixed_centroid_accuracy", {})
    wrong_drop = summary.get("wrong_context_fixed_centroid_drop", {})
    return {
        "seed": seed,
        "passes_stage_gate": bool(summary.get("passes_stage_gate")),
        "allows_stage43": bool(summary.get("allows_stage43")),
        "fixed_centroid_before": float(summary.get("fixed_centroid_average_before", 0.0)),
        "fixed_centroid_after": float(summary.get("fixed_centroid_average_after", 0.0)),
        "memory_fixed_centroid": float(final_fixed.get("memory_necessity_group", 0.0)),
        "rule_fixed_centroid": float(final_fixed.get("rule_necessity_group", 0.0)),
        "conflict_fixed_centroid": float(final_fixed.get("memory_rule_conflict_group", 0.0)),
        "combined_fixed_centroid": float(final_fixed.get("combined", 0.0)),
        "wrong_context_drop": float(wrong_drop.get("combined", 0.0)),
        "hidden_norm_ratio": float(summary.get("hidden_norm_ratio", 0.0)),
        "qwen_trainable_parameters": float(summary.get("qwen_trainable_parameters", -1.0)),
        "qwen_gradients": float(summary.get("qwen_gradients", -1.0)),
        "weights_unchanged": bool(summary.get("weights_unchanged")),
        "failed_stage": summary.get("failed_stage"),
        "failed_gate": summary.get("failed_gate"),
        "runtime_seconds": float(summary.get("runtime_seconds", 0.0)),
        "stage41_checkpoint": summary.get("stage41_checkpoint"),
        "stage42_alignment_checkpoint": summary.get("alignment_checkpoint"),
    }


def _mean_std_rows(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_metrics = (
        "fixed_centroid_before",
        "fixed_centroid_after",
        "memory_fixed_centroid",
        "rule_fixed_centroid",
        "conflict_fixed_centroid",
        "combined_fixed_centroid",
        "wrong_context_drop",
        "hidden_norm_ratio",
    )
    rows: list[dict[str, Any]] = []
    passed_rows = [row for row in per_seed_rows if row["passes_stage_gate"]]
    for metric in numeric_metrics:
        mean, std = _mean_std([float(row[metric]) for row in passed_rows])
        rows.append({"metric": metric, "mean": mean, "std": std, "n": len(passed_rows)})
    return rows


def _ensure_stage40_checkpoint(
    *,
    output: Path,
    seed: int,
    preferred_device: str | None,
    local_samples_per_label: int,
    local_train_groups: int,
    max_length: int,
    stage40_steps: int,
    reuse_existing: bool,
) -> tuple[Path, dict[str, Any] | None]:
    existing = _existing_stage40_checkpoint(seed)
    target = _stage40_checkpoint(output, seed)
    if reuse_existing and existing.exists():
        return existing, None
    if target.exists():
        return target, None
    print(f"stage42_multiseed_progress seed={seed} stage=stage40 start", flush=True)
    summary = run_qwen3_memory_delta_gradient_diagnostic(
        output_dir=_seed_stage40_dir(output, seed),
        seed=seed,
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        repair_steps=stage40_steps,
        max_length=max_length,
        preferred_device=preferred_device,
    )
    _cleanup_cuda()
    if not summary.get("passes_stage_gate"):
        raise RuntimeError(f"Stage40 failed for seed {seed}: {summary.get('failed_gate')}")
    return Path(summary["checkpoint_path"]), summary


def _ensure_stage41_checkpoint(
    *,
    output: Path,
    seed: int,
    stage40_checkpoint: Path,
    preferred_device: str | None,
    local_samples_per_label: int,
    local_train_groups: int,
    max_length: int,
    rule_steps: int,
    conflict_steps: int,
    combined_steps: int,
    projector_sanity_steps: int,
    reuse_existing: bool,
) -> tuple[Path, dict[str, Any] | None]:
    existing = _existing_stage41_checkpoint(seed)
    target = _stage41_checkpoint(output, seed)
    if reuse_existing and existing.exists():
        return existing, None
    if target.exists():
        return target, None
    print(f"stage42_multiseed_progress seed={seed} stage=stage41 start", flush=True)
    summary = run_qwen3_rule_conflict_group_recovery(
        output_dir=_seed_stage41_dir(output, seed),
        stage40_checkpoint=stage40_checkpoint,
        seed=seed,
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        rule_steps=rule_steps,
        conflict_steps=conflict_steps,
        combined_steps=combined_steps,
        projector_sanity_steps=projector_sanity_steps,
        max_length=max_length,
        preferred_device=preferred_device,
    )
    _cleanup_cuda()
    if not summary.get("passes_stage_gate"):
        raise RuntimeError(f"Stage41 failed for seed {seed}: {summary.get('failed_gate')}")
    return _stage41_checkpoint(output, seed), summary


def run_qwen3_stage42_multiseed_robustness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] = (202, 303, 404),
    preferred_device: str | None = None,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    max_length: int = 64,
    stage40_steps: int = 80,
    rule_steps: int = 80,
    conflict_steps: int = 80,
    combined_steps: int = 100,
    projector_sanity_steps: int = 20,
    stage42_alignment_steps: int = 100,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"stage42_multiseed_progress {index}/{len(seeds)} seed={seed} stage=begin",
            flush=True,
        )
        try:
            stage40_ckpt, stage40_summary = _ensure_stage40_checkpoint(
                output=output,
                seed=seed,
                preferred_device=preferred_device,
                local_samples_per_label=local_samples_per_label,
                local_train_groups=local_train_groups,
                max_length=max_length,
                stage40_steps=stage40_steps,
                reuse_existing=reuse_existing,
            )
            stage41_ckpt, stage41_summary = _ensure_stage41_checkpoint(
                output=output,
                seed=seed,
                stage40_checkpoint=stage40_ckpt,
                preferred_device=preferred_device,
                local_samples_per_label=local_samples_per_label,
                local_train_groups=local_train_groups,
                max_length=max_length,
                rule_steps=rule_steps,
                conflict_steps=conflict_steps,
                combined_steps=combined_steps,
                projector_sanity_steps=projector_sanity_steps,
                reuse_existing=reuse_existing,
            )
            print(f"stage42_multiseed_progress seed={seed} stage=stage42 start", flush=True)
            stage42_summary = run_qwen3_group_full_hidden_centroid_integration(
                output_dir=_seed_stage42_dir(output, seed),
                stage41_checkpoint=stage41_ckpt,
                seed=seed,
                local_samples_per_label=local_samples_per_label,
                local_train_groups=local_train_groups,
                alignment_steps=stage42_alignment_steps,
                max_length=max_length,
                preferred_device=preferred_device,
            )
            _cleanup_cuda()
            row = _metric_row(seed, stage42_summary)
            per_seed_rows.append(row)
            run_records.append(
                {
                    "seed": seed,
                    "stage40_checkpoint": str(stage40_ckpt),
                    "stage41_checkpoint": str(stage41_ckpt),
                    "stage40_summary_generated": stage40_summary is not None,
                    "stage41_summary_generated": stage41_summary is not None,
                    "stage42_summary_path": str(_seed_stage42_dir(output, seed) / "summary.json"),
                    "passes_stage_gate": row["passes_stage_gate"],
                }
            )
            if not row["passes_stage_gate"]:
                failures.append({"seed": seed, "failed_stage": row["failed_stage"], "failed_gate": row["failed_gate"]})
        except Exception as exc:  # noqa: BLE001 - artifact must preserve failed seed reason.
            failures.append({"seed": seed, "failed_stage": "exception", "failed_gate": type(exc).__name__, "message": str(exc)})
            run_records.append({"seed": seed, "passes_stage_gate": False, "error": str(exc)})
            print(f"stage42_multiseed_progress seed={seed} stage=failed error={exc}", flush=True)
            break
    mean_std = _mean_std_rows(per_seed_rows)
    passed_seed_count = sum(1 for row in per_seed_rows if row["passes_stage_gate"])
    summary = {
        "benchmark": "stage42_multiseed_robustness",
        "seeds": list(seeds),
        "completed_seed_count": len(per_seed_rows),
        "passed_seed_count": passed_seed_count,
        "passes_stage_gate": passed_seed_count == len(seeds) and not failures,
        "allows_paper_multiseed_claim": passed_seed_count == len(seeds) and not failures,
        "per_seed_metrics": per_seed_rows,
        "mean_std_metrics": mean_std,
        "failures": failures,
    }
    _write_csv(output / "per_seed_metrics.csv", per_seed_rows)
    _write_csv(output / "mean_std_metrics.csv", mean_std)
    _json_dump(output / "training_runs.json", run_records)
    _json_dump(output / "failure_cases.json", failures)
    _json_dump(output / "summary.json", summary)
    return summary


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 42 multi-seed robustness benchmark.")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "seeds": _parse_seeds(args.seeds),
        "preferred_device": args.preferred_device,
        "reuse_existing": not args.no_reuse_existing,
    }
    if args.smoke:
        kwargs.update(
            local_samples_per_label=4,
            local_train_groups=2,
            stage40_steps=4,
            rule_steps=4,
            conflict_steps=4,
            combined_steps=4,
            projector_sanity_steps=4,
            stage42_alignment_steps=4,
        )
    summary = run_qwen3_stage42_multiseed_robustness(**kwargs)
    print(
        "qwen3_stage42_multiseed_robustness_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "completed_seed_count": summary["completed_seed_count"],
                "passed_seed_count": summary["passed_seed_count"],
                "failures": summary["failures"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
