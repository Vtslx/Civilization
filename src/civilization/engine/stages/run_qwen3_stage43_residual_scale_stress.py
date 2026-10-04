from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Iterator

import torch
from transformers import AutoTokenizer

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .group_full_hidden_centroid_integration import (
    RAW_FULL_HIDDEN_RESIDUAL_SCALE,
    run_qwen3_group_full_hidden_centroid_integration,
)
from .memory_delta_gradient_diagnostic import run_qwen3_memory_delta_gradient_diagnostic
from .multiclass_group_curriculum_repair import SurfaceGroupCandidateBatch
from .rule_conflict_group_recovery import run_qwen3_rule_conflict_group_recovery


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/civilization/stage43_residual_scale_stress"
)
STAGE40_CHECKPOINT_NAME = (
    "multiclass_necessity_local_only_transfer_group_memory_delta_path_only_seed_{seed}.pt"
)
STAGE41_CHECKPOINT_NAME = "stage41_combined_seed_{seed}.pt"
LONG_CONTEXT_MIN_LENGTH = 128


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


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_dir(output: Path, seed: int, max_length: int) -> Path:
    return output / f"seed_{seed}_len_{max_length}"


def _stage40_checkpoint(run_dir: Path, seed: int) -> Path:
    return run_dir / "stage40" / "checkpoints" / STAGE40_CHECKPOINT_NAME.format(seed=seed)


def _stage41_checkpoint(run_dir: Path, seed: int) -> Path:
    return run_dir / "stage41" / "checkpoints" / "stage_combined" / STAGE41_CHECKPOINT_NAME.format(seed=seed)


def _neutral_padding(surface_group_id: str, max_length: int) -> str:
    if max_length < LONG_CONTEXT_MIN_LENGTH:
        return ""
    tokens = [
        "archive",
        "segment",
        "operational",
        "note",
        "status",
        "ledger",
        "routine",
        "observer",
        "neutral",
        "background",
        "workspace",
        "record",
    ]
    repeated = " ".join(tokens * 3)
    return f" neutral_context_padding_v1 {surface_group_id} {repeated}"


def _pad_pair(pair, max_length: int):
    padding = _neutral_padding(pair.full_sample.surface_group_id, max_length)
    if not padding:
        return pair
    full = replace(
        pair.full_sample,
        text=f"{pair.full_sample.text} {padding}",
        difficulty_level=max(pair.full_sample.difficulty_level, 2),
        distractor_count=pair.full_sample.distractor_count + 1,
    )
    counterfactual = replace(
        pair.counterfactual_sample,
        text=f"{pair.counterfactual_sample.text} {padding}",
        difficulty_level=max(pair.counterfactual_sample.difficulty_level, 2),
        distractor_count=pair.counterfactual_sample.distractor_count + 1,
    )
    return replace(pair, full_sample=full, counterfactual_sample=counterfactual)


def _pad_groups(groups: list[SurfaceGroupCandidateBatch], max_length: int) -> list[SurfaceGroupCandidateBatch]:
    return [
        SurfaceGroupCandidateBatch(
            group_type=group.group_type,
            surface_group_id=group.surface_group_id,
            pairs=tuple(_pad_pair(pair, max_length) for pair in group.pairs),
        )
        for group in groups
    ]


def _group_texts(group: SurfaceGroupCandidateBatch) -> list[str]:
    texts: list[str] = []
    for pair in group.pairs:
        texts.append(pair.full_sample.text)
        texts.append(pair.counterfactual_sample.text)
    return texts


def _truncation_audit_rows(tokenizer, groups: list[SurfaceGroupCandidateBatch], max_length: int, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        texts = _group_texts(group)
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        for index, text in enumerate(texts):
            token_count = len(encoded["input_ids"][index])
            truncated = token_count > max_length
            rows.append(
                {
                    "split": split,
                    "group_type": group.group_type,
                    "surface_group_id": group.surface_group_id,
                    "sample_index": index,
                    "max_length": max_length,
                    "original_token_count": token_count,
                    "truncated_token_count": min(token_count, max_length),
                    "truncated": truncated,
                    "core_logic_may_be_lost": truncated,
                    "text_has_padding": "neutral_context_padding_v1" in text,
                }
            )
    return rows


@contextmanager
def _patched_group_builders(
    max_length: int,
    group_store: list[tuple[str, list[SurfaceGroupCandidateBatch]]],
    truncation_store: list[dict[str, Any]],
    tokenizer,
) -> Iterator[None]:
    from . import group_full_hidden_centroid_integration as stage42_module
    from . import memory_group_gate_repair as memory_split_module
    from . import rule_conflict_group_recovery as group_split_module

    original_memory_builder = memory_split_module._build_memory_group_splits
    original_group_builder = group_split_module._build_group_splits
    original_stage42_group_builder = stage42_module._build_group_splits

    def memory_builder_wrapper(*args, **kwargs):
        train, heldout, manifest = original_memory_builder(*args, **kwargs)
        train = _pad_groups(train, max_length)
        heldout = _pad_groups(heldout, max_length)
        group_store.append(("train", train))
        group_store.append(("heldout", heldout))
        rows = _truncation_audit_rows(tokenizer, train, max_length, "train") + _truncation_audit_rows(tokenizer, heldout, max_length, "heldout")
        truncation_store.extend(rows)
        if any(row["core_logic_may_be_lost"] for row in rows):
            raise RuntimeError("core logic truncation detected in Stage43 memory groups")
        manifest = dict(manifest)
        manifest["stage43_long_context_padding"] = max_length >= LONG_CONTEXT_MIN_LENGTH
        return train, heldout, manifest

    def group_builder_wrapper(*args, **kwargs):
        train, heldout, manifest = original_group_builder(*args, **kwargs)
        train = _pad_groups(train, max_length)
        heldout = _pad_groups(heldout, max_length)
        group_store.append(("train", train))
        group_store.append(("heldout", heldout))
        rows = _truncation_audit_rows(tokenizer, train, max_length, "train") + _truncation_audit_rows(tokenizer, heldout, max_length, "heldout")
        truncation_store.extend(rows)
        if any(row["core_logic_may_be_lost"] for row in rows):
            raise RuntimeError("core logic truncation detected in Stage43 groups")
        manifest = dict(manifest)
        manifest["stage43_long_context_padding"] = max_length >= LONG_CONTEXT_MIN_LENGTH
        return train, heldout, manifest

    memory_split_module._build_memory_group_splits = memory_builder_wrapper
    group_split_module._build_group_splits = group_builder_wrapper
    stage42_module._build_group_splits = group_builder_wrapper
    try:
        yield
    finally:
        memory_split_module._build_memory_group_splits = original_memory_builder
        group_split_module._build_group_splits = original_group_builder
        stage42_module._build_group_splits = original_stage42_group_builder


def _metric_row(seed: int, max_length: int, summary: dict[str, Any]) -> dict[str, Any]:
    final_fixed = summary.get("final_fixed_centroid_accuracy", {})
    wrong_drop = summary.get("wrong_context_fixed_centroid_drop", {})
    return {
        "seed": seed,
        "max_length": max_length,
        "passes_stage_gate": bool(summary.get("passes_stage_gate")),
        "fixed_centroid_before": float(summary.get("fixed_centroid_average_before", 0.0)),
        "fixed_centroid_after": float(summary.get("fixed_centroid_average_after", 0.0)),
        "fixed_centroid_improvement": float(summary.get("fixed_centroid_average_after", 0.0))
        - float(summary.get("fixed_centroid_average_before", 0.0)),
        "memory_fixed_centroid": float(final_fixed.get("memory_necessity_group", 0.0)),
        "rule_fixed_centroid": float(final_fixed.get("rule_necessity_group", 0.0)),
        "conflict_fixed_centroid": float(final_fixed.get("memory_rule_conflict_group", 0.0)),
        "combined_fixed_centroid": float(final_fixed.get("combined", 0.0)),
        "wrong_context_drop": float(wrong_drop.get("combined", 0.0)),
        "hidden_norm_ratio": float(summary.get("hidden_norm_ratio", 0.0)),
        "qwen_trainable_parameters": float(summary.get("qwen_trainable_parameters", -1.0)),
        "qwen_gradients": float(summary.get("qwen_gradients", -1.0)),
        "weights_unchanged": bool(summary.get("weights_unchanged")),
        "raw_full_hidden_residual_scale": float(summary.get("raw_full_hidden_residual_scale", 0.0)),
        "failed_stage": summary.get("failed_stage"),
        "failed_gate": summary.get("failed_gate"),
        "runtime_seconds": float(summary.get("runtime_seconds", 0.0)),
    }


def _stage40_row(seed: int, max_length: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "max_length": max_length,
        "passes_stage_gate": bool(summary.get("passes_stage_gate")),
        "memory_delta_to_option_accuracy": float(summary.get("memory_delta_to_option_accuracy", 0.0)),
        "memory_necessity_group_success": float(summary.get("memory_necessity_group_success", 0.0)),
        "no_memory_drop": float(summary.get("no_memory_drop", 0.0)),
        "no_rule_drop": float(summary.get("no_rule_drop", 0.0)),
        "wrong_context_drop": float(summary.get("wrong_context_drop", 0.0)),
        "hidden_norm_ratio": float(summary.get("hidden_norm_ratio", 0.0)),
        "checkpoint_path": summary.get("checkpoint_path"),
    }


def _stage41_rows(seed: int, max_length: int, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = summary.get("final_metrics", {})
    for group_type, values in metrics.items():
        rows.append(
            {
                "seed": seed,
                "max_length": max_length,
                "group_type": group_type,
                "passes_stage_gate": bool(summary.get("passes_stage_gate")),
                "accuracy": float(values.get("accuracy", 0.0)),
                "group_success": float(values.get("group_success", 0.0)),
                "no_memory_path_drop": float(values.get("no_memory_path_drop", 0.0)),
                "no_rule_path_drop": float(values.get("no_rule_path_drop", 0.0)),
                "wrong_context_drop": float(values.get("wrong_context_drop", 0.0)),
            }
        )
    return rows


def _mean_std_by_seq(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "fixed_centroid_after",
        "fixed_centroid_improvement",
        "wrong_context_drop",
        "hidden_norm_ratio",
    )
    result: list[dict[str, Any]] = []
    for max_length in sorted({int(row["max_length"]) for row in rows}):
        current = [row for row in rows if int(row["max_length"]) == max_length and row["passes_stage_gate"]]
        for metric in metrics:
            mean, std = _mean_std([float(row[metric]) for row in current])
            result.append({"max_length": max_length, "metric": metric, "mean": mean, "std": std, "n": len(current)})
    return result


def _seed_stability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        current = [row for row in rows if int(row["seed"]) == seed]
        result.append(
            {
                "seed": seed,
                "run_count": len(current),
                "all_passed": all(row["passes_stage_gate"] for row in current),
                "min_fixed_centroid_after": min(float(row["fixed_centroid_after"]) for row in current),
                "max_hidden_norm_ratio": max(float(row["hidden_norm_ratio"]) for row in current),
            }
        )
    return result


def _seq_len_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[int(row["max_length"])] = row
    for seed, values in sorted(by_seed.items()):
        if 64 in values and 128 in values:
            result.append(
                {
                    "seed": seed,
                    "fixed_centroid_after_len64": values[64]["fixed_centroid_after"],
                    "fixed_centroid_after_len128": values[128]["fixed_centroid_after"],
                    "len128_drop_vs_64": float(values[64]["fixed_centroid_after"]) - float(values[128]["fixed_centroid_after"]),
                    "hidden_norm_ratio_len64": values[64]["hidden_norm_ratio"],
                    "hidden_norm_ratio_len128": values[128]["hidden_norm_ratio"],
                }
            )
    return result


def _stage43_gate_failures(rows: list[dict[str, Any]], truncation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if any(row["core_logic_may_be_lost"] for row in truncation_rows):
        failures.append({"failed_stage": "dataset", "failed_gate": "core_logic_truncation"})
    for row in rows:
        gates = (
            ("run_passes_stage_gate", row["passes_stage_gate"], True, "eq"),
            ("qwen_trainable_parameters", row["qwen_trainable_parameters"], 0.0, "eq"),
            ("qwen_gradients", row["qwen_gradients"], 0.0, "eq"),
            ("weights_unchanged", row["weights_unchanged"], True, "eq"),
            ("residual_scale", row["raw_full_hidden_residual_scale"], RAW_FULL_HIDDEN_RESIDUAL_SCALE, "eq"),
            ("hidden_norm_ratio", row["hidden_norm_ratio"], 2.0, "le"),
        )
        for name, actual, expected, op in gates:
            if op == "eq":
                passed = actual == expected
            elif op == "le":
                passed = float(actual) <= float(expected)
            else:
                passed = float(actual) >= float(expected)
            if not passed:
                failures.append(
                    {
                        "seed": row["seed"],
                        "max_length": row["max_length"],
                        "failed_stage": "per_run_gate",
                        "failed_gate": name,
                        "actual_value": actual,
                        "expected_threshold": expected,
                    }
                )
    for max_length in sorted({int(row["max_length"]) for row in rows}):
        current = [row for row in rows if int(row["max_length"]) == max_length and row["passes_stage_gate"]]
        after_mean, after_std = _mean_std([float(row["fixed_centroid_after"]) for row in current])
        hidden_mean, hidden_std = _mean_std([float(row["hidden_norm_ratio"]) for row in current])
        if len(current) and (after_mean is None or after_mean < 0.90):
            failures.append({"max_length": max_length, "failed_stage": "residual_scale_stability", "failed_gate": "fixed_centroid_after_mean", "actual_value": after_mean, "expected_threshold": 0.90})
        if len(current) and (after_std is None or after_std > 0.08):
            failures.append({"max_length": max_length, "failed_stage": "residual_scale_stability", "failed_gate": "fixed_centroid_after_std", "actual_value": after_std, "expected_threshold": 0.08})
        if len(current) and (hidden_std is None or hidden_std > 0.08):
            failures.append({"max_length": max_length, "failed_stage": "residual_scale_stability", "failed_gate": "hidden_norm_ratio_std", "actual_value": hidden_std, "expected_threshold": 0.08})
    comparisons = _seq_len_comparison(rows)
    for row in comparisons:
        if float(row["len128_drop_vs_64"]) > 0.08:
            failures.append({"seed": row["seed"], "failed_stage": "seq_len_stability", "failed_gate": "len128_fixed_centroid_drop", "actual_value": row["len128_drop_vs_64"], "expected_threshold": 0.08})
    return failures


def run_qwen3_stage43_residual_scale_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seeds: tuple[int, ...] = (202, 303, 404, 505, 606),
    max_lengths: tuple[int, ...] = (64, 128),
    preferred_device: str | None = None,
    local_samples_per_label: int = 24,
    local_train_groups: int = 16,
    stage40_steps: int = 80,
    rule_steps: int = 80,
    conflict_steps: int = 80,
    combined_steps: int = 100,
    projector_sanity_steps: int = 20,
    stage42_alignment_steps: int = 100,
    strict_stage_gates: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    stage40_rows: list[dict[str, Any]] = []
    stage41_rows: list[dict[str, Any]] = []
    truncation_rows: list[dict[str, Any]] = []
    training_runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for max_length in max_lengths:
        for seed in seeds:
            run_dir = _run_dir(output, seed, max_length)
            print(
                f"stage43_residual_scale_stress_progress seed={seed} max_length={max_length} stage=begin",
                flush=True,
            )
            group_store: list[tuple[str, list[SurfaceGroupCandidateBatch]]] = []
            run_truncation_rows: list[dict[str, Any]] = []
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
                with _patched_group_builders(max_length, group_store, run_truncation_rows, tokenizer):
                    print(f"stage43_residual_scale_stress_progress seed={seed} max_length={max_length} stage=stage40", flush=True)
                    stage40_summary = run_qwen3_memory_delta_gradient_diagnostic(
                        output_dir=run_dir / "stage40",
                        model_path=model_path,
                        seed=seed,
                        local_samples_per_label=local_samples_per_label,
                        local_train_groups=local_train_groups,
                        repair_steps=stage40_steps,
                        max_length=max_length,
                        preferred_device=preferred_device,
                    )
                    stage40_rows.append(_stage40_row(seed, max_length, stage40_summary))
                    if not stage40_summary.get("passes_stage_gate"):
                        if strict_stage_gates:
                            raise RuntimeError(f"Stage40 failed for seed {seed} length {max_length}: {stage40_summary.get('failed_gate')}")
                    _cleanup_cuda()

                    print(f"stage43_residual_scale_stress_progress seed={seed} max_length={max_length} stage=stage41", flush=True)
                    stage41_summary = run_qwen3_rule_conflict_group_recovery(
                        output_dir=run_dir / "stage41",
                        model_path=model_path,
                        stage40_checkpoint=Path(stage40_summary["checkpoint_path"]),
                        seed=seed,
                        local_samples_per_label=local_samples_per_label,
                        local_train_groups=local_train_groups,
                        rule_steps=rule_steps,
                        conflict_steps=conflict_steps,
                        combined_steps=combined_steps,
                        projector_sanity_steps=projector_sanity_steps,
                        max_length=max_length,
                        preferred_device=preferred_device,
                        strict_stage_gates=strict_stage_gates,
                    )
                    stage41_rows.extend(_stage41_rows(seed, max_length, stage41_summary))
                    if not stage41_summary.get("passes_stage_gate"):
                        if strict_stage_gates:
                            raise RuntimeError(f"Stage41 failed for seed {seed} length {max_length}: {stage41_summary.get('failed_gate')}")
                    _cleanup_cuda()

                    print(f"stage43_residual_scale_stress_progress seed={seed} max_length={max_length} stage=stage42", flush=True)
                    stage42_summary = run_qwen3_group_full_hidden_centroid_integration(
                        output_dir=run_dir / "stage42",
                        model_path=model_path,
                        stage41_checkpoint=_stage41_checkpoint(run_dir, seed),
                        seed=seed,
                        local_samples_per_label=local_samples_per_label,
                        local_train_groups=local_train_groups,
                        alignment_steps=stage42_alignment_steps,
                        max_length=max_length,
                        preferred_device=preferred_device,
                        strict_stage_gates=strict_stage_gates,
                    )
                    row = _metric_row(seed, max_length, stage42_summary)
                    run_rows.append(row)
                    if not stage42_summary.get("passes_stage_gate"):
                        failures.append({"seed": seed, "max_length": max_length, "failed_stage": stage42_summary.get("failed_stage"), "failed_gate": stage42_summary.get("failed_gate")})
                    training_runs.append(
                        {
                            "seed": seed,
                            "max_length": max_length,
                            "stage40_checkpoint": stage40_summary.get("checkpoint_path"),
                            "stage41_checkpoint": str(_stage41_checkpoint(run_dir, seed)),
                            "stage42_checkpoint": stage42_summary.get("alignment_checkpoint"),
                            "passes_stage_gate": row["passes_stage_gate"],
                        }
                    )
                    _cleanup_cuda()
                seen: set[tuple[str, str, str, int]] = set()
                deduped_truncation_rows: list[dict[str, Any]] = []
                for split, groups in group_store:
                    for row in _truncation_audit_rows(tokenizer, groups, max_length, split):
                        key = (split, row["group_type"], row["surface_group_id"], int(row["sample_index"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        row["seed"] = seed
                        deduped_truncation_rows.append(row)
                truncation_rows.extend(deduped_truncation_rows)
                _write_csv(run_dir / "truncation_audit.csv", deduped_truncation_rows)
            except Exception as exc:  # noqa: BLE001 - failed run must be recorded.
                for row in run_truncation_rows:
                    row["seed"] = seed
                truncation_rows.extend(run_truncation_rows)
                _write_csv(run_dir / "truncation_audit.csv", run_truncation_rows)
                failures.append({"seed": seed, "max_length": max_length, "failed_stage": "exception", "failed_gate": type(exc).__name__, "message": str(exc)})
                training_runs.append({"seed": seed, "max_length": max_length, "passes_stage_gate": False, "error": str(exc)})
                print(f"stage43_residual_scale_stress_progress seed={seed} max_length={max_length} stage=failed error={exc}", flush=True)
                break
        if failures:
            break
    gate_failures = _stage43_gate_failures(run_rows, truncation_rows)
    failures.extend(gate_failures)
    mean_std_rows = _mean_std_by_seq(run_rows)
    seed_rows = _seed_stability_rows(run_rows)
    seq_rows = _seq_len_comparison(run_rows)
    passed_run_count = sum(1 for row in run_rows if row["passes_stage_gate"])
    expected_run_count = len(seeds) * len(max_lengths)
    summary = {
        "benchmark": "stage43_residual_scale_stress",
        "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        "seeds": list(seeds),
        "max_lengths": list(max_lengths),
        "completed_run_count": len(run_rows),
        "passed_run_count": passed_run_count,
        "expected_run_count": expected_run_count,
        "passes_stage_gate": len(run_rows) == expected_run_count and passed_run_count == expected_run_count and not failures,
        "allows_stage44": len(run_rows) == expected_run_count and passed_run_count == expected_run_count and not failures,
        "failures": failures,
    }
    _write_csv(output / "per_run_metrics.csv", run_rows)
    _write_csv(output / "mean_std_by_seq_len.csv", mean_std_rows)
    _write_csv(output / "seed_stability.csv", seed_rows)
    _write_csv(output / "seq_len_comparison.csv", seq_rows)
    _write_csv(output / "stage40_gate_metrics.csv", stage40_rows)
    _write_csv(output / "stage41_gate_metrics.csv", stage41_rows)
    _write_csv(output / "stage42_fixed_centroid_metrics.csv", run_rows)
    _write_csv(output / "hidden_norm_metrics.csv", [{"seed": row["seed"], "max_length": row["max_length"], "hidden_norm_ratio": row["hidden_norm_ratio"]} for row in run_rows])
    _write_csv(output / "wrong_context_metrics.csv", [{"seed": row["seed"], "max_length": row["max_length"], "wrong_context_drop": row["wrong_context_drop"]} for row in run_rows])
    _write_csv(output / "truncation_audit.csv", truncation_rows)
    _json_dump(output / "training_runs.json", training_runs)
    _json_dump(output / "failure_cases.json", failures)
    _json_dump(output / "summary.json", summary)
    return summary


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 43 residual scale 200 stress benchmark.")
    parser.add_argument("--seeds", default="202,303,404,505,606")
    parser.add_argument("--max-lengths", default="64,128")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "seeds": _parse_ints(args.seeds),
        "max_lengths": _parse_ints(args.max_lengths),
        "preferred_device": args.preferred_device,
    }
    if args.smoke:
        kwargs.update(
            seeds=(202,),
            max_lengths=(64,),
            local_samples_per_label=4,
            local_train_groups=2,
            stage40_steps=24,
            rule_steps=12,
            conflict_steps=12,
            combined_steps=16,
            projector_sanity_steps=8,
            stage42_alignment_steps=12,
            strict_stage_gates=False,
        )
    summary = run_qwen3_stage43_residual_scale_stress(**kwargs)
    print(
        "qwen3_stage43_residual_scale_stress_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "completed_run_count": summary["completed_run_count"],
                "passed_run_count": summary["passed_run_count"],
                "failures": summary["failures"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
