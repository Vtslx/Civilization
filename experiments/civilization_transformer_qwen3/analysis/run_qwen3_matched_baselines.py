from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .matched_baseline_benchmark import (
    DEFAULT_OPTIMIZATION_STEPS,
    SUPPORTED_METHODS,
    run_matched_baseline_seed,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resource_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    value = payload.get("runtime_seconds", payload.get("seconds"))
    return float(value) if value is not None else None


def _civilization_reference_rows(root: Path, resource_audit_root: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        stage42 = root / "stage42" / f"seed_{seed}"
        summary_path = stage42 / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        drops = {}
        ablation_path = stage42 / "path_ablation_drop.csv"
        if ablation_path.exists():
            with ablation_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    drops[(row["group_type"], row["mode"])] = float(row["fixed_centroid_drop"])
        if seed == 202:
            stage40_resource = Path("experiments/civilization_transformer_qwen3/artifacts/memory_delta_gradient_diagnostic/resource_usage.json")
            stage41_resource = Path("experiments/civilization_transformer_qwen3/artifacts/rule_conflict_group_recovery/resource_usage.json")
        else:
            stage40_resource = root / "stage40" / f"seed_{seed}" / "resource_usage.json"
            stage41_resource = root / "stage41" / f"seed_{seed}" / "resource_usage.json"
        resource_parts = [
            _resource_seconds(stage40_resource),
            _resource_seconds(stage41_resource),
            _resource_seconds(stage42 / "resource_usage.json"),
        ]
        runtime = sum(value for value in resource_parts if value is not None)
        audited_resource_path = resource_audit_root / f"seed_{seed}" / "resource_usage.json"
        audited_resource = {}
        if audited_resource_path.exists():
            payload = json.loads(audited_resource_path.read_text(encoding="utf-8"))
            audited_resource = payload[0] if isinstance(payload, list) and payload else payload
        wrong_values = list(summary["wrong_context_fixed_centroid_drop"].values())
        memory_drop = drops.get(("memory_necessity_group", "no_memory_path"))
        rule_values = [
            drops[key] for key in drops
            if key[1] == "no_rule_path" and key[0] in {"rule_necessity_group", "memory_rule_conflict_group"}
        ]
        rows.append({
            "method": "civilization_path",
            "seed": seed,
            "model": "Qwen3-0.6B",
            "model_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
            "optimization_steps": 440,
            "reference_optimization_steps": 440,
            "trainable_parameter_count": 3_740_164,
            "parameter_match_relative_error": 0.0,
            "accuracy": float(summary["fixed_centroid_average_after"]),
            "wrong_context_drop": statistics.fmean(wrong_values),
            "memory_ablation_drop": memory_drop,
            "rule_ablation_drop": statistics.fmean(rule_values),
            "runtime_seconds": runtime,
            "runtime_scope": "historical_stage40_to_stage42_sum",
            "stage42_resource_audit_runtime_seconds": audited_resource.get("runtime_seconds"),
            "peak_cuda_allocated_bytes": audited_resource.get("peak_cuda_allocated_bytes"),
            "peak_cuda_reserved_bytes": audited_resource.get("peak_cuda_reserved_bytes"),
            "runtime_comparable": False,
            "memory_comparable": bool(audited_resource),
            "ablation_semantics": "native_hidden_state_path",
            "qwen_trainable_parameters": summary["qwen_trainable_parameters"],
            "qwen_gradients": summary["qwen_gradients"],
            "qwen_weights_unchanged": summary["weights_unchanged"],
        })
    return rows


def _contract_hash(path: Path) -> str:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    rows = []
    if args.aggregate_only:
        for seed in seeds:
            for method in methods:
                path = output / method / f"seed_{seed}" / "summary.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing completed baseline summary: {path}")
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        backend = Qwen3Backend(args.model_path, preferred_device=args.preferred_device)
        for seed in seeds:
            for method in methods:
                rows.append(run_matched_baseline_seed(
                    backend=backend,
                    output_dir=output / method / f"seed_{seed}",
                    method=method,
                    seed=seed,
                    optimization_steps=args.steps,
                    local_samples_per_label=args.local_samples_per_label,
                    local_train_groups=args.local_train_groups,
                    max_length=args.max_length,
                ))
    contract_rows = []
    for seed in seeds:
        hashes = {
            method: _contract_hash(output / method / f"seed_{seed}" / "sample_contract.csv")
            for method in methods
        }
        contract_rows.append({"seed": seed, "hashes": hashes, "all_methods_identical": len(set(hashes.values())) == 1})
    if not all(row["all_methods_identical"] for row in contract_rows):
        raise RuntimeError("matched baseline sample contracts differ across methods")
    reference_rows = _civilization_reference_rows(
        Path(args.civilization_artifacts), Path(args.civilization_resource_audit), seeds
    )
    all_rows = rows + reference_rows
    flat_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in all_rows
    ]
    aggregate = []
    metrics = ("accuracy", "wrong_context_drop", "memory_ablation_drop", "rule_ablation_drop", "runtime_seconds", "peak_cuda_allocated_bytes")
    aggregate_methods = list(methods) + (["civilization_path"] if reference_rows else [])
    for method in aggregate_methods:
        selected = [row for row in all_rows if row["method"] == method]
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            if not values:
                continue
            aggregate.append({
                "method": method,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            })
    summary = {
        "benchmark": "qwen3_matched_baselines_round1",
        "model": "Qwen3-0.6B",
        "seeds": seeds,
        "methods": methods,
        "optimization_steps": args.steps,
        "fairness_audit": {
            "sample_contracts": contract_rows,
            "same_model_sha256": len({row["model_sha256"] for row in all_rows}) == 1,
            "mlp_parameter_match_within_one_percent": all(
                row.get("parameter_match_relative_error", 0.0) < 0.01
                for row in rows if row["method"] == "mlp_adapter"
            ),
        },
        "results": all_rows,
    }
    _write_csv(output / "per_seed_metrics.csv", flat_rows)
    _write_csv(output / "mean_std_metrics.csv", aggregate)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched Qwen3 Civilization baselines.")
    parser.add_argument("--output-dir", default="experiments/civilization_transformer_qwen3/artifacts/matched_baselines_round1")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--steps", type=int, default=DEFAULT_OPTIMIZATION_STEPS)
    parser.add_argument("--local-samples-per-label", type=int, default=16)
    parser.add_argument("--local-train-groups", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--civilization-artifacts",
        default="experiments/civilization_transformer_qwen3/artifacts/stage42_multiseed_robustness",
    )
    parser.add_argument(
        "--civilization-resource-audit",
        default="experiments/civilization_transformer_qwen3/artifacts/matched_baselines_civilization_resource_audit",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.steps = 2
        args.seeds = args.seeds.split(",")[0]
        args.local_samples_per_label = 4
        args.local_train_groups = 2
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
