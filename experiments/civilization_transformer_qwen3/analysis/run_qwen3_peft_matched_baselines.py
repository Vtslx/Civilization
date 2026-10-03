from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .matched_baseline_benchmark import DEFAULT_OPTIMIZATION_STEPS
from .peft_matched_baseline_benchmark import SUPPORTED_PEFT_METHODS, run_peft_matched_baseline_seed


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = tuple(value for value in args.methods.split(",") if value)
    if set(methods) - set(SUPPORTED_PEFT_METHODS):
        raise ValueError(f"unsupported PEFT methods: {methods}")
    rows = []
    if args.aggregate_only:
        for seed in seeds:
            for method in methods:
                rows.append(json.loads((output / method / f"seed_{seed}" / "summary.json").read_text()))
    else:
        backend = Qwen3Backend(args.model_path, preferred_device=args.preferred_device)
        for seed in seeds:
            for method in methods:
                rows.append(run_peft_matched_baseline_seed(
                    backend=backend,
                    output_dir=output / method / f"seed_{seed}",
                    method=method,
                    seed=seed,
                    optimization_steps=args.steps,
                    local_samples_per_label=args.local_samples_per_label,
                    local_train_groups=args.local_train_groups,
                    max_length=args.max_length,
                ))
    aggregate = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for metric in ("accuracy", "wrong_context_drop", "memory_ablation_drop", "rule_ablation_drop", "runtime_seconds", "peak_cuda_allocated_bytes"):
            values = [float(row[metric]) for row in selected]
            aggregate.append({
                "method": method,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            })
    contract_audit = []
    first_round = Path(args.first_round_artifacts)
    for seed in seeds:
        hashes = {
            method: hashlib.sha256(
                (output / method / f"seed_{seed}" / "sample_contract.csv").read_bytes()
            ).hexdigest()
            for method in methods
        }
        first_round_path = first_round / "mlp_adapter" / f"seed_{seed}" / "sample_contract.csv"
        first_round_hash = (
            None
            if args.skip_first_round_contract
            else hashlib.sha256(first_round_path.read_bytes()).hexdigest()
        )
        contract_audit.append({
            "seed": seed,
            "method_hashes": hashes,
            "first_round_hash": first_round_hash,
            "all_identical": (
                len(set(hashes.values())) == 1
                and (first_round_hash is None or first_round_hash in set(hashes.values()))
            ),
        })
    if not all(row["all_identical"] for row in contract_audit):
        raise RuntimeError("round-two sample contract differs from round one")
    summary = {
        "benchmark": "qwen3_matched_baselines_round2_peft",
        "model": "Qwen3-0.6B",
        "seeds": seeds,
        "methods": methods,
        "optimization_steps": args.steps,
        "fairness_audit": {
            "sample_contracts": contract_audit,
            "same_model_sha256": len({row["model_sha256"] for row in rows}) == 1,
            "parameter_match_within_one_percent": all(
                row["parameter_match_relative_error"] < 0.01 for row in rows
            ),
        },
        "results": rows,
    }
    _write_csv(output / "per_seed_metrics.csv", rows)
    _write_csv(output / "mean_std_metrics.csv", aggregate)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/civilization_transformer_qwen3/artifacts/matched_baselines_round2_peft")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--methods", default=",".join(SUPPORTED_PEFT_METHODS))
    parser.add_argument("--steps", type=int, default=DEFAULT_OPTIMIZATION_STEPS)
    parser.add_argument("--local-samples-per-label", type=int, default=16)
    parser.add_argument("--local-train-groups", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--first-round-artifacts",
        default="experiments/civilization_transformer_qwen3/artifacts/matched_baselines_round1",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-first-round-contract", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = args.seeds.split(",")[0]
        args.steps = 2
        args.local_samples_per_label = 4
        args.local_train_groups = 2
        args.skip_first_round_contract = True
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
