from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..device import resolve_device
from ..memory import MemoryItem
from ..model import MiniTransformerTorch, TransformerConfigTorch
from ..rules import RuleEngineTorch, RuleItem
from ..state import StateConfig
from .alignment import run_alignment_training
from .codebook import build_logic_codebook_train_test, run_codebook_generalization_matrix
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, LogicSample, build_logic_variant_datasets
from .fusion import CivilizationFusionController
from .hidden_states import samples_to_tensor
from .injection import LogicCodeInjector, nearest_centroid_label
from .run_hidden_injection import run_hidden_injection_analysis


def _split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


def _memory_for_label(label: str) -> list[MemoryItem]:
    content = {
        "causality": "causal memory because cause leads to effect",
        "negation": "negation memory not never deny reject",
        "conflict": "conflict memory contradiction always never",
        "priority": "priority memory critical urgent override",
        "condition": "condition memory if then threshold",
    }[label]
    return [MemoryItem(id=f"m-{label}", summary=f"{label} memory", content=content, relation_type=label, priority=1.0, confidence=1.0)]


def _rule_engine(mode: str, model_dim: int, device: torch.device) -> tuple[RuleEngineTorch, str]:
    if mode == "blocked":
        text = "unsafe fusion request"
    elif mode == "rule_gated":
        text = "evidence says always and never"
    else:
        text = "evidence preferred"
    rules = [
        RuleItem("r-hard", "hard", "unsafe", "block injection", 1.0, "stage14"),
        RuleItem("r-soft", "soft", "evidence", "reduce alpha for evidence preference", 0.7, "stage14"),
        RuleItem("r-conflict", "conflict", "always|never", "record conflict", 1.0, "stage14"),
    ]
    return RuleEngineTorch(rules, model_dim=model_dim, device=device), text


@torch.no_grad()
def _pooled(model: MiniTransformerTorch, samples: list[LogicSample], device: torch.device) -> np.ndarray:
    output = model(samples_to_tensor(samples, device))
    pooled = output.hidden_states[-1].mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled hidden states contain NaN or Inf")
    return pooled.detach().cpu().numpy()


def _evaluate_mode(
    model: MiniTransformerTorch,
    sample: LogicSample,
    centroids: dict[str, np.ndarray],
    controller: CivilizationFusionController,
    mode: str,
    device: torch.device,
) -> dict:
    input_ids = samples_to_tensor([sample], device)
    baseline = model(input_ids)
    baseline_vector = baseline.hidden_states[-1].mean(dim=1).detach().cpu().numpy()[0]
    baseline_label, baseline_distance = nearest_centroid_label(baseline_vector, centroids)
    forced_target = sample.label
    if mode == "wrong_label":
        forced_target = LOGIC_LABELS[(LOGIC_LABELS.index(sample.label) + 1) % len(LOGIC_LABELS)]
    state = StateConfig(rigor=0.85, creativity=0.10, defensiveness=0.90)
    if mode == "state_guided":
        state = StateConfig(rigor=0.10, creativity=0.85, defensiveness=0.10)
    memory_items = _memory_for_label(forced_target if mode in {"memory_guided", "full_fusion", "wrong_label"} else sample.label)
    engine, rule_text = _rule_engine(mode, model.config.model_dim, device)
    rule_result = engine.evaluate(rule_text)
    decision_mode = "full_fusion" if mode == "wrong_label" else mode
    if mode == "baseline":
        injected = baseline
        decision = None
        fusion_trace = {"mode": "baseline", "blocked": False}
    else:
        decision = controller.decide(
            baseline_label=baseline_label,
            memory_items=memory_items,
            state=state,
            rule_result=rule_result,
            mode=decision_mode,
            forced_target_label=forced_target if mode in {"codebook_only", "wrong_label"} else None,
        )
        injector = controller.build_injector(decision)
        if injector is None:
            injected = baseline
            fusion_trace = {**decision.trace, "injector_trace": None}
        else:
            injected = model(input_ids, hidden_injection_hook=injector)
            fusion_trace = {**decision.trace, "injector_trace": injected.injection_traces[0] if injected.injection_traces else None}

    injected_vector = injected.hidden_states[-1].mean(dim=1).detach().cpu().numpy()[0]
    injected_label, injected_distance = nearest_centroid_label(injected_vector, centroids)
    logits_delta_norm = float(torch.linalg.vector_norm(injected.logits - baseline.logits).detach().cpu().item())
    hidden_delta_norm = float(np.linalg.norm(injected_vector - baseline_vector))
    injector_trace = fusion_trace.get("injector_trace") or {}
    return {
        "variant": sample.variant,
        "true_label": sample.label,
        "template_id": sample.template_id,
        "mode": mode,
        "target_label": forced_target if mode != "baseline" else baseline_label,
        "baseline_nearest_label": baseline_label,
        "baseline_distance": baseline_distance,
        "injected_nearest_label": injected_label,
        "injected_distance": injected_distance,
        "target_hit": mode == "baseline" or injected_label == (forced_target if mode != "alpha_zero" else baseline_label),
        "wrong_label_drift": mode == "wrong_label" and injected_label == forced_target,
        "blocked": bool(fusion_trace.get("blocked", False)),
        "block_reason": fusion_trace.get("block_reason"),
        "alpha": fusion_trace.get("alpha", 0.0),
        "strategy": fusion_trace.get("strategy"),
        "logits_delta_norm": logits_delta_norm,
        "hidden_delta_norm": hidden_delta_norm,
        "zero_equivalent": mode != "alpha_zero" or (logits_delta_norm == 0.0 and hidden_delta_norm == 0.0),
        "hidden_norm_ratio": injector_trace.get("hidden_norm_ratio"),
        "warning": injector_trace.get("warning"),
        "fusion_trace": fusion_trace,
    }


def run_fusion_injection_analysis(
    output_dir: str | Path = "artifacts/torch-line/fusion_injection",
    seeds: tuple[int, ...] = (202, 303, 404),
    samples_per_label: int = 60,
    train_per_label: int = 40,
    max_seq_len: int = 18,
    device: str | torch.device | None = "cpu",
    run_regressions: bool = True,
) -> dict:
    target_device = resolve_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage11 = run_codebook_generalization_matrix(
        output_dir=output_path / "stage11_regression",
        seeds=seeds,
        samples_per_label=samples_per_label,
        train_per_label=train_per_label,
        device=target_device,
        template_bank="expanded_v1",
        model_training_mode="cross_template_alignment",
        max_seq_len=max_seq_len,
    )
    stage13 = None
    if run_regressions:
        stage13 = run_hidden_injection_analysis(
            output_dir=output_path / "stage13_regression",
            seeds=seeds,
            samples_per_label=samples_per_label,
            train_per_label=train_per_label,
            max_seq_len=max_seq_len,
            device=target_device,
        )

    rows: list[dict] = []
    modes = ("baseline", "codebook_only", "memory_guided", "state_guided", "rule_gated", "full_fusion", "alpha_zero", "blocked", "wrong_label")
    for seed in seeds:
        datasets, tokenizer = build_logic_variant_datasets(samples_per_label=samples_per_label, max_seq_len=max_seq_len, seed=seed, template_bank="expanded_v1")
        config = TransformerConfigTorch(vocab_size=tokenizer.vocab_size, model_dim=24, hidden_dim=48, num_heads=4, num_layers=2, max_seq_len=max_seq_len, seed=seed)
        torch.manual_seed(seed)
        model = MiniTransformerTorch(config).to(target_device)
        run_alignment_training(model, datasets, target_device, seed=seed, train_per_label=train_per_label)
        canonical_train, _ = _split_by_label(datasets["canonical"], train_per_label)
        train_vectors = _pooled(model, canonical_train, target_device)
        train_labels = np.array([sample.label for sample in canonical_train])
        codebook = build_logic_codebook_train_test(train_vectors, train_labels, train_vectors, train_labels)
        controller = CivilizationFusionController(codebook.centroids, model_dim=config.model_dim, injection_layer=config.num_layers - 1)
        for variant in LOGIC_VARIANTS:
            _, test_samples = _split_by_label(datasets[variant], train_per_label)
            for sample_index, sample in enumerate(test_samples):
                for mode in modes:
                    row = _evaluate_mode(model, sample, codebook.centroids, controller, mode, target_device)
                    row.update({"seed": seed, "sample_index": sample_index})
                    rows.append(row)

    summary = _summarize(rows, stage11, stage13, seeds, samples_per_label, train_per_label)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "fusion_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_path / "fusion_results.csv", rows)
    return summary


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _summarize(rows: list[dict], stage11: dict, stage13: dict | None, seeds: tuple[int, ...], samples_per_label: int, train_per_label: int) -> dict:
    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in sorted({row["mode"] for row in rows})}
    mode_hit_rates = {mode: _rate(mode_rows, "target_hit") for mode, mode_rows in by_mode.items()}
    zero_rate = _rate(by_mode["alpha_zero"], "zero_equivalent")
    block_rate = _rate(by_mode["blocked"], "blocked")
    wrong_drift = _rate(by_mode["wrong_label"], "wrong_label_drift")
    norm_exceeded = sum(1 for row in rows if row["hidden_norm_ratio"] is not None and row["hidden_norm_ratio"] > 2.0)
    missing_trace = sum(1 for row in rows if row["mode"] != "baseline" and not row["fusion_trace"])
    full_vs_codebook = mode_hit_rates["full_fusion"] >= mode_hit_rates["codebook_only"]
    stage13_pass = True if stage13 is None else bool(stage13["passes_stage_gate"])
    return {
        "seeds": list(seeds),
        "samples_per_label": samples_per_label,
        "train_per_label": train_per_label,
        "num_rows": len(rows),
        "mode_target_hit_rates": mode_hit_rates,
        "alpha_zero_equivalence_rate": zero_rate,
        "hard_rule_block_rate": block_rate,
        "wrong_label_drift_rate": wrong_drift,
        "hidden_norm_exceeded_count": norm_exceeded,
        "missing_trace_count": missing_trace,
        "full_fusion_not_worse_than_codebook_only": full_vs_codebook,
        "stage11_regression": {
            "allows_hidden_state_injection_planning": stage11["allows_hidden_state_injection_planning"],
            "trained_mean_average_accuracy_by_variant": stage11["trained_mean_average_accuracy_by_variant"],
            "masked_keyword_drop": stage11["masked_keyword_drop"],
        },
        "stage13_regression": None if stage13 is None else {
            "passes_stage_gate": stage13["passes_stage_gate"],
            "target_hit_rate": stage13["target_hit_rate"],
            "zero_equivalence_rate": stage13["zero_equivalence_rate"],
            "wrong_label_drift_rate": stage13["wrong_label_drift_rate"],
        },
        "passes_stage_gate": bool(
            stage11["allows_hidden_state_injection_planning"]
            and stage13_pass
            and zero_rate == 1.0
            and block_rate == 1.0
            and full_vs_codebook
            and wrong_drift <= 0.50
            and norm_exceeded == 0
            and missing_trace == 0
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [key for key in rows[0].keys() if key != "fusion_trace"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    summary = run_fusion_injection_analysis()
    print("mode_target_hit_rates", summary["mode_target_hit_rates"])
    print("alpha_zero_equivalence_rate", summary["alpha_zero_equivalence_rate"])
    print("hard_rule_block_rate", summary["hard_rule_block_rate"])
    print("wrong_label_drift_rate", summary["wrong_label_drift_rate"])
    print("hidden_norm_exceeded_count", summary["hidden_norm_exceeded_count"])
    print("missing_trace_count", summary["missing_trace_count"])
    print("full_fusion_not_worse_than_codebook_only", summary["full_fusion_not_worse_than_codebook_only"])
    print("passes_stage_gate", summary["passes_stage_gate"])


if __name__ == "__main__":
    main()
