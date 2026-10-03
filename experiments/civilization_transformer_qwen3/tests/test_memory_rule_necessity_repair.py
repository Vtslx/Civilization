from __future__ import annotations

from dataclasses import asdict
import json

import torch

from experiments.civilization_transformer_qwen3.analysis.memory_rule_necessity_benchmark import (
    run_qwen3_memory_rule_necessity_repair,
)
from experiments.civilization_transformer_qwen3.analysis.memory_rule_necessity_training import (
    _deterministic_pair_sequence,
    _path_drop_loss,
)
from experiments.civilization_transformer_qwen3.analysis.necessity_alignment_data import (
    assert_necessity_pairs_valid,
    build_necessity_pairs,
)
from experiments.civilization_transformer_qwen3.analysis.evidence_answer_data import (
    build_evidence_answer_samples,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_data import (
    _normalize_external_row,
    build_local_semireal_task_records,
    records_to_logic_datasets,
)
from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS
import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def _write_external_cache(cache_dir) -> None:
    cache_dir.mkdir()
    rte_rows = [
        {"sentence1": "A starts.", "sentence2": "A runs.", "label": 0},
        {"sentence1": "B starts.", "sentence2": "B is denied.", "label": 1},
        {"sentence1": "C starts.", "sentence2": "C runs.", "label": 0},
        {"sentence1": "D starts.", "sentence2": "D is denied.", "label": 1},
    ]
    with (cache_dir / "glue_rte.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rte_rows):
            record = _normalize_external_row(
                "glue_rte",
                row,
                index,
                context_grounding_mode="grounded_v1",
            )
            assert record is not None
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")
    boolq_rows = [
        {"passage": "The service is available.", "question": "Is the service available?", "answer": True},
        {"passage": "The service is unavailable.", "question": "Is the service available?", "answer": False},
        {"passage": "The report is approved.", "question": "Is the report approved?", "answer": True},
        {"passage": "The report is rejected.", "question": "Is the report approved?", "answer": False},
    ]
    with (cache_dir / "boolq.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(boolq_rows):
            record = _normalize_external_row(
                "boolq",
                row,
                index,
                context_grounding_mode="grounded_v1",
            )
            assert record is not None
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def _local_records():
    records = build_local_semireal_task_records(
        2,
        seed=202,
        context_grounding_mode="grounded_v1",
    )
    datasets, _ = records_to_logic_datasets(
        records,
        max_seq_len=64,
        context_grounding_mode="grounded_v1",
    )
    return build_evidence_answer_samples(datasets["operation_decision"], "local_semireal")


def test_necessity_pairs_preserve_surface_and_change_only_required_context() -> None:
    pairs = build_necessity_pairs(_local_records())
    assert_necessity_pairs_valid(pairs)
    assert {pair.pair_type for pair in pairs} == {
        "memory_necessity_pair",
        "rule_necessity_pair",
        "memory_rule_conflict_pair",
    }
    forbidden = set(LOGIC_LABELS)
    for pair in pairs:
        assert pair.full_sample.text == pair.counterfactual_sample.text
        assert pair.expected_full_option_id != pair.expected_counterfactual_option_id
        text = f"{pair.full_sample.memory_target} {pair.full_sample.rule_target} {pair.counterfactual_context}".lower()
        assert not (forbidden & set(text.replace("_", " ").replace("-", " ").split()))
        if pair.pair_type == "memory_necessity_pair":
            assert pair.full_sample.rule_target == pair.counterfactual_sample.rule_target
            assert pair.full_sample.memory_target != pair.counterfactual_sample.memory_target
        if pair.pair_type == "rule_necessity_pair":
            assert pair.full_sample.memory_target == pair.counterfactual_sample.memory_target
            assert pair.full_sample.rule_target != pair.counterfactual_sample.rule_target


def test_necessity_pair_sequence_and_path_drop_loss_are_deterministic() -> None:
    pairs = build_necessity_pairs(_local_records())
    assert _deterministic_pair_sequence(pairs, 20, 202) == _deterministic_pair_sequence(pairs, 20, 202)
    full_scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    weak_ablated = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
    strong_ablated = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    weak_loss = _path_drop_loss(full_scores, weak_ablated, targets, 0.2)
    strong_loss = _path_drop_loss(full_scores, strong_ablated, targets, 0.2)
    assert weak_loss > strong_loss


def test_memory_rule_necessity_smoke_writes_required_artifacts(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_external_cache(cache_dir)
    summary = run_qwen3_memory_rule_necessity_repair(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=2,
        local_train_groups=1,
        external_train_per_label=1,
        external_heldout_per_label=1,
        external_task_names=("glue_rte", "boolq"),
        external_cache_dir=cache_dir,
        training_steps=1,
        gradient_accumulation=1,
        preferred_device="cpu",
        evaluation_batch_size=20,
        evaluation_modes=(
            "full",
            "adapter_disabled",
            "zero_scale",
            "no_memory_path",
            "no_state_path",
            "no_rule_path",
            "wrong_context",
        ),
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "necessity_pair_metrics.csv",
        "answer_option_metrics.csv",
        "fixed_centroid_metrics.csv",
        "path_ablation_drop.csv",
        "wrong_context_metrics.csv",
        "route_comparison.csv",
        "local_retention.csv",
        "external_task_metrics.csv",
        "language_preservation.csv",
        "resource_usage.json",
        "failure_cases.json",
        "truncation_cases.json",
    }
    assert required.issubset({path.name for path in (tmp_path / "out").iterdir()})
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["stage_gates"]["separate_route_models"]
    assert summary["weights_unchanged"]
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 2
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "qwen_state_dict" not in payload
