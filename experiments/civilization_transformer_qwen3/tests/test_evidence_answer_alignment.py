from __future__ import annotations

from dataclasses import asdict
import json

import torch

from experiments.civilization_transformer_qwen3.analysis.evidence_answer_benchmark import (
    run_qwen3_evidence_answer_alignment,
)
from experiments.civilization_transformer_qwen3.analysis.evidence_answer_data import (
    LOCAL_ANSWER_OPTIONS,
    assert_no_logic_label_leakage,
    build_evidence_answer_samples,
    wrong_context_sample,
)
from experiments.civilization_transformer_qwen3.analysis.evidence_answer_training import (
    _deterministic_sequence,
    _margin_loss,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_data import (
    _normalize_external_row,
    build_local_semireal_task_records,
    records_to_logic_datasets,
)
import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def _write_rte_cache(path) -> None:
    rows = [
        {"sentence1": "A starts.", "sentence2": "A runs.", "label": 0},
        {"sentence1": "B starts.", "sentence2": "B is denied.", "label": 1},
        {"sentence1": "C starts.", "sentence2": "C runs.", "label": 0},
        {"sentence1": "D starts.", "sentence2": "D is denied.", "label": 1},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = _normalize_external_row(
                "glue_rte",
                row,
                index,
                context_grounding_mode="grounded_v1",
            )
            assert record is not None
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def test_answer_alignment_records_are_deterministic_and_context_paired() -> None:
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
    samples = datasets["operation_decision"]
    first = build_evidence_answer_samples(samples, "local_semireal")
    second = build_evidence_answer_samples(samples, "local_semireal")
    assert first == second
    assert_no_logic_label_leakage(first)
    assert {record.answer_options for record in first} == {LOCAL_ANSWER_OPTIONS}
    for record in first:
        wrong = wrong_context_sample(record)
        assert wrong.text == record.sample.text
        assert wrong.surface_group_id == record.context_pair_id
        assert wrong.label != record.sample.label
        assert record.answer_options[record.correct_option_id] == record.sample.expected_pattern
        assert record.wrong_context_option_id != record.correct_option_id


def test_deterministic_training_order_and_margin_loss() -> None:
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
    aligned = build_evidence_answer_samples(datasets["causal_trace"], "local_semireal")
    assert _deterministic_sequence(aligned, 12, 202) == _deterministic_sequence(aligned, 12, 202)
    correct_scores = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float32)
    wrong_scores = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
    correct_loss, correct_margin = _margin_loss(correct_scores, target=0, margin=0.2)
    wrong_loss, wrong_margin = _margin_loss(wrong_scores, target=0, margin=0.2)
    assert correct_loss < wrong_loss
    assert correct_margin > wrong_margin


def test_evidence_answer_alignment_smoke_writes_route_artifacts(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_rte_cache(cache_dir / "glue_rte.jsonl")
    summary = run_qwen3_evidence_answer_alignment(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=2,
        local_train_groups=1,
        external_train_per_label=1,
        external_heldout_per_label=1,
        external_task_names=("glue_rte",),
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
        "route_comparison.csv",
        "training_runs.json",
        "loss_curves.csv",
        "answer_option_metrics.csv",
        "fixed_centroid_metrics.csv",
        "path_ablation_drop.csv",
        "wrong_context_metrics.csv",
        "context_pair_metrics.csv",
        "external_task_metrics.csv",
        "language_preservation.csv",
        "resource_usage.json",
        "failure_cases.json",
        "truncation_cases.json",
    }
    assert required.issubset({path.name for path in (tmp_path / "out").iterdir()})
    assert summary["routes"] == ["local_only_transfer", "external_few_shot"]
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["stage_gates"]["separate_route_models"]
    assert summary["weights_unchanged"]
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 2
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints]
    assert all("adapter_state_dict" in payload for payload in payloads)
    assert all("diagnostic_heads_state_dict" in payload for payload in payloads)
    assert all("qwen_state_dict" not in payload for payload in payloads)
    assert any(
        not torch.equal(
            payloads[0]["adapter_state_dict"][key],
            payloads[1]["adapter_state_dict"][key],
        )
        for key in payloads[0]["adapter_state_dict"]
    )
