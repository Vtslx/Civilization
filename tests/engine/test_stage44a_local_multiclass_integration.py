from __future__ import annotations

from pathlib import Path

import pytest
import torch

from civilization.research.torch_line.analysis.dataset import LOGIC_LABELS
from civilization.engine.stages.real_task_data import (
    LOCAL_REAL_TASK_TYPES,
    build_local_semireal_task_records,
    records_to_logic_datasets,
)
from civilization.engine.stages.stage44a_local_multiclass_integration import (
    RAW_FULL_HIDDEN_RESIDUAL_SCALE,
    TASK_PATH,
    _context_ownership_audit,
    _verify_stage43_checkpoint,
    build_stage44a_groups,
    split_stage44a_groups,
)


def test_stage44a_path_grounded_groups_are_complete_and_path_owned() -> None:
    records = build_local_semireal_task_records(
        4, seed=202, context_grounding_mode="stage44a_path_grounded_v2"
    )
    datasets, _ = records_to_logic_datasets(
        records, max_seq_len=128, context_grounding_mode="stage44a_path_grounded_v2"
    )
    groups = build_stage44a_groups(datasets)
    assert {group.group_type for group in groups} == set(LOCAL_REAL_TASK_TYPES)
    assert len(groups) == len(LOCAL_REAL_TASK_TYPES) * 4
    for group in groups:
        assert len(group.pairs) == 5
        assert len({pair.full_sample.text for pair in group.pairs}) == 1
        assert {pair.expected_full_option_id for pair in group.pairs} == set(range(5))
        for pair in group.pairs:
            assert TASK_PATH[group.group_type] in pair.full_sample.required_paths
            assert pair.expected_full_option_id != pair.expected_counterfactual_option_id
            assert pair.full_sample.text == pair.counterfactual_sample.text
    rows, failures = _context_ownership_audit(records)
    assert rows
    assert not failures
    assert not any(row["logic_label_leakage"] for row in rows)


def test_stage44a_context_changes_only_the_declared_owner() -> None:
    records = build_local_semireal_task_records(
        2, seed=202, context_grounding_mode="stage44a_path_grounded_v2"
    )
    for task_type, task_records in records.items():
        group = task_records[: len(LOGIC_LABELS)]
        memory = {tuple((item.summary, item.content, item.relation_type) for item in row.memory_items) for row in group}
        rules = {tuple((item.condition, item.effect, item.source) for item in row.rule_items) for row in group}
        states = {row.state_values for row in group}
        if TASK_PATH[task_type] == "memory":
            assert len(memory) == 5
            assert len(rules) == 1
            assert len(states) == 1
        elif TASK_PATH[task_type] == "rule" and task_type != "rule_conflict":
            assert len(memory) == 1
            assert len(rules) == 5
            assert len(states) == 1
        elif task_type == "rule_conflict":
            assert len(memory) == 5
            assert len(rules) == 5
            assert len(states) == 1
            assert all("superseded" in row.memory_items[0].relation_type for row in group)
        else:
            assert len(memory) == 1
            assert len(rules) == 1
            assert len(states) == 5


def test_stage44a_split_is_seeded_and_surface_safe() -> None:
    records = build_local_semireal_task_records(
        6, seed=202, context_grounding_mode="stage44a_path_grounded_v2"
    )
    datasets, _ = records_to_logic_datasets(
        records, max_seq_len=128, context_grounding_mode="stage44a_path_grounded_v2"
    )
    groups = build_stage44a_groups(datasets)
    train_a, test_a = split_stage44a_groups(groups, 4, 202)
    train_b, test_b = split_stage44a_groups(groups, 4, 202)
    train_c, _ = split_stage44a_groups(groups, 4, 303)
    assert [group.surface_group_id for group in train_a] == [group.surface_group_id for group in train_b]
    assert [group.surface_group_id for group in test_a] == [group.surface_group_id for group in test_b]
    assert {group.surface_group_id for group in train_a}.isdisjoint(
        {group.surface_group_id for group in test_a}
    )
    assert [group.surface_group_id for group in train_a] != [group.surface_group_id for group in train_c]


def test_stage44a_checkpoint_verification_rejects_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "stage43"
    summary = root / "seed_202_len_128" / "stage42" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        '{"passes_stage_gate": true, "fixed_centroid_average_after": 1.0, '
        '"wrong_context_fixed_centroid_drop": {"combined": 0.8}, "hidden_norm_ratio": 1.2}',
        encoding="utf-8",
    )
    payload = {
        "training_metadata": {
            "seed": 303,
            "adapter_variant": "path_specific_v2",
            "target_layers": [16, 24],
            "raw_full_hidden_residual_scale": RAW_FULL_HIDDEN_RESIDUAL_SCALE,
        }
    }
    _verification, failures = _verify_stage43_checkpoint(
        root, tmp_path / "checkpoint.pt", 202, 128, payload
    )
    assert any(failure["failed_gate"] == "checkpoint_seed" for failure in failures)


def test_stage44a_checkpoint_payload_contains_no_qwen_state() -> None:
    payload = {
        "adapter_state_dict": {"layer.weight": torch.zeros(1)},
        "projector_state_dict": {},
        "full_hidden_projector_state_dict": {},
    }
    assert "qwen_state_dict" not in payload
    assert "model_state_dict" not in payload


def test_stage44a_split_rejects_no_heldout_group() -> None:
    records = build_local_semireal_task_records(
        2, seed=202, context_grounding_mode="stage44a_path_grounded_v2"
    )
    datasets, _ = records_to_logic_datasets(
        records, max_seq_len=64, context_grounding_mode="stage44a_path_grounded_v2"
    )
    with pytest.raises(ValueError, match="leave held-out"):
        split_stage44a_groups(build_stage44a_groups(datasets), 2, 202)


def test_stage44a_centroid_scope_is_per_task() -> None:
    records = build_local_semireal_task_records(
        3, seed=202, context_grounding_mode="stage44a_path_grounded_v2"
    )
    datasets, _ = records_to_logic_datasets(
        records, max_seq_len=64, context_grounding_mode="stage44a_path_grounded_v2"
    )
    groups = build_stage44a_groups(datasets)
    assert {group.group_type for group in groups} == set(LOCAL_REAL_TASK_TYPES)
    for task_type in LOCAL_REAL_TASK_TYPES:
        task_groups = [group for group in groups if group.group_type == task_type]
        assert len(task_groups) == 3
        assert all({pair.expected_full_option_id for pair in group.pairs} == set(range(5)) for group in task_groups)
