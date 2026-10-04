from __future__ import annotations

from dataclasses import replace
import json

import pytest

from civilization.engine.stages.evidence_answer_data import (
    EXTERNAL_ANSWER_OPTIONS,
)
from civilization.engine.stages.real_task_data import RealTaskRecord
from civilization.engine.stages.stage44b_external_recovery_audit import (
    audit_answer_option_mapping,
    audit_token_lengths,
    deterministic_external_split,
    project_external_source_records,
)
from civilization.research.torch_line.memory import MemoryItem
from civilization.research.torch_line.rules import RuleItem


def _record(index: int, label: str, text: str | None = None) -> RealTaskRecord:
    return RealTaskRecord(
        text=text or f"premise {index} hypothesis {index}",
        label="condition" if label == "entailment" else "negation",
        task_type="glue_rte",
        memory_items=(
            MemoryItem(
                id=f"memory_{index}",
                summary="legacy memory_selects_condition",
                content="legacy context",
                relation_type="legacy",
                priority=1.0,
                confidence=1.0,
            ),
        ),
        state_values=(0.5, 0.5, 0.5),
        rule_items=(
            RuleItem(
                id=f"rule_{index}",
                type="soft",
                condition="legacy",
                effect="rule_selects_condition",
                priority=1.0,
                source="test",
            ),
        ),
        expected_answer=label,
        surface_group_id=f"external_glue_rte_surface_{index:05d}",
        source_type="external_benchmark",
        source_id=f"glue_rte_{index:05d}",
        external_label=label,
    )


def test_source_projection_drops_legacy_context_and_keeps_external_options() -> None:
    projected = project_external_source_records(
        {"glue_rte": [_record(0, "entailment"), _record(1, "not_entailment")]}
    )
    assert projected["glue_rte"][0].answer_options == EXTERNAL_ANSWER_OPTIONS["glue_rte"]
    assert not hasattr(projected["glue_rte"][0], "memory_items")
    assert not hasattr(projected["glue_rte"][0], "rule_items")


def test_answer_option_mapping_uses_external_option_order() -> None:
    projected = project_external_source_records(
        {"glue_rte": [_record(0, "entailment"), _record(1, "not_entailment")]}
    )
    rows, failures = audit_answer_option_mapping(projected)
    assert not failures
    assert [(row["external_label"], row["score_column"]) for row in rows] == [
        ("entailment", 0),
        ("not_entailment", 1),
    ]


def test_external_split_is_deterministic_balanced_and_text_isolated() -> None:
    records = [
        _record(index, "entailment" if index % 2 == 0 else "not_entailment")
        for index in range(20)
    ]
    projected = project_external_source_records({"glue_rte": records})["glue_rte"]
    first = deterministic_external_split(
        projected, seed=202, train_per_label=4, heldout_per_label=3
    )
    second = deterministic_external_split(
        projected, seed=202, train_per_label=4, heldout_per_label=3
    )
    assert [[row.source_id for row in split] for split in first] == [
        [row.source_id for row in split] for split in second
    ]
    train, heldout = first
    assert {row.external_label for row in train} == {"entailment", "not_entailment"}
    assert {row.external_label for row in heldout} == {"entailment", "not_entailment"}
    assert {row.text_hash for row in train}.isdisjoint({row.text_hash for row in heldout})


def test_external_split_rejects_missing_heldout_label() -> None:
    records = [_record(0, "entailment"), _record(1, "not_entailment")]
    projected = project_external_source_records({"glue_rte": records})["glue_rte"]
    with pytest.raises(ValueError, match="heldout split is missing labels"):
        deterministic_external_split(
            projected, seed=202, train_per_label=1, heldout_per_label=1
        )


def test_external_split_reserves_heldout_for_small_classes() -> None:
    records = [
        _record(index, "entailment" if index < 5 else "not_entailment")
        for index in range(10)
    ]
    projected = project_external_source_records({"glue_rte": records})["glue_rte"]
    train, heldout = deterministic_external_split(
        projected, seed=202, train_per_label=12, heldout_per_label=12
    )
    assert len(train) == 4
    assert len(heldout) == 6
    assert {row.external_label for row in train} == {"entailment", "not_entailment"}
    assert {row.external_label for row in heldout} == {"entailment", "not_entailment"}


class _Tokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


def test_truncation_audit_requires_explicit_head_tail_policy() -> None:
    short = _record(0, "entailment", text="short premise hypothesis")
    long = replace(
        _record(1, "not_entailment"),
        text=" ".join(f"token{index}" for index in range(500)),
    )
    projected = project_external_source_records({"glue_rte": [short, long]})
    rows, failures = audit_token_lengths(
        _Tokenizer(), projected, max_length=128, tail_reserve=32
    )
    assert not failures
    assert rows[0]["truncation_policy"] == "none"
    assert rows[1]["truncation_policy"] == "head_tail"
    assert rows[1]["head_token_budget"] == 96
    assert rows[1]["tail_token_budget"] == 32
    assert not rows[1]["silent_truncation"]


def test_invalid_external_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="not valid"):
        project_external_source_records(
            {"glue_rte": [replace(_record(0, "entailment"), external_label="neutral")]}
        )
