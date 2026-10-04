from __future__ import annotations

import torch

from civilization.engine.stages.real_task_data import (
    RealTaskRecord,
    solidify_structured_external_cache,
)
from civilization.engine.stages.stage44b_external_recovery_audit import (
    Stage44BSourceRecord,
    project_external_source_records,
)
from civilization.engine.stages.stage44b_external_recovery_training import (
    TaskSpecificPathReadoutProjector,
    _evidence_snippet,
    _source_to_sample,
    _boolq_relevant_evidence,
    _hard_negative_source,
    _dataset_fields_cache_manifest,
    build_external_recovery_records,
    encode_head_tail,
    _token_safe_context_text,
    recover_external_task_boundary,
    stage44b1_answer_option_texts,
)
from civilization.research.torch_line.memory import MemoryItem
from civilization.research.torch_line.rules import RuleItem


def _source(index: int, label: str) -> Stage44BSourceRecord:
    options = ("entailment", "not_entailment")
    text = f"premise evidence {index} hypothesis statement {index}"
    return Stage44BSourceRecord(
        task_name="glue_rte",
        source_id=f"rte_{index}",
        surface_group_id=f"rte_surface_{index}",
        text=text,
        external_label=label,
        answer_options=options,
        correct_option_id=options.index(label),
        text_hash=str(index),
    )


def test_external_grounding_contains_no_target_label() -> None:
    source = _source(0, "entailment")
    sample = _source_to_sample(source)
    assert source.external_label not in sample.memory_target
    assert "not_entailment" not in sample.memory_target
    assert sample.required_paths == ("memory", "rule")
    assert sample.leakage_family == "stage44b_external_grounded_v1"


def test_wrong_context_uses_opposite_label_evidence_without_changing_surface() -> None:
    records = build_external_recovery_records(
        [_source(0, "entailment"), _source(1, "not_entailment")]
    )
    for record in records:
        assert record.sample.text == record.wrong_sample.text
        assert record.sample.memory_target != record.wrong_sample.memory_target
        assert record.wrong_option_id != record.source.correct_option_id


def test_evidence_snippet_preserves_head_and_tail() -> None:
    text = " ".join(f"token{index}" for index in range(100))
    snippet = _evidence_snippet(text, head_words=4, tail_words=5)
    assert snippet.startswith("token0 token1 token2 token3")
    assert snippet.endswith("token95 token96 token97 token98 token99")


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _text, **_kwargs):
        return {"input_ids": list(range(200))}

    def decode(self, ids, **_kwargs):
        return " ".join(str(value) for value in ids)


class _Backend:
    tokenizer = _Tokenizer()
    device = torch.device("cpu")


def test_head_tail_encoding_preserves_both_ends() -> None:
    encoded, truncated = encode_head_tail(_Backend(), "long text", 128, tail_reserve=32)
    ids = encoded["input_ids"][0].tolist()
    assert truncated
    assert ids[:3] == [0, 1, 2]
    assert ids[-3:] == [197, 198, 199]
    assert len(ids) == 128


def test_context_text_is_token_bounded() -> None:
    text = " ".join(f"token{index}" for index in range(100))
    bounded = _token_safe_context_text(_Backend(), text, max_tokens=32)
    assert len(bounded.split()) == 32
    assert bounded.startswith("0 1 2")
    assert bounded.endswith("197 198 199")


def test_stage44b1_recovers_boolq_passage_question_boundary() -> None:
    source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_0",
        surface_group_id="boolq_surface_0",
        text="The passage contains supporting evidence. is the evidence present",
        external_label="true",
        answer_options=("true", "false"),
        correct_option_id=0,
        text_hash="hash",
    )
    boundary = recover_external_task_boundary(source)
    sample = _source_to_sample(
        source,
        boundary=boundary,
        grounding_mode="stage44b1_grounded_v2",
    )
    assert boundary.boundary_ok
    assert boundary.evidence_text == "The passage contains supporting evidence."
    assert boundary.query_text == "is the evidence present"
    assert "Passage:" in sample.text
    assert "Question:" in sample.text
    assert "passage evidence:" in sample.memory_target
    assert "answer true" in sample.rule_target
    assert sample.leakage_family == "stage44b1_grounded_v2"


def test_stage44b1_wrong_context_keeps_surface_and_changes_memory() -> None:
    records = build_external_recovery_records(
        [
            _source(0, "entailment"),
            _source(1, "not_entailment"),
        ],
        grounding_mode="stage44b1_grounded_v2",
    )
    for record in records:
        assert record.sample.text == record.wrong_sample.text
        assert record.sample.memory_target != record.wrong_sample.memory_target
        assert record.sample.rule_target == record.wrong_sample.rule_target
        assert record.wrong_option_id != record.source.correct_option_id


def test_task_specific_projector_has_independent_task_parameters() -> None:
    projector = TaskSpecificPathReadoutProjector(("glue_rte", "boolq"))
    assert set(projector.projectors) == {"glue_rte", "boolq"}
    assert projector.projectors["glue_rte"] is not projector.projectors["boolq"]
    vector = torch.randn(2, 1024)
    assert projector(vector, "glue_rte").shape == (2, 1024)
    assert projector(vector, "boolq").shape == (2, 1024)


def test_stage44b1_answer_option_texts_are_task_grounded() -> None:
    boolq_options = stage44b1_answer_option_texts("boolq")
    assert boolq_options != ("true", "false")
    assert "passage" in boolq_options[0]
    assert "question" in boolq_options[1]
    assert len(stage44b1_answer_option_texts("super_glue_cb")) == 3


def test_stage44b2_boolq_relevant_evidence_selects_question_terms() -> None:
    source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_1",
        surface_group_id="boolq_surface_1",
        text=(
            "Cats sleep on mats. Sugarcane ethanol returns more energy than corn ethanol. "
            "Weather reports are unrelated. does sugarcane ethanol return more energy"
        ),
        external_label="true",
        answer_options=("true", "false"),
        correct_option_id=0,
        text_hash="hash",
    )
    boundary = recover_external_task_boundary(source)
    relevant = _boolq_relevant_evidence(boundary)
    assert "Sugarcane ethanol" in relevant
    assert "Weather reports" not in relevant


def test_stage44b2_records_use_relevant_boolq_grounding() -> None:
    true_source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_true",
        surface_group_id="boolq_surface_true",
        text="Alpha evidence is irrelevant. The bridge is open after repairs. is the bridge open",
        external_label="true",
        answer_options=("true", "false"),
        correct_option_id=0,
        text_hash="true",
    )
    false_source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_false",
        surface_group_id="boolq_surface_false",
        text="Beta evidence is irrelevant. The bridge remains closed after repairs. is the bridge open",
        external_label="false",
        answer_options=("true", "false"),
        correct_option_id=1,
        text_hash="false",
    )
    records = build_external_recovery_records(
        [true_source, false_source],
        grounding_mode="stage44b2_grounded_v3",
    )
    assert all(record.sample.leakage_family == "stage44b2_grounded_v3" for record in records)
    assert "Question:" in records[0].sample.text
    assert "bridge" in records[0].sample.memory_target
    assert records[0].sample.memory_target != records[0].wrong_sample.memory_target


def test_stage44b3_hard_negative_prefers_query_overlap() -> None:
    source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_source",
        surface_group_id="boolq_source_surface",
        text="The bridge is open after repairs. is the bridge open",
        external_label="true",
        answer_options=("true", "false"),
        correct_option_id=0,
        text_hash="source",
    )
    weak = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_weak",
        surface_group_id="boolq_weak_surface",
        text="The museum closes on Monday. is the museum open",
        external_label="false",
        answer_options=("true", "false"),
        correct_option_id=1,
        text_hash="weak",
    )
    hard = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_hard",
        surface_group_id="boolq_hard_surface",
        text="The bridge is closed for repairs. is the bridge open",
        external_label="false",
        answer_options=("true", "false"),
        correct_option_id=1,
        text_hash="hard",
    )
    selected = _hard_negative_source(
        source,
        [weak, hard],
        grounding_mode="stage44b3_grounded_v4",
        fallback_index=0,
    )
    assert selected.source_id == "boolq_hard"


def test_stage44b3_records_use_hard_negative_memory() -> None:
    true_source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_true",
        surface_group_id="boolq_surface_true",
        text="The bridge is open after repairs. is the bridge open",
        external_label="true",
        answer_options=("true", "false"),
        correct_option_id=0,
        text_hash="true",
    )
    false_source = Stage44BSourceRecord(
        task_name="boolq",
        source_id="boolq_false",
        surface_group_id="boolq_surface_false",
        text="The bridge is closed for repairs. is the bridge open",
        external_label="false",
        answer_options=("true", "false"),
        correct_option_id=1,
        text_hash="false",
    )
    records = build_external_recovery_records(
        [true_source, false_source],
        grounding_mode="stage44b3_grounded_v4",
    )
    assert records[0].sample.text == records[0].wrong_sample.text
    assert "closed" in records[0].wrong_sample.memory_target
    assert records[0].wrong_option_id == 1


def test_stage44b4_projection_recovers_structured_fields_from_cache_text() -> None:
    record = RealTaskRecord(
        text="A premise sentence supports the claim. the claim is supported",
        label="condition",
        task_type="glue_rte",
        memory_items=(),
        state_values=(0.5, 0.5, 0.5),
        rule_items=(),
        expected_answer="entailment",
        surface_group_id="surface",
        source_type="external_benchmark",
        source_id="glue_rte_1",
        external_label="entailment",
    )
    projected = project_external_source_records({"glue_rte": [record]})["glue_rte"][0]
    assert projected.premise == "A premise sentence supports the claim."
    assert projected.hypothesis == "the claim is supported"
    assert projected.structure_source == "parsed_text_tail"
    boundary = recover_external_task_boundary(projected)
    assert boundary.evidence_text == projected.premise
    assert boundary.query_text == projected.hypothesis


def test_stage44b4_wrong_context_is_same_sample_semantic_counterfactual() -> None:
    source = Stage44BSourceRecord(
        task_name="glue_rte",
        source_id="rte_true",
        surface_group_id="rte_surface_true",
        text="A premise sentence supports the claim. the claim is supported",
        external_label="entailment",
        answer_options=("entailment", "not_entailment"),
        correct_option_id=0,
        text_hash="true",
        premise="A premise sentence supports the claim.",
        hypothesis="the claim is supported",
        structure_source="parsed_text_tail",
    )
    opposite = Stage44BSourceRecord(
        task_name="glue_rte",
        source_id="rte_false",
        surface_group_id="rte_surface_false",
        text="Another premise omits the claim. the claim is supported",
        external_label="not_entailment",
        answer_options=("entailment", "not_entailment"),
        correct_option_id=1,
        text_hash="false",
        premise="Another premise omits the claim.",
        hypothesis="the claim is supported",
        structure_source="parsed_text_tail",
    )
    records = build_external_recovery_records(
        [source, opposite],
        grounding_mode="stage44b4_structured_semantic_v5",
    )
    record = records[0]
    assert record.sample.text == record.wrong_sample.text
    assert "A premise sentence supports the claim" in record.wrong_sample.memory_target
    assert "Another premise" not in record.wrong_sample.memory_target
    assert "does not provide enough support" in record.wrong_sample.memory_target
    assert record.wrong_option_id == 1


def test_stage44b5_structured_cache_solidifies_legacy_jsonl(tmp_path) -> None:
    source_cache = tmp_path / "external_cache"
    source_cache.mkdir()
    row = RealTaskRecord(
        text="A premise sentence supports the claim. the claim is supported",
        label="condition",
        task_type="glue_rte",
        memory_items=(
            MemoryItem(
                id="m",
                summary="legacy",
                content="legacy",
                relation_type="legacy",
                priority=1.0,
                confidence=1.0,
            ),
        ),
        state_values=(0.5, 0.5, 0.5),
        rule_items=(
            RuleItem(
                id="r",
                type="soft",
                condition="legacy",
                effect="legacy",
                priority=1.0,
                source="test",
            ),
        ),
        expected_answer="entailment",
        surface_group_id="surface",
        source_type="external_benchmark",
        source_id="glue_rte_1",
        external_label="entailment",
    )
    legacy_payload = {
        "text": row.text,
        "label": row.label,
        "task_type": row.task_type,
        "memory_items": [row.memory_items[0].__dict__],
        "state_values": row.state_values,
        "rule_items": [row.rule_items[0].__dict__],
        "expected_answer": row.expected_answer,
        "surface_group_id": row.surface_group_id,
        "source_type": row.source_type,
        "source_id": row.source_id,
        "external_label": row.external_label,
    }
    (source_cache / "glue_rte.jsonl").write_text(
        __import__("json").dumps(legacy_payload) + "\n",
        encoding="utf-8",
    )
    # Minimal files for the other configured tasks keep the helper fully offline and deterministic.
    (source_cache / "super_glue_cb.jsonl").write_text(
        __import__("json").dumps({**legacy_payload, "task_type": "super_glue_cb", "source_id": "super_glue_cb_1"}) + "\n",
        encoding="utf-8",
    )
    (source_cache / "boolq.jsonl").write_text(
        __import__("json").dumps({
            **legacy_payload,
            "text": "A passage states the claim. is the claim supported",
            "label": "causality",
            "task_type": "boolq",
            "source_id": "boolq_1",
            "external_label": "true",
            "expected_answer": "true",
        }) + "\n",
        encoding="utf-8",
    )
    structured_cache = tmp_path / "external_cache_structured"
    manifest = solidify_structured_external_cache(source_cache, structured_cache)
    assert all(row["structured_ok"] for row in manifest)
    structured_text = (structured_cache / "glue_rte.jsonl").read_text(encoding="utf-8")
    assert '"premise": "A premise sentence supports the claim."' in structured_text
    assert '"hypothesis": "the claim is supported"' in structured_text
    assert (structured_cache / "structured_cache_manifest.json").exists()


def test_stage44b6_dataset_fields_cache_manifest_requires_dataset_fields(tmp_path) -> None:
    cache = tmp_path / "dataset_fields_cache"
    cache.mkdir()
    base_payload = {
        "memory_items": [
            {
                "id": "m",
                "summary": "legacy",
                "content": "legacy",
                "relation_type": "legacy",
                "priority": 1.0,
                "confidence": 1.0,
                "embedding": None,
            }
        ],
        "state_values": [0.5, 0.5, 0.5],
        "rule_items": [
            {
                "id": "r",
                "type": "soft",
                "condition": "legacy",
                "effect": "legacy",
                "priority": 1.0,
                "source": "test",
            }
        ],
        "source_type": "external_benchmark",
    }
    rows = {
        "glue_rte": {
            **base_payload,
            "text": "Premise text. Hypothesis text",
            "label": "condition",
            "task_type": "glue_rte",
            "expected_answer": "entailment",
            "surface_group_id": "rte_surface",
            "source_id": "glue_rte_1",
            "external_label": "entailment",
            "premise": "Premise text.",
            "hypothesis": "Hypothesis text",
            "structure_source": "dataset_fields",
        },
        "super_glue_cb": {
            **base_payload,
            "text": "Premise text. Hypothesis text",
            "label": "causality",
            "task_type": "super_glue_cb",
            "expected_answer": "entailment",
            "surface_group_id": "cb_surface",
            "source_id": "super_glue_cb_1",
            "external_label": "entailment",
            "premise": "Premise text.",
            "hypothesis": "Hypothesis text",
            "structure_source": "dataset_fields",
        },
        "boolq": {
            **base_payload,
            "text": "Passage text. is it true",
            "label": "causality",
            "task_type": "boolq",
            "expected_answer": "true",
            "surface_group_id": "boolq_surface",
            "source_id": "boolq_1",
            "external_label": "true",
            "passage": "Passage text.",
            "question": "is it true",
            "structure_source": "dataset_fields",
        },
    }
    for task_name, row in rows.items():
        (cache / f"{task_name}.jsonl").write_text(
            __import__("json").dumps(row) + "\n",
            encoding="utf-8",
        )
    manifest = _dataset_fields_cache_manifest(cache)
    assert all(row["dataset_fields_structured_ok"] for row in manifest)
    assert all(row["structure_sources"] == ["dataset_fields"] for row in manifest)
