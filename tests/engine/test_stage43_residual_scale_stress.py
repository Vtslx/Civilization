from __future__ import annotations

from dataclasses import replace

from civilization.engine.stages.evidence_answer_data import EvidenceAnswerSample
from civilization.engine.stages.group_full_hidden_centroid_integration import (
    RAW_FULL_HIDDEN_RESIDUAL_SCALE,
)
from civilization.engine.stages.necessity_alignment_data import NecessityPair
from civilization.engine.stages.run_qwen3_stage43_residual_scale_stress import (
    _neutral_padding,
    _pad_groups,
    _stage43_gate_failures,
    _truncation_audit_rows,
)
from civilization.engine.stages.multiclass_group_curriculum_repair import (
    SurfaceGroupCandidateBatch,
)
from civilization.research.torch_line.analysis.dataset import LogicSample


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False):
        return {"input_ids": [[0] + text.split() + [1] for text in texts]}


def _sample(option_id: int, text: str = "shared surface text") -> LogicSample:
    return LogicSample(
        label=("causality", "negation", "conflict", "priority", "condition")[option_id],
        text=text,
        token_ids=(),
        expected_pattern=f"option_{option_id}",
        variant="local",
        surface_group_id="surface_0001",
        required_paths=("memory", "rule", "state"),
        memory_target=f"memory option {option_id}",
        rule_target=f"rule option {option_id}",
    )


def _record() -> EvidenceAnswerSample:
    return EvidenceAnswerSample(
        sample=_sample(0),
        answer_options=("option_0", "option_1", "option_2", "option_3", "option_4"),
        correct_option_id=0,
        evidence_items=("unit evidence",),
        wrong_context_option_id=1,
        context_pair_id="pair_0001",
        source_type="local_semireal",
    )


def _group() -> SurfaceGroupCandidateBatch:
    record = _record()
    pairs = []
    for option_id in range(5):
        pairs.append(
            NecessityPair(
                base_record=record,
                full_sample=_sample(option_id),
                counterfactual_sample=replace(_sample((option_id + 1) % 5), text="shared surface text"),
                pair_type="memory_necessity_pair",
                pair_id=f"pair_{option_id}",
                required_path="memory",
                counterfactual_context="counterfactual",
                expected_full_option_id=option_id,
                expected_counterfactual_option_id=(option_id + 1) % 5,
            )
        )
    return SurfaceGroupCandidateBatch("memory_necessity_group", "surface_0001", tuple(pairs))


def test_residual_scale_is_fixed_to_200() -> None:
    assert RAW_FULL_HIDDEN_RESIDUAL_SCALE == 200.0


def test_long_context_padding_preserves_five_candidate_group() -> None:
    padded = _pad_groups([_group()], max_length=128)[0]
    assert len(padded.pairs) == 5
    assert [pair.expected_full_option_id for pair in padded.pairs] == [0, 1, 2, 3, 4]
    assert len({pair.full_sample.text for pair in padded.pairs}) == 1
    assert all("neutral_context_padding_v1" in pair.full_sample.text for pair in padded.pairs)
    assert _neutral_padding("surface_0001", 64) == ""


def test_truncation_audit_detects_core_logic_loss() -> None:
    rows = _truncation_audit_rows(FakeTokenizer(), [_group()], max_length=3, split="train")
    assert rows
    assert any(row["core_logic_may_be_lost"] for row in rows)
    rows_without_truncation = _truncation_audit_rows(FakeTokenizer(), [_group()], max_length=64, split="train")
    assert not any(row["core_logic_may_be_lost"] for row in rows_without_truncation)


def test_stage43_gate_failures_reject_hidden_norm_or_length_drop() -> None:
    rows = [
        {
            "seed": 202,
            "max_length": 64,
            "passes_stage_gate": True,
            "fixed_centroid_after": 1.0,
            "hidden_norm_ratio": 1.1,
            "qwen_trainable_parameters": 0.0,
            "qwen_gradients": 0.0,
            "weights_unchanged": True,
            "raw_full_hidden_residual_scale": 200.0,
        },
        {
            "seed": 202,
            "max_length": 128,
            "passes_stage_gate": True,
            "fixed_centroid_after": 0.7,
            "hidden_norm_ratio": 2.1,
            "qwen_trainable_parameters": 0.0,
            "qwen_gradients": 0.0,
            "weights_unchanged": True,
            "raw_full_hidden_residual_scale": 200.0,
        },
    ]
    failures = _stage43_gate_failures(rows, [])
    gates = {failure["failed_gate"] for failure in failures}
    assert "hidden_norm_ratio" in gates
    assert "len128_fixed_centroid_drop" in gates
