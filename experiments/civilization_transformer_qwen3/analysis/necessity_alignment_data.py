from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.civilization_transformer_torch.analysis.dataset import LogicSample

from .evidence_answer_data import (
    EvidenceAnswerSample,
    EXTERNAL_OPTION_TO_LABEL,
    LABEL_TO_LOCAL_OPTION,
)


PAIR_TYPES = (
    "memory_necessity_pair",
    "rule_necessity_pair",
    "memory_rule_conflict_pair",
)


@dataclass(frozen=True)
class NecessityPair:
    base_record: EvidenceAnswerSample
    full_sample: LogicSample
    counterfactual_sample: LogicSample
    pair_type: str
    pair_id: str
    required_path: str
    counterfactual_context: str
    expected_full_option_id: int
    expected_counterfactual_option_id: int

    def __post_init__(self) -> None:
        if self.pair_type not in PAIR_TYPES:
            raise ValueError(f"unsupported pair type: {self.pair_type}")
        if self.full_sample.text != self.counterfactual_sample.text:
            raise ValueError("necessity pair surface text must match")
        if self.expected_full_option_id == self.expected_counterfactual_option_id:
            raise ValueError("necessity pair must flip the expected option")
        if not self.pair_id:
            raise ValueError("pair_id cannot be empty")


def option_to_label(record: EvidenceAnswerSample, option_id: int) -> str:
    option = record.answer_options[option_id]
    if record.sample.variant in EXTERNAL_OPTION_TO_LABEL:
        return EXTERNAL_OPTION_TO_LABEL[record.sample.variant][option]
    return next(
        label
        for label, answer in LABEL_TO_LOCAL_OPTION.items()
        if answer == option
    )


def _option_phrase(record: EvidenceAnswerSample, option_id: int) -> str:
    return record.answer_options[option_id].replace("_", " ")


def _base_sample(
    record: EvidenceAnswerSample,
    option_id: int,
    *,
    memory_target: str,
    rule_target: str,
) -> LogicSample:
    return replace(
        record.sample,
        label=option_to_label(record, option_id),
        expected_pattern=record.answer_options[option_id],
        memory_target=memory_target,
        rule_target=rule_target,
        required_paths=("memory", "rule", "state"),
    )


def _memory_pair(record: EvidenceAnswerSample) -> NecessityPair:
    full_option = record.correct_option_id
    counterfactual_option = record.wrong_context_option_id
    full_phrase = _option_phrase(record, full_option)
    counterfactual_phrase = _option_phrase(record, counterfactual_option)
    rule_target = "control rule: use the memory evidence for this unchanged surface"
    full_sample = _base_sample(
        record,
        full_option,
        memory_target=f"memory evidence selects answer option: {full_phrase}",
        rule_target=rule_target,
    )
    counterfactual = _base_sample(
        record,
        counterfactual_option,
        memory_target=f"memory evidence selects answer option: {counterfactual_phrase}",
        rule_target=rule_target,
    )
    return NecessityPair(
        base_record=record,
        full_sample=full_sample,
        counterfactual_sample=counterfactual,
        pair_type="memory_necessity_pair",
        pair_id=f"{record.context_pair_id}:memory",
        required_path="memory",
        counterfactual_context=counterfactual.memory_target,
        expected_full_option_id=full_option,
        expected_counterfactual_option_id=counterfactual_option,
    )


def _rule_pair(record: EvidenceAnswerSample) -> NecessityPair:
    full_option = record.correct_option_id
    counterfactual_option = record.wrong_context_option_id
    full_phrase = _option_phrase(record, full_option)
    counterfactual_phrase = _option_phrase(record, counterfactual_option)
    memory_target = "background memory: evidence is neutral and does not decide the answer"
    full_sample = _base_sample(
        record,
        full_option,
        memory_target=memory_target,
        rule_target=f"control rule selects answer option: {full_phrase}",
    )
    counterfactual = _base_sample(
        record,
        counterfactual_option,
        memory_target=memory_target,
        rule_target=f"control rule selects answer option: {counterfactual_phrase}",
    )
    return NecessityPair(
        base_record=record,
        full_sample=full_sample,
        counterfactual_sample=counterfactual,
        pair_type="rule_necessity_pair",
        pair_id=f"{record.context_pair_id}:rule",
        required_path="rule",
        counterfactual_context=counterfactual.rule_target,
        expected_full_option_id=full_option,
        expected_counterfactual_option_id=counterfactual_option,
    )


def _conflict_pair(record: EvidenceAnswerSample) -> NecessityPair:
    full_option = record.correct_option_id
    counterfactual_option = record.wrong_context_option_id
    full_phrase = _option_phrase(record, full_option)
    counterfactual_phrase = _option_phrase(record, counterfactual_option)
    full_sample = _base_sample(
        record,
        full_option,
        memory_target=f"memory evidence selects answer option: {counterfactual_phrase}",
        rule_target=f"higher rank rule overrides memory and selects answer option: {full_phrase}",
    )
    counterfactual = _base_sample(
        record,
        counterfactual_option,
        memory_target=f"memory evidence selects answer option: {full_phrase}",
        rule_target=f"higher rank rule overrides memory and selects answer option: {counterfactual_phrase}",
    )
    return NecessityPair(
        base_record=record,
        full_sample=full_sample,
        counterfactual_sample=counterfactual,
        pair_type="memory_rule_conflict_pair",
        pair_id=f"{record.context_pair_id}:conflict",
        required_path="memory_rule_conflict",
        counterfactual_context=counterfactual.rule_target,
        expected_full_option_id=full_option,
        expected_counterfactual_option_id=counterfactual_option,
    )


def build_necessity_pairs(records: list[EvidenceAnswerSample]) -> list[NecessityPair]:
    pairs: list[NecessityPair] = []
    for record in records:
        pairs.extend((_memory_pair(record), _rule_pair(record), _conflict_pair(record)))
    return pairs


def assert_necessity_pairs_valid(pairs: list[NecessityPair]) -> None:
    for pair in pairs:
        if pair.pair_type == "memory_necessity_pair":
            if pair.full_sample.rule_target != pair.counterfactual_sample.rule_target:
                raise ValueError("memory pair changed rule context")
            if pair.full_sample.memory_target == pair.counterfactual_sample.memory_target:
                raise ValueError("memory pair did not change memory context")
        elif pair.pair_type == "rule_necessity_pair":
            if pair.full_sample.memory_target != pair.counterfactual_sample.memory_target:
                raise ValueError("rule pair changed memory context")
            if pair.full_sample.rule_target == pair.counterfactual_sample.rule_target:
                raise ValueError("rule pair did not change rule context")
        elif pair.pair_type == "memory_rule_conflict_pair":
            if "overrides memory" not in pair.full_sample.rule_target:
                raise ValueError("conflict pair lacks controlling rule")
            if pair.full_sample.memory_target == pair.full_sample.rule_target:
                raise ValueError("conflict pair lacks memory/rule conflict")
