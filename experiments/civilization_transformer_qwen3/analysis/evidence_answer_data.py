from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS, LogicSample


LOCAL_ANSWER_OPTIONS = (
    "trace the upstream change",
    "reject the current proposal",
    "escalate the inconsistent report",
    "handle the urgent item first",
    "wait until the prerequisite is satisfied",
)

LABEL_TO_LOCAL_OPTION = dict(zip(LOGIC_LABELS, LOCAL_ANSWER_OPTIONS, strict=True))

EXTERNAL_ANSWER_OPTIONS: dict[str, tuple[str, ...]] = {
    "glue_rte": ("entailment", "not_entailment"),
    "super_glue_cb": ("entailment", "contradiction", "neutral"),
    "boolq": ("true", "false"),
}

EXTERNAL_OPTION_TO_LABEL: dict[str, dict[str, str]] = {
    "glue_rte": {"entailment": "condition", "not_entailment": "negation"},
    "super_glue_cb": {
        "entailment": "causality",
        "contradiction": "conflict",
        "neutral": "condition",
    },
    "boolq": {"true": "causality", "false": "negation"},
}


@dataclass(frozen=True)
class EvidenceAnswerSample:
    sample: LogicSample
    answer_options: tuple[str, ...]
    correct_option_id: int
    evidence_items: tuple[str, ...]
    wrong_context_option_id: int
    context_pair_id: str
    source_type: str

    def __post_init__(self) -> None:
        if len(self.answer_options) < 2 or len(set(self.answer_options)) != len(self.answer_options):
            raise ValueError("answer options must contain at least two unique values")
        if not 0 <= self.correct_option_id < len(self.answer_options):
            raise ValueError("correct_option_id is out of range")
        if not 0 <= self.wrong_context_option_id < len(self.answer_options):
            raise ValueError("wrong_context_option_id is out of range")
        if self.correct_option_id == self.wrong_context_option_id:
            raise ValueError("wrong context option must differ from the correct option")
        if not self.evidence_items:
            raise ValueError("evidence_items cannot be empty")
        if not self.context_pair_id:
            raise ValueError("context_pair_id cannot be empty")


def _options_for_sample(sample: LogicSample) -> tuple[str, ...]:
    if sample.variant in EXTERNAL_ANSWER_OPTIONS:
        return EXTERNAL_ANSWER_OPTIONS[sample.variant]
    return LOCAL_ANSWER_OPTIONS


def build_evidence_answer_samples(
    samples: list[LogicSample],
    source_type: str,
) -> list[EvidenceAnswerSample]:
    result = []
    for sample in samples:
        options = _options_for_sample(sample)
        expected = (
            sample.expected_pattern
            if sample.variant in EXTERNAL_ANSWER_OPTIONS
            else LABEL_TO_LOCAL_OPTION[sample.label]
        )
        if expected not in options:
            raise ValueError(f"expected answer {expected!r} not found for {sample.variant}")
        correct_option_id = options.index(expected)
        wrong_context_option_id = (correct_option_id + 1) % len(options)
        aligned_sample = replace(sample, expected_pattern=expected)
        evidence = tuple(
            value
            for value in (aligned_sample.memory_target, aligned_sample.rule_target)
            if value
        )
        result.append(
            EvidenceAnswerSample(
                sample=aligned_sample,
                answer_options=options,
                correct_option_id=correct_option_id,
                evidence_items=evidence,
                wrong_context_option_id=wrong_context_option_id,
                context_pair_id=aligned_sample.surface_group_id,
                source_type=source_type,
            )
        )
    return result


def wrong_context_sample(record: EvidenceAnswerSample) -> LogicSample:
    wrong_option = record.answer_options[record.wrong_context_option_id]
    if record.sample.variant in EXTERNAL_OPTION_TO_LABEL:
        wrong_label = EXTERNAL_OPTION_TO_LABEL[record.sample.variant][wrong_option]
    else:
        wrong_label = next(
            label
            for label, option in LABEL_TO_LOCAL_OPTION.items()
            if option == wrong_option
        )
    return replace(
        record.sample,
        label=wrong_label,
        memory_target=f"the available evidence supports this answer: {wrong_option}",
        rule_target=f"the controlling decision rule selects this answer: {wrong_option}",
    )


def assert_no_logic_label_leakage(records: list[EvidenceAnswerSample]) -> None:
    forbidden = set(LOGIC_LABELS)
    for record in records:
        text = " ".join((*record.answer_options, *record.evidence_items)).lower()
        words = set(text.replace("_", " ").replace("-", " ").split())
        leaked = forbidden & words
        if leaked:
            raise ValueError(f"logic label leakage detected: {sorted(leaked)}")
