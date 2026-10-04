from __future__ import annotations

from dataclasses import dataclass, replace
import random

from civilization.research.torch_line.analysis.dataset import LogicSample, LogicTokenizer


BINARY_BASE_DIAGNOSTIC_MODES = (
    "memory_only_diagnostic",
    "rule_only_diagnostic",
    "memory_rule_conflict_diagnostic",
)
RULE_CONDITIONED_CONFLICT_MODE = "rule_conditioned_conflict_v2"

BINARY_DIAGNOSTIC_MODES = (
    *BINARY_BASE_DIAGNOSTIC_MODES,
    "combined_binary_diagnostic",
)

BINARY_PAIR_TYPES = (
    "memory_binary_pair",
    "rule_binary_pair",
    "memory_rule_conflict_binary_pair",
)
_ALLOWED_PAIR_MODES = (*BINARY_BASE_DIAGNOSTIC_MODES, RULE_CONDITIONED_CONFLICT_MODE)

BINARY_ANSWER_OPTIONS = (
    "select route alpha",
    "select route beta",
)

_OPTION_LABELS = ("causality", "negation")
_FORBIDDEN_WORDS = {"causality", "negation", "conflict", "priority", "condition"}


@dataclass(frozen=True)
class BinaryDiagnosticPair:
    mode: str
    pair_type: str
    pair_id: str
    surface_group_id: str
    answer_options: tuple[str, str]
    full_sample: LogicSample
    counterfactual_sample: LogicSample
    expected_full_option_id: int
    expected_counterfactual_option_id: int
    required_path: str

    def __post_init__(self) -> None:
        if self.mode not in _ALLOWED_PAIR_MODES:
            raise ValueError(f"unsupported diagnostic mode: {self.mode}")
        if self.pair_type not in BINARY_PAIR_TYPES:
            raise ValueError(f"unsupported binary pair type: {self.pair_type}")
        if len(self.answer_options) != 2 or len(set(self.answer_options)) != 2:
            raise ValueError("binary diagnostic pair must have exactly two unique options")
        if self.full_sample.text != self.counterfactual_sample.text:
            raise ValueError("binary diagnostic pair surface text must match")
        if self.expected_full_option_id == self.expected_counterfactual_option_id:
            raise ValueError("binary diagnostic pair must flip its correct option")
        if not self.pair_id or not self.surface_group_id:
            raise ValueError("pair_id and surface_group_id cannot be empty")


def _option_phrase(option_id: int) -> str:
    return BINARY_ANSWER_OPTIONS[option_id].replace("select ", "")


def _sample(
    *,
    text: str,
    mode: str,
    index: int,
    option_id: int,
    memory_target: str,
    rule_target: str,
    required_paths: tuple[str, ...],
    tokenizer: LogicTokenizer,
) -> LogicSample:
    return LogicSample(
        label=_OPTION_LABELS[option_id],
        text=text,
        token_ids=tokenizer.encode(text),
        expected_pattern=BINARY_ANSWER_OPTIONS[option_id],
        variant=mode,
        template_id=index,
        difficulty_level=10,
        logic_depth=1,
        distractor_count=0,
        leakage_family="binary_path_diagnostic_grounded_v1",
        required_paths=required_paths,
        memory_target=memory_target,
        state_target="state profile adjusts trust in private context and override policy",
        rule_target=rule_target,
        surface_group_id=f"{mode}_surface_{index:04d}",
        expected_label_by_context=(
            ("alpha_context", BINARY_ANSWER_OPTIONS[0]),
            ("beta_context", BINARY_ANSWER_OPTIONS[1]),
        ),
        stress_profile="binary_path_diagnostic_v1",
        context_noise_count=0,
        conflict_context_count=1 if mode == "memory_rule_conflict_diagnostic" else 0,
    )


def _surface_text(mode: str, index: int) -> str:
    return (
        f"binary diagnostic case {mode} {index:04d}. The visible report is intentionally ambiguous. "
        "Choose one route only from private context, not from the surface text."
    )


def _memory_pair(index: int, tokenizer: LogicTokenizer) -> BinaryDiagnosticPair:
    text = _surface_text("memory_only_diagnostic", index)
    rule_target = "unchanged instruction: follow the private memory note for this ambiguous report"
    full = _sample(
        text=text,
        mode="memory_only_diagnostic",
        index=index,
        option_id=0,
        memory_target=f"private memory evidence points to {_option_phrase(0)}",
        rule_target=rule_target,
        required_paths=("memory", "rule"),
        tokenizer=tokenizer,
    )
    counterfactual = replace(
        full,
        label=_OPTION_LABELS[1],
        expected_pattern=BINARY_ANSWER_OPTIONS[1],
        memory_target=f"private memory evidence points to {_option_phrase(1)}",
    )
    return BinaryDiagnosticPair(
        mode="memory_only_diagnostic",
        pair_type="memory_binary_pair",
        pair_id=f"memory_binary_pair_{index:04d}",
        surface_group_id=full.surface_group_id,
        answer_options=BINARY_ANSWER_OPTIONS,
        full_sample=full,
        counterfactual_sample=counterfactual,
        expected_full_option_id=0,
        expected_counterfactual_option_id=1,
        required_path="memory",
    )


def _rule_pair(index: int, tokenizer: LogicTokenizer) -> BinaryDiagnosticPair:
    text = _surface_text("rule_only_diagnostic", index)
    memory_target = "background memory is neutral and cannot decide between the two routes"
    full = _sample(
        text=text,
        mode="rule_only_diagnostic",
        index=index,
        option_id=0,
        memory_target=memory_target,
        rule_target=f"binding rule points to {_option_phrase(0)}",
        required_paths=("memory", "rule"),
        tokenizer=tokenizer,
    )
    counterfactual = replace(
        full,
        label=_OPTION_LABELS[1],
        expected_pattern=BINARY_ANSWER_OPTIONS[1],
        rule_target=f"binding rule points to {_option_phrase(1)}",
    )
    return BinaryDiagnosticPair(
        mode="rule_only_diagnostic",
        pair_type="rule_binary_pair",
        pair_id=f"rule_binary_pair_{index:04d}",
        surface_group_id=full.surface_group_id,
        answer_options=BINARY_ANSWER_OPTIONS,
        full_sample=full,
        counterfactual_sample=counterfactual,
        expected_full_option_id=0,
        expected_counterfactual_option_id=1,
        required_path="rule",
    )


def _conflict_pair(index: int, tokenizer: LogicTokenizer) -> BinaryDiagnosticPair:
    text = _surface_text("memory_rule_conflict_diagnostic", index)
    full = _sample(
        text=text,
        mode="memory_rule_conflict_diagnostic",
        index=index,
        option_id=1,
        memory_target=f"private memory evidence points to {_option_phrase(0)}",
        rule_target=f"override policy rejects memory and points to {_option_phrase(1)}",
        required_paths=("memory", "rule", "state"),
        tokenizer=tokenizer,
    )
    counterfactual = replace(
        full,
        label=_OPTION_LABELS[0],
        expected_pattern=BINARY_ANSWER_OPTIONS[0],
        memory_target=f"private memory evidence points to {_option_phrase(1)}",
        rule_target=f"override policy rejects memory and points to {_option_phrase(0)}",
    )
    return BinaryDiagnosticPair(
        mode="memory_rule_conflict_diagnostic",
        pair_type="memory_rule_conflict_binary_pair",
        pair_id=f"memory_rule_conflict_binary_pair_{index:04d}",
        surface_group_id=full.surface_group_id,
        answer_options=BINARY_ANSWER_OPTIONS,
        full_sample=full,
        counterfactual_sample=counterfactual,
        expected_full_option_id=1,
        expected_counterfactual_option_id=0,
        required_path="memory_rule_conflict",
    )


def _rule_conditioned_conflict_pair(index: int, tokenizer: LogicTokenizer) -> BinaryDiagnosticPair:
    text = _surface_text(RULE_CONDITIONED_CONFLICT_MODE, index)
    full = _sample(
        text=text,
        mode=RULE_CONDITIONED_CONFLICT_MODE,
        index=index,
        option_id=1,
        memory_target="private memory evidence is stale and must not decide the active route",
        rule_target=f"binding override rule points to {_option_phrase(1)}",
        required_paths=("memory", "rule", "state"),
        tokenizer=tokenizer,
    )
    full = replace(
        full,
        state_target="state enables rule arbitration without naming either route",
        leakage_family="rule_conditioned_conflict_v2_grounded_v1",
        stress_profile="rule_conditioned_conflict_v2",
        context_noise_count=0,
        conflict_context_count=0,
    )
    counterfactual = replace(
        full,
        label=_OPTION_LABELS[0],
        expected_pattern=BINARY_ANSWER_OPTIONS[0],
        rule_target=f"binding override rule points to {_option_phrase(0)}",
    )
    return BinaryDiagnosticPair(
        mode=RULE_CONDITIONED_CONFLICT_MODE,
        pair_type="memory_rule_conflict_binary_pair",
        pair_id=f"rule_conditioned_conflict_pair_{index:04d}",
        surface_group_id=full.surface_group_id,
        answer_options=BINARY_ANSWER_OPTIONS,
        full_sample=full,
        counterfactual_sample=counterfactual,
        expected_full_option_id=1,
        expected_counterfactual_option_id=0,
        required_path="rule_conditioned_conflict",
    )


def build_binary_path_diagnostic_pairs(
    pairs_per_mode: int = 120,
    seed: int = 202,
    max_seq_len: int = 64,
) -> dict[str, list[BinaryDiagnosticPair]]:
    if pairs_per_mode <= 0:
        raise ValueError("pairs_per_mode must be positive")
    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    builders = {
        "memory_only_diagnostic": _memory_pair,
        "rule_only_diagnostic": _rule_pair,
        "memory_rule_conflict_diagnostic": _conflict_pair,
    }
    result = {
        mode: [builder(index, tokenizer) for index in range(pairs_per_mode)]
        for mode, builder in builders.items()
    }
    rng = random.Random(seed)
    for pairs in result.values():
        rng.shuffle(pairs)
        pairs.sort(key=lambda pair: pair.pair_id)
    return result


def build_rule_conditioned_conflict_pairs(
    pairs_per_mode: int = 120,
    seed: int = 202,
    max_seq_len: int = 64,
) -> list[BinaryDiagnosticPair]:
    if pairs_per_mode <= 0:
        raise ValueError("pairs_per_mode must be positive")
    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    pairs = [_rule_conditioned_conflict_pair(index, tokenizer) for index in range(pairs_per_mode)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return sorted(pairs, key=lambda pair: pair.pair_id)


def assert_rule_conditioned_conflict_pairs_valid(pairs: list[BinaryDiagnosticPair]) -> None:
    for pair in pairs:
        if pair.mode != RULE_CONDITIONED_CONFLICT_MODE:
            raise ValueError("rule-conditioned conflict validator received another mode")
        if pair.full_sample.text != pair.counterfactual_sample.text:
            raise ValueError("rule-conditioned conflict pair surface text mismatch")
        if pair.full_sample.memory_target != pair.counterfactual_sample.memory_target:
            raise ValueError("rule-conditioned conflict pair changed memory context")
        if pair.full_sample.rule_target == pair.counterfactual_sample.rule_target:
            raise ValueError("rule-conditioned conflict pair did not flip rule context")
        if pair.full_sample.state_target != pair.counterfactual_sample.state_target:
            raise ValueError("rule-conditioned conflict pair changed state target")
        for option in BINARY_ANSWER_OPTIONS:
            if _option_phrase(BINARY_ANSWER_OPTIONS.index(option)) in pair.full_sample.memory_target:
                raise ValueError("rule-conditioned conflict memory leaked a route option")


def build_combined_binary_path_diagnostic_pairs(
    grouped_pairs: dict[str, list[BinaryDiagnosticPair]],
    pairs_per_mode: int,
) -> list[BinaryDiagnosticPair]:
    """Return a deterministic mixed diagnostic view without mutating pair metadata."""
    missing = [mode for mode in BINARY_BASE_DIAGNOSTIC_MODES if mode not in grouped_pairs]
    if missing:
        raise ValueError(f"missing base diagnostic modes for combined view: {missing}")
    if pairs_per_mode <= 0:
        raise ValueError("pairs_per_mode must be positive")
    per_base_mode = max(1, pairs_per_mode // len(BINARY_BASE_DIAGNOSTIC_MODES))
    combined: list[BinaryDiagnosticPair] = []
    for mode in BINARY_BASE_DIAGNOSTIC_MODES:
        combined.extend(sorted(grouped_pairs[mode], key=lambda pair: pair.pair_id)[:per_base_mode])
    remaining = pairs_per_mode - len(combined)
    if remaining > 0:
        tail_pool = [
            pair
            for mode in BINARY_BASE_DIAGNOSTIC_MODES
            for pair in sorted(grouped_pairs[mode], key=lambda item: item.pair_id)[per_base_mode:]
        ]
        combined.extend(tail_pool[:remaining])
    return sorted(combined, key=lambda pair: (pair.mode, pair.pair_id))


def split_binary_pairs(
    pairs: list[BinaryDiagnosticPair],
    train_pairs: int,
    held_out_pairs: int,
) -> tuple[list[BinaryDiagnosticPair], list[BinaryDiagnosticPair]]:
    if train_pairs <= 0 or held_out_pairs <= 0:
        raise ValueError("train_pairs and held_out_pairs must be positive")
    if train_pairs + held_out_pairs > len(pairs):
        raise ValueError("not enough binary pairs for requested split")
    ordered = sorted(pairs, key=lambda pair: pair.pair_id)
    train = ordered[:train_pairs]
    test = ordered[train_pairs : train_pairs + held_out_pairs]
    if {pair.pair_id for pair in train} & {pair.pair_id for pair in test}:
        raise ValueError("binary pair split leaked pair_id")
    if {pair.surface_group_id for pair in train} & {pair.surface_group_id for pair in test}:
        raise ValueError("binary pair split leaked surface_group_id")
    return train, test


def assert_binary_pairs_valid(pairs: list[BinaryDiagnosticPair]) -> None:
    for pair in pairs:
        if pair.full_sample.text != pair.counterfactual_sample.text:
            raise ValueError("binary pair text mismatch")
        if pair.expected_full_option_id == pair.expected_counterfactual_option_id:
            raise ValueError("binary pair did not flip")
        if pair.pair_type == "memory_binary_pair":
            if pair.full_sample.rule_target != pair.counterfactual_sample.rule_target:
                raise ValueError("memory binary pair changed rule context")
            if pair.full_sample.memory_target == pair.counterfactual_sample.memory_target:
                raise ValueError("memory binary pair did not change memory context")
        elif pair.pair_type == "rule_binary_pair":
            if pair.full_sample.memory_target != pair.counterfactual_sample.memory_target:
                raise ValueError("rule binary pair changed memory context")
            if pair.full_sample.rule_target == pair.counterfactual_sample.rule_target:
                raise ValueError("rule binary pair did not change rule context")
        elif pair.pair_type == "memory_rule_conflict_binary_pair":
            if "override policy" not in pair.full_sample.rule_target and "override rule" not in pair.full_sample.rule_target:
                raise ValueError("conflict binary pair lacks controlling rule")
        text = " ".join(
            (
                *pair.answer_options,
                pair.full_sample.memory_target,
                pair.full_sample.rule_target,
                pair.counterfactual_sample.memory_target,
                pair.counterfactual_sample.rule_target,
            )
        ).lower()
        words = set(text.replace("_", " ").replace("-", " ").split())
        leaked = _FORBIDDEN_WORDS & words
        if leaked:
            raise ValueError(f"logic label leakage detected: {sorted(leaked)}")
