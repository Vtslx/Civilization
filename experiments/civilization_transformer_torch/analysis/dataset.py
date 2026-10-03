from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


LOGIC_LABELS = ("causality", "negation", "conflict", "priority", "condition")
LOGIC_VARIANTS = ("canonical", "synonym", "perturbed", "masked_keywords")
HARD_LOGIC_SCENARIOS = (
    "canonical_hard",
    "synonym_hard",
    "order_shuffled",
    "long_context",
    "distractor_facts",
    "adversarial_keywords",
    "masked_keywords_hard",
    "two_hop_logic",
    "three_hop_logic",
    "counterfactual_pair",
    "mixed_logic_priority",
    "ood_surface",
)
PATH_DEPENDENCY_SCENARIOS = (
    "memory_required_two_hop",
    "state_required_disambiguation",
    "rule_required_priority",
    "memory_conflict_resolution",
    "counterfactual_memory_swap",
    "surface_invariant_label_flip",
)
PATH_STRESS_PROFILES = ("direct_v1", "obfuscated_v1", "noisy_context_v1", "conflicting_context_v1")
OBFUSCATED_CONTEXT_CODES = {
    "causality": "ctx_a13",
    "negation": "ctx_b27",
    "conflict": "ctx_c41",
    "priority": "ctx_d59",
    "condition": "ctx_e83",
}
LEAKAGE_TOKENS = {
    "causality": {"because", "therefore"},
    "negation": {"not", "never"},
    "conflict": {"always", "never"},
    "priority": {"critical", "priority"},
    "condition": {"if", "then"},
}


@dataclass(frozen=True)
class LogicSample:
    label: str
    text: str
    token_ids: tuple[int, ...]
    expected_pattern: str
    variant: str = "canonical"
    template_id: int = 0
    difficulty_level: int = 1
    logic_depth: int = 1
    distractor_count: int = 0
    leakage_family: str = "none"
    pair_id: str = ""
    chain_nodes: tuple[str, ...] = ()
    chain_edges: tuple[tuple[str, str], ...] = ()
    intermediate_targets: tuple[str, ...] = ()
    final_target: str = ""
    primary_label_rule: str = ""
    secondary_labels: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    memory_target: str = ""
    state_target: str = ""
    rule_target: str = ""
    surface_group_id: str = ""
    expected_label_by_context: tuple[tuple[str, str], ...] = ()
    stress_profile: str = "direct_v1"
    context_noise_count: int = 0
    conflict_context_count: int = 0


class LogicTokenizer:
    def __init__(self, max_seq_len: int = 12):
        self.max_seq_len = max_seq_len
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.vocab: dict[str, int] = {self.pad_token: 0, self.unk_token: 1}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> tuple[int, ...]:
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in self.tokenize(text)]
        ids = ids[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids.extend([self.vocab[self.pad_token]] * (self.max_seq_len - len(ids)))
        return tuple(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def _template_rows(label: str, index: int, variant: str, template_bank: str = "basic") -> list[str]:
    subject = f"unit_{index:02d}"
    target = f"result_{index:02d}"
    if variant not in LOGIC_VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if template_bank not in {"basic", "expanded_v1"}:
        raise ValueError(f"unknown template_bank {template_bank}")
    basic = {
        "causality": {
            "canonical": [f"because {subject} increases pressure therefore {target} changes"],
            "synonym": [f"{subject} raises pressure so {target} shifts"],
            "perturbed": [f"in the test chamber {subject} raises pressure and the later reading {target} shifts"],
            "masked_keywords": [f"marker_a {subject} increases pressure marker_b {target} changes"],
        },
        "negation": {
            "canonical": [f"not {subject} valid and never {target} accepted"],
            "synonym": [f"{subject} is denied while {target} is rejected"],
            "perturbed": [f"under review {subject} is denied and the proposed {target} is rejected"],
            "masked_keywords": [f"marker_c {subject} valid and marker_d {target} accepted"],
        },
        "conflict": {
            "canonical": [f"{subject} says always stable but {target} says never stable"],
            "synonym": [f"{subject} claims stable while {target} contradicts stable"],
            "perturbed": [f"report {subject} claims stable yet audit {target} contradicts that stable claim"],
            "masked_keywords": [f"{subject} says marker_e stable but {target} says marker_f stable"],
        },
        "priority": {
            "canonical": [f"critical priority {subject} overrides normal {target}"],
            "synonym": [f"urgent rank {subject} supersedes routine {target}"],
            "perturbed": [f"when queues collide urgent rank {subject} supersedes routine task {target}"],
            "masked_keywords": [f"marker_g marker_h {subject} overrides normal {target}"],
        },
        "condition": {
            "canonical": [f"if {subject} passes threshold then {target} activates"],
            "synonym": [f"when {subject} passes threshold {target} activates"],
            "perturbed": [f"during calibration when {subject} passes threshold the module {target} activates"],
            "masked_keywords": [f"marker_i {subject} passes threshold marker_j {target} activates"],
        },
    }
    expanded = {
        "causality": {
            "canonical": [
                f"because {subject} increases pressure therefore {target} changes",
                f"because {subject} heats the chamber therefore {target} rises",
                f"because {subject} blocks airflow therefore {target} slows",
                f"because {subject} adds load therefore {target} drops",
            ],
            "synonym": [
                f"{subject} raises pressure so {target} shifts",
                f"{subject} heats the chamber so {target} rises",
                f"{subject} blocks airflow so {target} slows",
                f"{subject} adds load so {target} drops",
            ],
            "perturbed": [
                f"in the test chamber {subject} raises pressure and the later reading {target} shifts",
                f"after a noisy setup {subject} heats the chamber and the delayed meter {target} rises",
                f"during audit logs {subject} blocks airflow and the observed signal {target} slows",
                f"while sensors reset {subject} adds load and the final trace {target} drops",
            ],
            "masked_keywords": [
                f"marker_a {subject} increases pressure marker_b {target} changes",
                f"marker_a {subject} heats the chamber marker_b {target} rises",
                f"marker_a {subject} blocks airflow marker_b {target} slows",
                f"marker_a {subject} adds load marker_b {target} drops",
            ],
        },
        "negation": {
            "canonical": [
                f"not {subject} valid and never {target} accepted",
                f"not {subject} enabled and never {target} released",
                f"not {subject} verified and never {target} trusted",
                f"not {subject} approved and never {target} applied",
            ],
            "synonym": [
                f"{subject} is denied while {target} is rejected",
                f"{subject} is disabled while {target} is withheld",
                f"{subject} is unverified while {target} is distrusted",
                f"{subject} is refused while {target} is blocked",
            ],
            "perturbed": [
                f"under review {subject} is denied and the proposed {target} is rejected",
                f"after review notes {subject} is disabled and the scheduled {target} is withheld",
                f"during audit {subject} is unverified and the referenced {target} is distrusted",
                f"inside the control log {subject} is refused and the pending {target} is blocked",
            ],
            "masked_keywords": [
                f"marker_c {subject} valid and marker_d {target} accepted",
                f"marker_c {subject} enabled and marker_d {target} released",
                f"marker_c {subject} verified and marker_d {target} trusted",
                f"marker_c {subject} approved and marker_d {target} applied",
            ],
        },
        "conflict": {
            "canonical": [
                f"{subject} says always stable but {target} says never stable",
                f"{subject} says always open but {target} says never open",
                f"{subject} says always ready but {target} says never ready",
                f"{subject} says always aligned but {target} says never aligned",
            ],
            "synonym": [
                f"{subject} claims stable while {target} contradicts stable",
                f"{subject} claims open while {target} contradicts open",
                f"{subject} claims ready while {target} contradicts ready",
                f"{subject} claims aligned while {target} contradicts aligned",
            ],
            "perturbed": [
                f"report {subject} claims stable yet audit {target} contradicts that stable claim",
                f"record {subject} claims open yet audit {target} contradicts that open claim",
                f"trace {subject} claims ready yet review {target} contradicts that ready claim",
                f"packet {subject} claims aligned yet review {target} contradicts that aligned claim",
            ],
            "masked_keywords": [
                f"{subject} says marker_e stable but {target} says marker_f stable",
                f"{subject} says marker_e open but {target} says marker_f open",
                f"{subject} says marker_e ready but {target} says marker_f ready",
                f"{subject} says marker_e aligned but {target} says marker_f aligned",
            ],
        },
        "priority": {
            "canonical": [
                f"critical priority {subject} overrides normal {target}",
                f"critical priority {subject} outranks routine {target}",
                f"critical priority {subject} preempts standard {target}",
                f"critical priority {subject} dominates default {target}",
            ],
            "synonym": [
                f"urgent rank {subject} supersedes routine {target}",
                f"urgent rank {subject} outranks routine {target}",
                f"urgent rank {subject} preempts standard {target}",
                f"urgent rank {subject} dominates default {target}",
            ],
            "perturbed": [
                f"when queues collide urgent rank {subject} supersedes routine task {target}",
                f"inside queue repair urgent rank {subject} outranks routine task {target}",
                f"during batch merge urgent rank {subject} preempts standard task {target}",
                f"after scheduler review urgent rank {subject} dominates default task {target}",
            ],
            "masked_keywords": [
                f"marker_g marker_h {subject} overrides normal {target}",
                f"marker_g marker_h {subject} outranks routine {target}",
                f"marker_g marker_h {subject} preempts standard {target}",
                f"marker_g marker_h {subject} dominates default {target}",
            ],
        },
        "condition": {
            "canonical": [
                f"if {subject} passes threshold then {target} activates",
                f"if {subject} reaches limit then {target} opens",
                f"if {subject} clears review then {target} starts",
                f"if {subject} matches signal then {target} unlocks",
            ],
            "synonym": [
                f"when {subject} passes threshold {target} activates",
                f"when {subject} reaches limit {target} opens",
                f"when {subject} clears review {target} starts",
                f"when {subject} matches signal {target} unlocks",
            ],
            "perturbed": [
                f"during calibration when {subject} passes threshold the module {target} activates",
                f"after sensor check when {subject} reaches limit the module {target} opens",
                f"inside approval flow when {subject} clears review the module {target} starts",
                f"during signal match when {subject} matches signal the module {target} unlocks",
            ],
            "masked_keywords": [
                f"marker_i {subject} passes threshold marker_j {target} activates",
                f"marker_i {subject} reaches limit marker_j {target} opens",
                f"marker_i {subject} clears review marker_j {target} starts",
                f"marker_i {subject} matches signal marker_j {target} unlocks",
            ],
        },
    }
    source = basic if template_bank == "basic" else expanded
    if label in source:
        return source[label][variant]
    raise ValueError(f"unknown label {label}")


def _sample_text(label: str, index: int, variant: str = "canonical", template_bank: str = "basic") -> tuple[str, str, int]:
    templates = _template_rows(label, index, variant, template_bank)
    template_id = index % len(templates)
    patterns = {
        "causality": "cause leads to effect",
        "negation": "negation blocks assertion",
        "conflict": "mutual contradiction",
        "priority": "higher priority dominates",
        "condition": "condition gates consequence",
    }
    if label in patterns:
        return templates[template_id], patterns[label], template_id
    raise ValueError(f"unknown label {label}")


def build_logic_dataset(samples_per_label: int = 20, max_seq_len: int = 12, seed: int = 42) -> tuple[list[LogicSample], LogicTokenizer]:
    # Seed is part of the public interface for reproducibility. Generation is
    # deterministic today; keep the argument so future variants can shuffle
    # without changing call sites.
    _ = seed
    raw: list[tuple[str, str, str]] = []
    for label in LOGIC_LABELS:
        for index in range(samples_per_label):
            text, pattern, _ = _sample_text(label, index, "canonical")
            raw.append((label, text, pattern))

    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    tokenizer.fit(text for _, text, _ in raw)
    samples = [
        LogicSample(label=label, text=text, token_ids=tokenizer.encode(text), expected_pattern=pattern, variant="canonical")
        for label, text, pattern in raw
    ]
    return samples, tokenizer


def build_logic_variant_datasets(
    samples_per_label: int = 30,
    max_seq_len: int = 14,
    seed: int = 42,
    variants: tuple[str, ...] = LOGIC_VARIANTS,
    template_bank: str = "basic",
) -> tuple[dict[str, list[LogicSample]], LogicTokenizer]:
    _ = seed
    raw_by_variant: dict[str, list[tuple[str, str, str, str, int]]] = {}
    all_texts: list[str] = []
    for variant in variants:
        if variant not in LOGIC_VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        rows: list[tuple[str, str, str, str, int]] = []
        for label in LOGIC_LABELS:
            for index in range(samples_per_label):
                text, pattern, template_id = _sample_text(label, index, variant, template_bank)
                rows.append((label, text, pattern, variant, template_id))
                all_texts.append(text)
        raw_by_variant[variant] = rows

    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    tokenizer.fit(all_texts)
    datasets = {
        variant: [
            LogicSample(label=label, text=text, token_ids=tokenizer.encode(text), expected_pattern=pattern, variant=row_variant, template_id=template_id)
            for label, text, pattern, row_variant, template_id in rows
        ]
        for variant, rows in raw_by_variant.items()
    }
    return datasets, tokenizer


def _hard_terms(index: int) -> dict[str, str]:
    family = index % 8
    return {
        "subject": f"node_{index:04d}",
        "target": f"outcome_{index:04d}",
        "bridge": f"bridge_{index:04d}",
        "middle": f"mid_{index:04d}",
        "final": f"final_{index:04d}",
        "noise_a": f"noise_{family}_a",
        "noise_b": f"noise_{family}_b",
        "family": f"family_{family}",
    }


def _hard_pattern(label: str) -> str:
    patterns = {
        "causality": "cause leads to effect under hard scenario",
        "negation": "negation blocks assertion under hard scenario",
        "conflict": "contradictory claims require conflict detection",
        "priority": "higher priority dominates competing path",
        "condition": "condition gates consequence under hard scenario",
    }
    if label not in patterns:
        raise ValueError(f"unknown label {label}")
    return patterns[label]


def _hard_label_text(label: str, index: int, scenario: str) -> tuple[str, int, int, int, str, str]:
    t = _hard_terms(index)
    s = t["subject"]
    target = t["target"]
    bridge = t["bridge"]
    middle = t["middle"]
    final = t["final"]
    noise_a = t["noise_a"]
    noise_b = t["noise_b"]
    family = t["family"]
    template_id = index % 16
    pair_id = f"{scenario}_{index:04d}" if scenario == "counterfactual_pair" else ""

    base = {
        "causality": f"cause_signal {s} raises load consequence_signal {target} shifts",
        "negation": f"deny_signal {s} is invalid reject_signal {target} is blocked",
        "conflict": f"claim_left {s} reports stable claim_right {target} disputes stable",
        "priority": f"urgent_rank {s} outranks routine_rank {target} in scheduler",
        "condition": f"gate_signal {s} reaches threshold outcome_signal {target} activates",
    }[label]
    synonym = {
        "causality": f"{s} initiates pressure change so downstream {target} moves",
        "negation": f"{s} is refused while downstream {target} is withheld",
        "conflict": f"{s} asserts stable although {target} reverses that assertion",
        "priority": f"{s} has urgent rank and supersedes routine item {target}",
        "condition": f"when {s} satisfies limit downstream {target} begins",
    }[label]
    shuffled = {
        "causality": f"the later {target} shifts after telemetry notes {s} raised load",
        "negation": f"the blocked {target} appears after audit marks {s} invalid",
        "conflict": f"stable is disputed by {target} even though {s} first asserted it",
        "priority": f"routine item {target} yields after scheduler promotes {s}",
        "condition": f"{target} begins only after limit evidence confirms {s}",
    }[label]
    masked = {
        "causality": f"mask_x{index % 7} {s} raises load mask_y{(index + 2) % 7} {target} shifts",
        "negation": f"mask_x{index % 7} {s} invalid mask_y{(index + 3) % 7} {target} blocked",
        "conflict": f"mask_x{index % 7} {s} stable mask_y{(index + 4) % 7} {target} unstable",
        "priority": f"mask_x{index % 7} {s} outranks mask_y{(index + 5) % 7} {target}",
        "condition": f"mask_x{index % 7} {s} reaches threshold mask_y{(index + 6) % 7} {target} activates",
    }[label]

    if scenario == "canonical_hard":
        return f"{base} audit_id {family} template_{template_id}", 2, 1, 0, "canonical", pair_id
    if scenario == "synonym_hard":
        return f"{synonym} review_tag {family} wording_{template_id}", 3, 1, 0, "synonym", pair_id
    if scenario == "order_shuffled":
        return f"context_a {noise_a} {shuffled} context_b {noise_b}", 4, 1, 2, "order", pair_id
    if scenario == "long_context":
        return f"preface {noise_a} filler_0 filler_1 filler_2 {base} spacer_0 spacer_1 spacer_2 appendix {noise_b}", 5, 1, 8, "long", pair_id
    if scenario == "distractor_facts":
        return f"{base} distractor_a {noise_a} suggests open distractor_b {noise_b} suggests closed distractor_c neutral_{index % 5}", 5, 1, 3, "distractor", pair_id
    if scenario == "adversarial_keywords":
        return f"{base} adversarial because not always critical if then never therefore priority keyword_noise_{index % 9}", 6, 1, 8, "adversarial", pair_id
    if scenario == "masked_keywords_hard":
        return f"{masked} neutral_marker_{(index + len(label)) % 11} family_marker_{index % 13}", 6, 1, 0, "masked", pair_id
    if scenario == "two_hop_logic":
        return f"step_one {s} moves {bridge} step_two {bridge} moves {target} label_path {label}", 7, 2, 0, "two_hop", pair_id
    if scenario == "three_hop_logic":
        return f"step_one {s} moves {middle} step_two {middle} moves {bridge} step_three {bridge} moves {final} target {target} label_path {label}", 8, 3, 0, "three_hop", pair_id
    if scenario == "counterfactual_pair":
        return f"shared_entity entity_{index:04d} {base} counter_context label_anchor {label}", 8, 1, 1, "counterfactual", pair_id
    if scenario == "mixed_logic_priority":
        return f"mixed_case {base} secondary_if {s} reaches threshold then {target} opens priority_rule main_label {label} wins", 9, 2, 2, "mixed", pair_id
    if scenario == "ood_surface":
        return f"packet_{template_id} observation {family} core {synonym} closing_token ood_{index % 17}", 7, 1, 1, "ood", pair_id
    raise ValueError(f"unknown hard scenario {scenario}")


def build_hard_logic_datasets(
    samples_per_label: int = 300,
    max_seq_len: int = 64,
    seed: int = 42,
    scenarios: tuple[str, ...] = HARD_LOGIC_SCENARIOS,
    include_chain_supervision: bool = False,
) -> tuple[dict[str, list[LogicSample]], LogicTokenizer]:
    _ = seed
    raw_by_scenario: dict[str, list[tuple[str, str, str, str, int, int, int, int, str, str]]] = {}
    all_texts: list[str] = []
    for scenario in scenarios:
        if scenario not in HARD_LOGIC_SCENARIOS:
            raise ValueError(f"unknown hard scenario {scenario}")
        rows: list[tuple[str, str, str, str, int, int, int, int, str, str]] = []
        for label in LOGIC_LABELS:
            for index in range(samples_per_label):
                text, difficulty, depth, distractors, leakage_family, pair_id = _hard_label_text(label, index, scenario)
                template_id = index % 16
                rows.append((label, text, _hard_pattern(label), scenario, template_id, difficulty, depth, distractors, leakage_family, pair_id))
                all_texts.append(text)
        raw_by_scenario[scenario] = rows

    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    tokenizer.fit(all_texts)
    datasets = {
        scenario: [
            LogicSample(
                label=label,
                text=text,
                token_ids=tokenizer.encode(text),
                expected_pattern=pattern,
                variant=row_scenario,
                template_id=template_id,
                difficulty_level=difficulty,
                logic_depth=depth,
                distractor_count=distractors,
                leakage_family=leakage_family,
                pair_id=pair_id,
                **_chain_supervision_fields(label, template_id, row_scenario, depth, include_chain_supervision),
            )
            for label, text, pattern, row_scenario, template_id, difficulty, depth, distractors, leakage_family, pair_id in rows
        ]
        for scenario, rows in raw_by_scenario.items()
    }
    for scenario, samples in datasets.items():
        for sample in samples:
            token_count = len(tokenizer.tokenize(sample.text))
            if token_count > max_seq_len:
                raise ValueError(f"hard sample in {scenario} exceeds max_seq_len and would lose core logic")
            if max(sample.token_ids) >= tokenizer.vocab_size or min(sample.token_ids) < 0:
                raise ValueError("token ids outside tokenizer vocabulary")
    return datasets, tokenizer


def _dependency_pattern(label: str) -> str:
    return f"path dependent context must select {label}"


def _dependency_state_profile(label: str) -> str:
    profiles = {
        "causality": "low_low_low",
        "negation": "high_low_low",
        "conflict": "low_high_low",
        "priority": "low_low_high",
        "condition": "high_high_low",
    }
    if label not in profiles:
        raise ValueError(f"unknown label {label}")
    return profiles[label]


def _dependency_context_code(label: str, stress_profile: str) -> str:
    if stress_profile == "direct_v1":
        return label
    if stress_profile in {"obfuscated_v1", "noisy_context_v1", "conflicting_context_v1"}:
        return OBFUSCATED_CONTEXT_CODES[label]
    raise ValueError(f"unknown stress_profile {stress_profile}")


def _dependency_context_map(label: str, scenario: str, group_index: int, stress_profile: str) -> tuple[str, str, str, tuple[tuple[str, str], ...], int, int]:
    context_code = _dependency_context_code(label, stress_profile)
    memory_target = f"memory_selects_{context_code}"
    state_target = f"state_selects_{_dependency_state_profile(label)}"
    rule_target = f"rule_selects_{context_code}"
    expected = tuple((candidate, f"context_{_dependency_context_code(candidate, stress_profile)}_{group_index:04d}") for candidate in LOGIC_LABELS)
    noise_count = 3 if stress_profile == "noisy_context_v1" else 0
    conflict_count = 2 if stress_profile == "conflicting_context_v1" else 0
    if scenario == "memory_required_two_hop":
        return memory_target, "", "", expected, noise_count, conflict_count
    if scenario == "state_required_disambiguation":
        return "", state_target, "", expected, noise_count, conflict_count
    if scenario == "rule_required_priority":
        return "", "", rule_target, expected, noise_count, conflict_count
    if scenario == "memory_conflict_resolution":
        return memory_target, _dependency_state_profile(label), rule_target, expected, noise_count, max(conflict_count, 1 if stress_profile == "conflicting_context_v1" else 0)
    if scenario == "counterfactual_memory_swap":
        return memory_target, "", "", expected, noise_count, conflict_count
    if scenario == "surface_invariant_label_flip":
        return memory_target, _dependency_state_profile(label), rule_target, expected, noise_count, conflict_count
    raise ValueError(f"unknown dependency scenario {scenario}")


def _dependency_text(label: str, index: int, scenario: str, stress_profile: str) -> tuple[str, int, tuple[str, ...], str]:
    group_index = index // len(LOGIC_LABELS)
    group_id = f"{stress_profile}_{scenario}_surface_{group_index:04d}"
    profile_token = f"profile_{stress_profile}"
    if scenario == "memory_required_two_hop":
        text = f"shared_probe {group_id} {profile_token} step_one entity_a reaches hidden_bridge step_two hidden_bridge reaches unresolved_outcome"
        return text, 2, ("memory",), group_id
    if scenario == "state_required_disambiguation":
        text = f"shared_probe {group_id} {profile_token} ambiguous_control surface constant_state_choice"
        return text, 1, ("state",), group_id
    if scenario == "rule_required_priority":
        text = f"shared_probe {group_id} {profile_token} mixed_logic candidate_a candidate_b candidate_c main_target_unset"
        return text, 2, ("rule",), group_id
    if scenario == "memory_conflict_resolution":
        text = f"shared_probe {group_id} {profile_token} text_claim conflicts_with_context source_uncertain"
        return text, 2, ("memory", "state", "rule"), group_id
    if scenario == "counterfactual_memory_swap":
        text = f"shared_probe {group_id} {profile_token} same_entities alpha beta gamma outcome_hidden"
        return text, 2, ("memory",), group_id
    if scenario == "surface_invariant_label_flip":
        text = f"shared_probe {group_id} {profile_token} invariant_surface all_labels_share_this_template"
        return text, 3, ("memory", "state", "rule"), group_id
    raise ValueError(f"unknown dependency scenario {scenario}")


def build_path_dependency_datasets(
    samples_per_label: int = 300,
    max_seq_len: int = 64,
    seed: int = 42,
    scenarios: tuple[str, ...] = PATH_DEPENDENCY_SCENARIOS,
    stress_profile: str = "direct_v1",
) -> tuple[dict[str, list[LogicSample]], LogicTokenizer]:
    _ = seed
    if stress_profile not in PATH_STRESS_PROFILES:
        raise ValueError(f"unknown stress_profile {stress_profile}")
    raw_by_scenario: dict[str, list[tuple[str, str, str, str, int, int, tuple[str, ...], str, str, str, str, tuple[tuple[str, str], ...], int, int]]] = {}
    all_texts: list[str] = []
    for scenario in scenarios:
        if scenario not in PATH_DEPENDENCY_SCENARIOS:
            raise ValueError(f"unknown dependency scenario {scenario}")
        rows: list[tuple[str, str, str, str, int, int, tuple[str, ...], str, str, str, str, tuple[tuple[str, str], ...], int, int]] = []
        for label_index, label in enumerate(LOGIC_LABELS):
            for index in range(samples_per_label):
                shared_index = index * len(LOGIC_LABELS) + label_index
                text, depth, required_paths, surface_group_id = _dependency_text(label, shared_index, scenario, stress_profile)
                memory_target, state_target, rule_target, expected, noise_count, conflict_count = _dependency_context_map(label, scenario, index, stress_profile)
                template_id = index % 16
                rows.append(
                    (
                        label,
                        text,
                        _dependency_pattern(label),
                        scenario,
                        template_id,
                        depth,
                        required_paths,
                        memory_target,
                        state_target,
                        rule_target,
                        surface_group_id,
                        expected,
                        noise_count,
                        conflict_count,
                    )
                )
                all_texts.append(text)
        raw_by_scenario[scenario] = rows

    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    tokenizer.fit(all_texts)
    datasets = {
        scenario: [
            LogicSample(
                label=label,
                text=text,
                token_ids=tokenizer.encode(text),
                expected_pattern=pattern,
                variant=row_scenario,
                template_id=template_id,
                difficulty_level=8,
                logic_depth=depth,
                distractor_count=1,
                leakage_family="path_dependency",
                chain_nodes=(f"{surface_group_id}_input", f"{surface_group_id}_context", f"{surface_group_id}_target"),
                chain_edges=((f"{surface_group_id}_input", f"{surface_group_id}_context"), (f"{surface_group_id}_context", f"{surface_group_id}_target")),
                intermediate_targets=(memory_target or state_target or rule_target,),
                final_target=f"{label}_context_final",
                primary_label_rule=rule_target,
                secondary_labels=tuple(candidate for candidate in LOGIC_LABELS if candidate != label)[:2],
                required_paths=required_paths,
                memory_target=memory_target,
                state_target=state_target,
                rule_target=rule_target,
                surface_group_id=surface_group_id,
                expected_label_by_context=expected,
                stress_profile=stress_profile,
                context_noise_count=noise_count,
                conflict_context_count=conflict_count,
            )
            for label, text, pattern, row_scenario, template_id, depth, required_paths, memory_target, state_target, rule_target, surface_group_id, expected, noise_count, conflict_count in rows
        ]
        for scenario, rows in raw_by_scenario.items()
    }
    for scenario, samples in datasets.items():
        for sample in samples:
            token_count = len(tokenizer.tokenize(sample.text))
            if token_count > max_seq_len:
                raise ValueError(f"dependency sample in {scenario} exceeds max_seq_len")
            if max(sample.token_ids) >= tokenizer.vocab_size or min(sample.token_ids) < 0:
                raise ValueError("dependency token ids outside tokenizer vocabulary")
    return datasets, tokenizer


def _chain_supervision_fields(label: str, template_id: int, scenario: str, depth: int, enabled: bool) -> dict:
    if not enabled:
        return {}
    node_prefix = f"{scenario}_{label}_{template_id}"
    if scenario == "two_hop_logic":
        nodes = (f"{node_prefix}_a", f"{node_prefix}_b", f"{node_prefix}_c")
        return {
            "chain_nodes": nodes,
            "chain_edges": ((nodes[0], nodes[1]), (nodes[1], nodes[2])),
            "intermediate_targets": (f"{label}_bridge_state",),
            "final_target": f"{label}_final_state",
        }
    if scenario == "three_hop_logic":
        nodes = (f"{node_prefix}_a", f"{node_prefix}_b", f"{node_prefix}_c", f"{node_prefix}_d")
        return {
            "chain_nodes": nodes,
            "chain_edges": ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])),
            "intermediate_targets": (f"{label}_mid_state", f"{label}_bridge_state"),
            "final_target": f"{label}_final_state",
        }
    if scenario == "mixed_logic_priority":
        secondary = tuple(candidate for candidate in LOGIC_LABELS if candidate != label)[:2]
        return {
            "chain_nodes": (f"{node_prefix}_primary", f"{node_prefix}_secondary"),
            "chain_edges": ((f"{node_prefix}_primary", f"{node_prefix}_secondary"),),
            "intermediate_targets": (f"{label}_priority_context",),
            "final_target": f"{label}_priority_final",
            "primary_label_rule": f"priority_rule_selects_{label}",
            "secondary_labels": secondary,
        }
    if scenario == "ood_surface":
        nodes = (f"{node_prefix}_surface", f"{node_prefix}_semantic")
        return {
            "chain_nodes": nodes,
            "chain_edges": ((nodes[0], nodes[1]),),
            "intermediate_targets": (f"{label}_ood_semantic",),
            "final_target": f"{label}_ood_final",
        }
    if depth > 1:
        nodes = tuple(f"{node_prefix}_{index}" for index in range(depth + 1))
        return {
            "chain_nodes": nodes,
            "chain_edges": tuple((nodes[index], nodes[index + 1]) for index in range(depth)),
            "intermediate_targets": tuple(f"{label}_state_{index}" for index in range(max(0, depth - 1))),
            "final_target": f"{label}_final_state",
        }
    return {
        "chain_nodes": (f"{node_prefix}_single",),
        "chain_edges": (),
        "intermediate_targets": (),
        "final_target": f"{label}_final_state",
    }


def leakage_tokens_for_label(label: str) -> set[str]:
    if label not in LEAKAGE_TOKENS:
        raise ValueError(f"unknown label {label}")
    return LEAKAGE_TOKENS[label]
