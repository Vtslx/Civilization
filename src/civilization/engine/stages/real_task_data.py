from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from civilization.research.torch_line.analysis.dataset import (
    LOGIC_LABELS,
    LogicSample,
    LogicTokenizer,
)
from civilization.research.torch_line.analysis.path_dependency_training import (
    dependency_state_for_label,
)
from civilization.research.torch_line.memory import MemoryItem
from civilization.research.torch_line.rules import RuleItem


LOCAL_REAL_TASK_TYPES = (
    "operation_decision",
    "rule_conflict",
    "causal_trace",
    "condition_check",
    "priority_selection",
    "negation_constraint",
)

EXTERNAL_DATASET_SPECS = {
    "glue_rte": {
        "dataset_id": "nyu-mll/glue",
        "config": "rte",
        "split": "validation",
        "revision": "main",
        "text_fields": ("sentence1", "sentence2"),
        "label_field": "label",
        "label_map": {
            0: ("condition", "entailment"),
            1: ("negation", "not_entailment"),
        },
    },
    "super_glue_cb": {
        "dataset_id": "aps/super_glue",
        "config": "cb",
        "split": "validation",
        "revision": "main",
        "text_fields": ("premise", "hypothesis"),
        "label_field": "label",
        "label_map": {
            0: ("causality", "entailment"),
            1: ("conflict", "contradiction"),
            2: ("condition", "neutral"),
        },
    },
    "boolq": {
        "dataset_id": "google/boolq",
        "config": "default",
        "split": "validation",
        "revision": "main",
        "text_fields": ("passage", "question"),
        "label_field": "answer",
        "label_map": {
            True: ("causality", "true"),
            False: ("negation", "false"),
        },
    },
}


@dataclass(frozen=True)
class RealTaskRecord:
    text: str
    label: str
    task_type: str
    memory_items: tuple[MemoryItem, ...]
    state_values: tuple[float, float, float]
    rule_items: tuple[RuleItem, ...]
    expected_answer: str
    surface_group_id: str
    source_type: str
    source_id: str
    external_label: str = ""
    required_path: str = ""
    context_owner: str = ""
    counterfactual_context: str = ""
    context_hash: str = ""
    premise: str = ""
    hypothesis: str = ""
    passage: str = ""
    question: str = ""
    structure_source: str = ""

    def to_logic_sample(
        self,
        tokenizer: LogicTokenizer,
        template_id: int = 0,
        context_grounding_mode: str = "real_task_v1",
    ) -> LogicSample:
        if context_grounding_mode not in {"real_task_v1", "grounded_v1", "stage44a_path_grounded_v2"}:
            raise ValueError(f"unsupported context_grounding_mode: {context_grounding_mode}")
        if context_grounding_mode == "stage44a_path_grounded_v2":
            required_paths = tuple(path for path in self.required_path.split("+") if path)
        else:
            paths = []
            if self.memory_items:
                paths.append("memory")
            if self.rule_items:
                paths.append("rule")
            if self.state_values != (0.5, 0.5, 0.5):
                paths.append("state")
            required_paths = tuple(paths)
        if context_grounding_mode in {"grounded_v1", "stage44a_path_grounded_v2"}:
            memory_target = " | ".join(
                f"{item.summary}; {item.content}; relation={item.relation_type}"
                for item in self.memory_items
            )
            rule_target = " | ".join(
                f"control rule: when {item.condition}, apply {item.effect} from {item.source}"
                for item in self.rule_items
            )
            state_target = (
                "state profile: rigor controls evidence checking; creativity controls alternative search; "
                "defensiveness controls conflict rejection"
                if "state" in required_paths
                else ""
            )
            leakage_family = (
                "rule_conditioned_conflict_v2_stage44a_path_grounded_v2"
                if context_grounding_mode == "stage44a_path_grounded_v2" and self.task_type == "rule_conflict"
                else f"{self.source_type}_{context_grounding_mode}"
            )
        else:
            memory_target = f"memory_selects_{self.label}" if self.memory_items else ""
            state_target = f"state_selects_{self.label}" if "state" in required_paths else ""
            rule_target = f"rule_selects_{self.label}" if self.rule_items else ""
            leakage_family = self.source_type
        return LogicSample(
            label=self.label,
            text=self.text,
            token_ids=tokenizer.encode(self.text),
            expected_pattern=self.expected_answer,
            variant=self.task_type,
            template_id=template_id,
            difficulty_level=9,
            logic_depth=2,
            distractor_count=1,
            leakage_family=leakage_family,
            required_paths=required_paths,
            memory_target=memory_target,
            state_target=state_target,
            rule_target=rule_target,
            surface_group_id=self.surface_group_id,
            expected_label_by_context=tuple((label, f"context_{label}") for label in LOGIC_LABELS),
            stress_profile="real_task_v1",
            context_noise_count=max(0, len(self.memory_items) - 1),
            conflict_context_count=sum(1 for item in self.rule_items if item.type == "conflict"),
        )


def _state_tuple(label: str) -> tuple[float, float, float]:
    state = dependency_state_for_label(label)
    return (state.rigor, state.creativity, state.defensiveness)


def _record_context(
    label: str,
    task_type: str,
    group_id: str,
    context_grounding_mode: str = "real_task_v1",
) -> tuple[tuple[MemoryItem, ...], tuple[RuleItem, ...]]:
    if context_grounding_mode == "grounded_v1":
        evidence_bank = {
            "causality": (
                "upstream change is observed before downstream effect",
                "temporal evidence supports an ordered event to result link",
                "prefer cause-effect explanation when ordered evidence is present",
            ),
            "negation": (
                "explicit denial blocks the proposed action",
                "the available evidence says the prerequisite is not satisfied",
                "reject the candidate when a denial constraint is active",
            ),
            "conflict": (
                "two sources make incompatible claims about the same case",
                "evidence cannot both be accepted without contradiction",
                "flag inconsistency when mutually exclusive facts appear",
            ),
            "priority": (
                "one item has higher operational urgency than alternatives",
                "the controlling policy ranks the urgent item first",
                "choose the higher-ranked item when multiple actions compete",
            ),
            "condition": (
                "the action is valid only if a prerequisite is satisfied",
                "the evidence describes a gated if-then requirement",
                "apply the result only when the prerequisite is met",
            ),
        }
        summary, content, effect = evidence_bank[label]
        memories = (
            MemoryItem(
                id=f"memory_{group_id}_primary",
                summary="grounded evidence",
                content=content,
                relation_type=summary,
                priority=1.0,
                confidence=1.0,
            ),
            MemoryItem(
                id=f"memory_{group_id}_distractor",
                summary="background note",
                content="a neighboring case has routine status and does not decide this record",
                relation_type="background",
                priority=0.2,
                confidence=0.3,
            ),
        )
        rules = (
            RuleItem(
                id=f"rule_{group_id}_primary",
                type="soft" if task_type != "rule_conflict" else "conflict",
                condition="grounded evidence review",
                effect=effect,
                priority=1.0,
                source="real_task_grounded_v1",
            ),
        )
        return memories, rules
    memories = (
        MemoryItem(
            id=f"memory_{group_id}_{label}",
            summary=f"{task_type} memory selects {label}",
            content=f"Operational context points to {label} as the correct route",
            relation_type=f"memory_selects_{label}",
            priority=1.0,
            confidence=1.0,
        ),
        MemoryItem(
            id=f"memory_noise_{group_id}",
            summary="irrelevant operational note",
            content="This note is unrelated to the final decision",
            relation_type="noise",
            priority=0.2,
            confidence=0.3,
        ),
    )
    rules = (
        RuleItem(
            id=f"rule_{group_id}_{label}",
            type="soft" if task_type != "rule_conflict" else "conflict",
            condition=f"{task_type}_context",
            effect=f"rule_selects_{label}",
            priority=1.0,
            source="real_task_local",
        ),
    )
    return memories, rules


def build_local_semireal_task_records(
    samples_per_label: int = 40,
    seed: int = 42,
    context_grounding_mode: str = "real_task_v1",
) -> dict[str, list[RealTaskRecord]]:
    _ = seed
    if context_grounding_mode not in {"real_task_v1", "grounded_v1", "stage44a_path_grounded_v2"}:
        raise ValueError(f"unsupported context_grounding_mode: {context_grounding_mode}")
    records: dict[str, list[RealTaskRecord]] = {}
    templates = {
        "operation_decision": "The incident desk has a stable surface report. Operators must choose the route from hidden operational context.",
        "rule_conflict": "The policy review contains competing recommendations. The controlling rule decides the final route.",
        "causal_trace": "A customer-visible event follows an upstream system change. Context decides which causal explanation is active.",
        "condition_check": "A request is valid only when the surrounding condition profile is satisfied.",
        "priority_selection": "Several tasks are ready, but the priority context selects the action that must happen first.",
        "negation_constraint": "The visible note is ambiguous, and the negative constraint context decides what must not be accepted.",
    }
    for task_type in LOCAL_REAL_TASK_TYPES:
        rows: list[RealTaskRecord] = []
        for index in range(samples_per_label):
            group_id = f"local_{task_type}_surface_{index:04d}"
            shared_text = f"{templates[task_type]} Case {index:04d} keeps identical surface text for every context."
            for label in LOGIC_LABELS:
                if context_grounding_mode == "stage44a_path_grounded_v2":
                    memories, rules, state_values, required_path, counterfactual = _stage44a_context(
                        label, task_type, group_id
                    )
                else:
                    memories, rules = _record_context(label, task_type, group_id, context_grounding_mode)
                    state_values = _state_tuple(label)
                    required_path = ""
                    counterfactual = ""
                context_payload = json.dumps(
                    {
                        "memory": [asdict(item) for item in memories],
                        "rule": [asdict(item) for item in rules],
                        "state": state_values,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                rows.append(
                    RealTaskRecord(
                        text=shared_text,
                        label=label,
                        task_type=task_type,
                        memory_items=memories,
                        state_values=state_values,
                        rule_items=rules,
                        expected_answer=f"{task_type} expects {label}",
                        surface_group_id=group_id,
                        source_type="local_semireal",
                        source_id=f"{group_id}_{label}",
                        required_path=required_path,
                        context_owner="rule+conflict" if task_type == "rule_conflict" else required_path,
                        counterfactual_context=counterfactual,
                        context_hash=hashlib.sha256(context_payload.encode("utf-8")).hexdigest(),
                    )
                )
        records[task_type] = rows
    return records


def _neutral_memory(group_id: str) -> tuple[MemoryItem, ...]:
    return (
        MemoryItem(
            id=f"memory_{group_id}_neutral",
            summary="routine background evidence",
            content="the archived note does not select any available action",
            relation_type="background",
            priority=0.1,
            confidence=0.5,
        ),
    )


def _neutral_rule(group_id: str) -> tuple[RuleItem, ...]:
    return (
        RuleItem(
            id=f"rule_{group_id}_neutral",
            type="soft",
            condition="routine review",
            effect="preserve the available alternatives without selecting one",
            priority=0.1,
            source="stage44a_path_grounded_v2",
        ),
    )


def _stage44a_context(
    label: str,
    task_type: str,
    group_id: str,
) -> tuple[tuple[MemoryItem, ...], tuple[RuleItem, ...], tuple[float, float, float], str, str]:
    from .evidence_answer_data import LABEL_TO_LOCAL_OPTION

    option = LABEL_TO_LOCAL_OPTION[label]
    option_index = list(LOGIC_LABELS).index(label)
    wrong_label = LOGIC_LABELS[(option_index + 1) % len(LOGIC_LABELS)]
    wrong_option = LABEL_TO_LOCAL_OPTION[wrong_label]
    neutral_state = (0.5, 0.5, 0.5)
    if task_type in {"operation_decision", "causal_trace"}:
        memories = (
            MemoryItem(
                id=f"memory_{group_id}_{option_index}",
                summary="task evidence",
                content=f"the verified operational evidence supports the action: {option}",
                relation_type="evidence_to_action",
                priority=1.0,
                confidence=1.0,
            ),
        )
        return memories, _neutral_rule(group_id), neutral_state, "memory", wrong_option
    if task_type == "priority_selection":
        rules = (
            RuleItem(
                id=f"rule_{group_id}_{option_index}",
                type="hard",
                condition="several actions remain available",
                effect=f"the controlling policy requires the action: {option}",
                priority=1.0,
                source="stage44a_path_grounded_v2",
            ),
        )
        return _neutral_memory(group_id), rules, neutral_state, "rule", wrong_option
    if task_type == "rule_conflict":
        memories = (
            MemoryItem(
                id=f"memory_{group_id}_misdirection_{option_index}",
                summary="superseded recommendation",
                content=f"the non-controlling recommendation prefers: {wrong_option}",
                relation_type="superseded",
                priority=0.2,
                confidence=0.8,
            ),
        )
        rules = (
            RuleItem(
                id=f"rule_{group_id}_{option_index}",
                type="conflict",
                condition="a recommendation conflicts with the controlling policy",
                effect=f"ignore the recommendation and apply: {option}",
                priority=1.0,
                source="stage44a_path_grounded_v2",
            ),
        )
        return memories, rules, (0.9, 0.1, 0.9), "memory+rule+state", wrong_option
    if task_type in {"condition_check", "negation_constraint"}:
        return _neutral_memory(group_id), _neutral_rule(group_id), _state_tuple(label), "state", wrong_option
    raise ValueError(f"unsupported Stage44A task type: {task_type}")


def records_to_logic_datasets(
    records_by_task: dict[str, list[RealTaskRecord]],
    max_seq_len: int,
    context_grounding_mode: str = "real_task_v1",
) -> tuple[dict[str, list[LogicSample]], LogicTokenizer]:
    tokenizer = LogicTokenizer(max_seq_len=max_seq_len)
    tokenizer.fit(record.text for rows in records_by_task.values() for record in rows)
    datasets: dict[str, list[LogicSample]] = {}
    for task_type, records in records_by_task.items():
        datasets[task_type] = [
            record.to_logic_sample(
                tokenizer,
                template_id=index,
                context_grounding_mode=context_grounding_mode,
            )
            for index, record in enumerate(records)
        ]
    return datasets, tokenizer


def split_by_surface_group(samples: list[LogicSample], train_groups: int, seed: int) -> tuple[list[LogicSample], list[LogicSample]]:
    _ = seed
    groups = sorted({sample.surface_group_id for sample in samples})
    selected = set(groups[:train_groups])
    train = [sample for sample in samples if sample.surface_group_id in selected]
    test = [sample for sample in samples if sample.surface_group_id not in selected]
    if {sample.surface_group_id for sample in train} & {sample.surface_group_id for sample in test}:
        raise ValueError("surface group leaked across train/test split")
    return train, test


def local_real_task_manifest(records_by_task: dict[str, list[RealTaskRecord]]) -> list[dict[str, Any]]:
    rows = []
    for task_type, records in records_by_task.items():
        rows.append(
            {
                "source_type": "local_semireal",
                "task_type": task_type,
                "num_records": len(records),
                "labels": sorted({record.label for record in records}),
                "surface_groups": len({record.surface_group_id for record in records}),
                "fingerprint": _fingerprint([asdict(record) for record in records]),
            }
        )
    return rows


def _fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str | Path, task_name: str) -> Path:
    return Path(cache_dir) / f"{task_name}.jsonl"


def _split_external_evidence_query(text: str, task_name: str) -> tuple[str, str, str]:
    normalized = " ".join(text.split())
    split_at = -1
    for marker in (". ", "? ", "! "):
        index = normalized.rfind(marker)
        if index > split_at:
            split_at = index + len(marker)
    if 0 < split_at < len(normalized):
        return normalized[:split_at].strip(), normalized[split_at:].strip(), "parsed_text_tail"
    words = normalized.split()
    tail_words = 12 if task_name == "boolq" else 8
    if len(words) <= tail_words:
        return normalized, "", "unrecoverable"
    return " ".join(words[:-tail_words]).strip(), " ".join(words[-tail_words:]).strip(), "fallback_tail_words"


def _structured_fields_from_text(task_name: str, text: str) -> dict[str, str]:
    evidence, query, source = _split_external_evidence_query(text, task_name)
    if task_name in {"glue_rte", "super_glue_cb"}:
        return {
            "premise": evidence,
            "hypothesis": query,
            "passage": "",
            "question": "",
            "structure_source": source,
        }
    if task_name == "boolq":
        return {
            "premise": "",
            "hypothesis": "",
            "passage": evidence,
            "question": query,
            "structure_source": source,
        }
    return {
        "premise": "",
        "hypothesis": "",
        "passage": "",
        "question": "",
        "structure_source": "unstructured_text",
    }


def _normalize_external_row(
    task_name: str,
    row: dict[str, Any],
    index: int,
    context_grounding_mode: str = "real_task_v1",
) -> RealTaskRecord | None:
    spec = EXTERNAL_DATASET_SPECS[task_name]
    raw_label = row.get(spec["label_field"])
    if raw_label not in spec["label_map"]:
        return None
    label, external_label = spec["label_map"][raw_label]
    text_parts = [str(row.get(field, "")).strip() for field in spec["text_fields"]]
    if any(not part for part in text_parts):
        return None
    text = " ".join(text_parts)
    if task_name in {"glue_rte", "super_glue_cb"}:
        structured = {
            "premise": text_parts[0],
            "hypothesis": text_parts[1],
            "passage": "",
            "question": "",
            "structure_source": "dataset_fields",
        }
    elif task_name == "boolq":
        structured = {
            "premise": "",
            "hypothesis": "",
            "passage": text_parts[0],
            "question": text_parts[1],
            "structure_source": "dataset_fields",
        }
    else:
        structured = _structured_fields_from_text(task_name, text)
    group_id = f"external_{task_name}_surface_{index:05d}"
    memories, rules = _record_context(label, task_name, group_id, context_grounding_mode)
    return RealTaskRecord(
        text=text,
        label=label,
        task_type=task_name,
        memory_items=memories,
        state_values=_state_tuple(label),
        rule_items=rules,
        expected_answer=external_label,
        surface_group_id=group_id,
        source_type="external_benchmark",
        source_id=f"{task_name}_{index:05d}",
        external_label=external_label,
        premise=structured["premise"],
        hypothesis=structured["hypothesis"],
        passage=structured["passage"],
        question=structured["question"],
        structure_source=structured["structure_source"],
    )


def _write_cache(path: Path, records: list[RealTaskRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def _read_cache(path: Path) -> list[RealTaskRecord]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            memories = tuple(MemoryItem(**item) for item in row["memory_items"])
            rules = tuple(RuleItem(**item) for item in row["rule_items"])
            structured = {
                "premise": row.get("premise", ""),
                "hypothesis": row.get("hypothesis", ""),
                "passage": row.get("passage", ""),
                "question": row.get("question", ""),
                "structure_source": row.get("structure_source", ""),
            }
            if not structured["structure_source"]:
                structured = _structured_fields_from_text(row["task_type"], row["text"])
            records.append(
                RealTaskRecord(
                    text=row["text"],
                    label=row["label"],
                    task_type=row["task_type"],
                    memory_items=memories,
                    state_values=tuple(row["state_values"]),
                    rule_items=rules,
                    expected_answer=row["expected_answer"],
                    surface_group_id=row["surface_group_id"],
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    external_label=row.get("external_label", ""),
                    premise=structured["premise"],
                    hypothesis=structured["hypothesis"],
                    passage=structured["passage"],
                    question=structured["question"],
                    structure_source=structured["structure_source"],
                )
            )
    return records


def _reground_records(records: list[RealTaskRecord], context_grounding_mode: str) -> list[RealTaskRecord]:
    if context_grounding_mode == "real_task_v1":
        return records
    grounded = []
    for record in records:
        memories, rules = _record_context(
            record.label,
            record.task_type,
            record.surface_group_id,
            context_grounding_mode=context_grounding_mode,
        )
        grounded.append(
            RealTaskRecord(
                text=record.text,
                label=record.label,
                task_type=record.task_type,
                memory_items=memories,
                state_values=_state_tuple(record.label),
                rule_items=rules,
                expected_answer=record.expected_answer,
                surface_group_id=record.surface_group_id,
                source_type=record.source_type,
                source_id=record.source_id,
                external_label=record.external_label,
                premise=record.premise,
                hypothesis=record.hypothesis,
                passage=record.passage,
                question=record.question,
                structure_source=record.structure_source,
            )
        )
    return grounded


def solidify_structured_external_cache(
    source_cache_dir: str | Path = "src/civilization/engine/data/external_cache",
    output_cache_dir: str | Path = "src/civilization/engine/data/external_cache_structured",
    task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
) -> list[dict[str, Any]]:
    records_by_task, source_manifest = load_external_task_records(
        cache_dir=source_cache_dir,
        task_names=task_names,
        allow_download=False,
        context_grounding_mode="real_task_v1",
    )
    output = Path(output_cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    source_by_task = {row["task_name"]: row for row in source_manifest}
    for task_name, records in records_by_task.items():
        _write_cache(_cache_path(output, task_name), records)
        structure_sources = sorted({record.structure_source for record in records})
        structured_ok = all(
            (
                bool(record.premise and record.hypothesis)
                if task_name in {"glue_rte", "super_glue_cb"}
                else bool(record.passage and record.question)
            )
            for record in records
        )
        manifest.append(
            {
                "stage": "stage44b5_structured_cache",
                "task_name": task_name,
                "source_cache_path": source_by_task[task_name]["cache_path"],
                "structured_cache_path": str(_cache_path(output, task_name)),
                "num_records": len(records),
                "external_labels": sorted({record.external_label for record in records}),
                "structure_sources": structure_sources,
                "structured_ok": structured_ok,
                "fingerprint": _fingerprint([asdict(record) for record in records]),
            }
        )
    (output / "structured_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_external_task_records(
    cache_dir: str | Path = "src/civilization/engine/data/external_cache",
    task_names: tuple[str, ...] = tuple(EXTERNAL_DATASET_SPECS),
    allow_download: bool | None = None,
    max_records_per_task: int = 320,
    context_grounding_mode: str = "real_task_v1",
) -> tuple[dict[str, list[RealTaskRecord]], list[dict[str, Any]]]:
    if context_grounding_mode not in {"real_task_v1", "grounded_v1"}:
        raise ValueError(f"unsupported context_grounding_mode: {context_grounding_mode}")
    allow = (os.environ.get("ALLOW_DATASET_DOWNLOAD") == "1") if allow_download is None else allow_download
    datasets: dict[str, list[RealTaskRecord]] = {}
    manifest: list[dict[str, Any]] = []
    for task_name in task_names:
        if task_name not in EXTERNAL_DATASET_SPECS:
            raise ValueError(f"unknown external task: {task_name}")
        path = _cache_path(cache_dir, task_name)
        if path.exists():
            records = _reground_records(_read_cache(path), context_grounding_mode)
            source = "cache"
        elif allow:
            try:
                from datasets import load_dataset
            except ModuleNotFoundError as error:
                raise RuntimeError("datasets package is required for ALLOW_DATASET_DOWNLOAD=1") from error
            spec = EXTERNAL_DATASET_SPECS[task_name]
            loaded = load_dataset(
                spec["dataset_id"],
                spec["config"],
                split=spec["split"],
                revision=spec["revision"],
            )
            records = []
            for index, row in enumerate(loaded):
                record = _normalize_external_row(
                    task_name,
                    dict(row),
                    index,
                    context_grounding_mode=context_grounding_mode,
                )
                if record is not None:
                    records.append(record)
                if len(records) >= max_records_per_task:
                    break
            _write_cache(path, records)
            source = "download"
        else:
            raise FileNotFoundError(
                f"missing external cache {path}; set ALLOW_DATASET_DOWNLOAD=1 to create it"
            )
        if not records:
            raise ValueError(f"external task {task_name} has no usable records")
        spec = EXTERNAL_DATASET_SPECS[task_name]
        datasets[task_name] = records
        manifest.append(
            {
                "source_type": "external_benchmark",
                "task_name": task_name,
                "dataset_id": spec["dataset_id"],
                "config": spec["config"],
                "split": spec["split"],
                "revision": spec["revision"],
                "source": source,
                "cache_path": str(path),
                "num_records": len(records),
                "labels": sorted({record.label for record in records}),
                "external_labels": sorted({record.external_label for record in records}),
                "fingerprint": _fingerprint([asdict(record) for record in records]),
            }
        )
    return datasets, manifest
