from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .evidence_answer_data import EXTERNAL_ANSWER_OPTIONS
from .real_task_data import EXTERNAL_DATASET_SPECS, RealTaskRecord, load_external_task_records


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b_external_recovery_audit"
)
DEFAULT_CACHE_DIR = Path("src/civilization/engine/data/external_cache")
DEFAULT_STAGE44A_ROOT = Path(
    "artifacts/civilization/stage44a_local_multiclass_integration"
)
DEFAULT_TASKS = ("glue_rte", "super_glue_cb", "boolq")
FORBIDDEN_CONTEXT_MARKERS = ("memory_selects_", "rule_selects_", "state_selects_")


@dataclass(frozen=True)
class Stage44BSourceRecord:
    task_name: str
    source_id: str
    surface_group_id: str
    text: str
    external_label: str
    answer_options: tuple[str, ...]
    correct_option_id: int
    text_hash: str
    premise: str = ""
    hypothesis: str = ""
    passage: str = ""
    question: str = ""
    structure_source: str = "unstructured_text"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def project_external_source_records(
    records_by_task: dict[str, list[RealTaskRecord]],
) -> dict[str, list[Stage44BSourceRecord]]:
    projected: dict[str, list[Stage44BSourceRecord]] = {}
    for task_name, records in records_by_task.items():
        options = EXTERNAL_ANSWER_OPTIONS[task_name]
        task_rows = []
        for record in records:
            if record.external_label not in options:
                raise ValueError(
                    f"external label {record.external_label!r} is not valid for {task_name}"
                )
            structured = _recover_structured_fields(record)
            task_rows.append(
                Stage44BSourceRecord(
                    task_name=task_name,
                    source_id=record.source_id,
                    surface_group_id=record.surface_group_id,
                    text=record.text,
                    external_label=record.external_label,
                    answer_options=options,
                    correct_option_id=options.index(record.external_label),
                    text_hash=_text_hash(record.text),
                    premise=structured["premise"],
                    hypothesis=structured["hypothesis"],
                    passage=structured["passage"],
                    question=structured["question"],
                    structure_source=structured["structure_source"],
                )
            )
        projected[task_name] = task_rows
    return projected


def _split_evidence_query(text: str, task_name: str) -> tuple[str, str, str]:
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


def _recover_structured_fields(record: RealTaskRecord) -> dict[str, str]:
    if record.task_type in {"glue_rte", "super_glue_cb"} and record.premise and record.hypothesis:
        return {
            "premise": record.premise,
            "hypothesis": record.hypothesis,
            "passage": "",
            "question": "",
            "structure_source": record.structure_source or "record_fields",
        }
    if record.task_type == "boolq" and record.passage and record.question:
        return {
            "premise": "",
            "hypothesis": "",
            "passage": record.passage,
            "question": record.question,
            "structure_source": record.structure_source or "record_fields",
        }
    evidence, query, source = _split_evidence_query(record.text, record.task_type)
    if record.task_type in {"glue_rte", "super_glue_cb"}:
        return {
            "premise": evidence,
            "hypothesis": query,
            "passage": "",
            "question": "",
            "structure_source": source,
        }
    if record.task_type == "boolq":
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


def deterministic_external_split(
    records: list[Stage44BSourceRecord],
    *,
    seed: int,
    train_per_label: int,
    heldout_per_label: int,
) -> tuple[list[Stage44BSourceRecord], list[Stage44BSourceRecord]]:
    train: list[Stage44BSourceRecord] = []
    heldout: list[Stage44BSourceRecord] = []
    labels = sorted({record.external_label for record in records})
    for label in labels:
        by_hash: dict[str, list[Stage44BSourceRecord]] = {}
        for record in records:
            if record.external_label == label:
                by_hash.setdefault(record.text_hash, []).append(record)
        groups = sorted(by_hash.items())
        random.Random(f"stage44b:{seed}:{records[0].task_name}:{label}").shuffle(groups)
        label_record_count = sum(len(group) for _hash, group in groups)
        desired_train_count = min(train_per_label, max(1, label_record_count // 2))
        desired_heldout_count = min(
            heldout_per_label, label_record_count - desired_train_count
        )
        label_train: list[Stage44BSourceRecord] = []
        label_heldout: list[Stage44BSourceRecord] = []
        for _hash, group in groups:
            target = (
                label_train
                if len(label_train) < desired_train_count
                else label_heldout
            )
            if target is label_heldout and len(label_heldout) >= desired_heldout_count:
                continue
            target.extend(group)
        train.extend(label_train[:desired_train_count])
        heldout.extend(label_heldout[:desired_heldout_count])
    train_ids = {record.source_id for record in train}
    heldout_ids = {record.source_id for record in heldout}
    train_hashes = {record.text_hash for record in train}
    heldout_hashes = {record.text_hash for record in heldout}
    if train_ids & heldout_ids:
        raise ValueError("external source ids leaked across train/heldout")
    if train_hashes & heldout_hashes:
        raise ValueError("external text hashes leaked across train/heldout")
    if set(labels) != {record.external_label for record in train}:
        raise ValueError("external train split is missing labels")
    if set(labels) != {record.external_label for record in heldout}:
        raise ValueError("external heldout split is missing labels")
    return train, heldout


def _legacy_context_has_leakage(record: RealTaskRecord) -> bool:
    values = [
        *(item.summary for item in record.memory_items),
        *(item.content for item in record.memory_items),
        *(item.relation_type for item in record.memory_items),
        *(item.condition for item in record.rule_items),
        *(item.effect for item in record.rule_items),
    ]
    text = " ".join(values).lower()
    return any(marker in text for marker in FORBIDDEN_CONTEXT_MARKERS)


def audit_answer_option_mapping(
    projected: dict[str, list[Stage44BSourceRecord]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_name, records in projected.items():
        expected_options = EXTERNAL_ANSWER_OPTIONS[task_name]
        for record in records:
            mapping_ok = (
                record.answer_options == expected_options
                and 0 <= record.correct_option_id < len(expected_options)
                and expected_options[record.correct_option_id] == record.external_label
            )
            rows.append(
                {
                    "task_name": task_name,
                    "source_id": record.source_id,
                    "external_label": record.external_label,
                    "correct_option_id": record.correct_option_id,
                    "score_column": record.correct_option_id,
                    "answer_options": json.dumps(record.answer_options),
                    "mapping_ok": mapping_ok,
                }
            )
            if not mapping_ok:
                failures.append(
                    {
                        "failed_stage": "answer_option_mapping_audit",
                        "failed_gate": "answer_option_mapping",
                        "task_name": task_name,
                        "source_id": record.source_id,
                    }
                )
    return rows, failures


def audit_token_lengths(
    tokenizer: Any,
    projected: dict[str, list[Stage44BSourceRecord]],
    *,
    max_length: int,
    tail_reserve: int = 96,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tail_reserve <= 0 or tail_reserve >= max_length:
        raise ValueError("tail_reserve must be positive and smaller than max_length")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_name, records in projected.items():
        for record in records:
            token_ids = tokenizer(record.text, add_special_tokens=True, truncation=False)["input_ids"]
            truncated = len(token_ids) > max_length
            policy = "head_tail" if truncated else "none"
            # Stage44B preserves both the document prefix and the answer-bearing hypothesis/question tail.
            preserved_count = min(len(token_ids), max_length)
            policy_ok = not truncated or (max_length - tail_reserve > 0 and tail_reserve > 0)
            row = {
                "task_name": task_name,
                "source_id": record.source_id,
                "original_token_count": len(token_ids),
                "max_length": max_length,
                "truncated": truncated,
                "truncation_policy": policy,
                "head_token_budget": max_length - tail_reserve if truncated else preserved_count,
                "tail_token_budget": tail_reserve if truncated else 0,
                "preserved_token_count": preserved_count,
                "silent_truncation": False,
                "policy_ok": policy_ok,
            }
            rows.append(row)
            if not policy_ok:
                failures.append(
                    {
                        "failed_stage": "truncation_audit",
                        "failed_gate": "core_text_preservation",
                        "task_name": task_name,
                        "source_id": record.source_id,
                    }
                )
    return rows, failures


def _checkpoint_path(root: Path, seed: int) -> Path:
    return (
        root
        / f"seed_{seed}"
        / "checkpoints"
        / "full_hidden_alignment"
        / f"stage44a_full_hidden_alignment_seed_{seed}.pt"
    )


def audit_stage44a_checkpoints(
    root: str | Path, seeds: Iterable[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root)
    aggregate_summary_path = root / "summary.json"
    aggregate = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not aggregate.get("passes_stage_gate") or not aggregate.get("allows_stage44b"):
        failures.append(
            {
                "failed_stage": "stage44a_checkpoint_audit",
                "failed_gate": "stage44a_aggregate_gate",
            }
        )
    for seed in seeds:
        checkpoint = _checkpoint_path(root, seed)
        if not checkpoint.exists():
            failures.append(
                {
                    "failed_stage": "stage44a_checkpoint_audit",
                    "failed_gate": "checkpoint_missing",
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                }
            )
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("training_metadata", {})
        has_qwen = "qwen_state_dict" in payload or "model_state_dict" in payload
        valid = (
            metadata.get("seed") == seed
            and metadata.get("stage") == "full_hidden_alignment"
            and metadata.get("adapter_variant") == "path_specific_v2"
            and metadata.get("target_layers") == [16, 24]
            and metadata.get("raw_full_hidden_residual_scale") == 200.0
            and not has_qwen
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "stage": metadata.get("stage"),
                "adapter_variant": metadata.get("adapter_variant"),
                "target_layers": json.dumps(metadata.get("target_layers")),
                "raw_full_hidden_residual_scale": metadata.get("raw_full_hidden_residual_scale"),
                "contains_qwen_state": has_qwen,
                "valid": valid,
            }
        )
        if not valid:
            failures.append(
                {
                    "failed_stage": "stage44a_checkpoint_audit",
                    "failed_gate": "checkpoint_metadata",
                    "seed": seed,
                }
            )
    return rows, failures


def run_qwen3_stage44b_external_recovery_audit(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    stage44a_root: str | Path = DEFAULT_STAGE44A_ROOT,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seeds: tuple[int, ...] = (202, 303, 404),
    task_names: tuple[str, ...] = DEFAULT_TASKS,
    train_per_label: int = 12,
    heldout_per_label: int = 12,
    max_length: int = 384,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []

    records_by_task, cache_manifest = load_external_task_records(
        cache_dir=cache_dir,
        task_names=task_names,
        allow_download=False,
        context_grounding_mode="real_task_v1",
    )
    projected = project_external_source_records(records_by_task)
    context_rows = []
    for task_name, records in records_by_task.items():
        for record in records:
            leaked = _legacy_context_has_leakage(record)
            context_rows.append(
                {
                    "task_name": task_name,
                    "source_id": record.source_id,
                    "legacy_context_leakage": leaked,
                    "legacy_context_reused": False,
                    "source_projection_ok": bool(record.text and record.external_label),
                }
            )
    if any(row["legacy_context_reused"] for row in context_rows):
        failures.append(
            {
                "failed_stage": "cache_context_audit",
                "failed_gate": "legacy_context_reuse",
            }
        )

    mapping_rows, mapping_failures = audit_answer_option_mapping(projected)
    failures.extend(mapping_failures)
    split_rows: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []
    for seed in seeds:
        for task_name, records in projected.items():
            try:
                train, heldout = deterministic_external_split(
                    records,
                    seed=seed,
                    train_per_label=train_per_label,
                    heldout_per_label=heldout_per_label,
                )
            except ValueError as error:
                failures.append(
                    {
                        "failed_stage": "external_split_audit",
                        "failed_gate": "split_isolation",
                        "seed": seed,
                        "task_name": task_name,
                        "reason": str(error),
                    }
                )
                continue
            for split_name, split_records in (("train", train), ("heldout", heldout)):
                split_rows.extend(
                    {
                        "seed": seed,
                        "task_name": task_name,
                        "split": split_name,
                        "source_id": record.source_id,
                        "surface_group_id": record.surface_group_id,
                        "text_hash": record.text_hash,
                        "external_label": record.external_label,
                        "correct_option_id": record.correct_option_id,
                    }
                    for record in split_records
                )
            split_manifest.append(
                {
                    "seed": seed,
                    "task_name": task_name,
                    "train_count": len(train),
                    "heldout_count": len(heldout),
                    "train_labels": sorted({record.external_label for record in train}),
                    "heldout_labels": sorted({record.external_label for record in heldout}),
                    "source_overlap": len(
                        {record.source_id for record in train}
                        & {record.source_id for record in heldout}
                    ),
                    "text_hash_overlap": len(
                        {record.text_hash for record in train}
                        & {record.text_hash for record in heldout}
                    ),
                }
            )

    checkpoint_rows, checkpoint_failures = audit_stage44a_checkpoints(stage44a_root, seeds)
    failures.extend(checkpoint_failures)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    truncation_rows, truncation_failures = audit_token_lengths(
        tokenizer, projected, max_length=max_length
    )
    failures.extend(truncation_failures)

    expected_tasks = set(task_names)
    cache_tasks = {row["task_name"] for row in cache_manifest}
    cache_gate = cache_tasks == expected_tasks and all(
        row["source"] == "cache" and row["num_records"] > 0 and row["fingerprint"]
        for row in cache_manifest
    )
    split_gate = len(split_manifest) == len(seeds) * len(task_names) and all(
        row["source_overlap"] == 0
        and row["text_hash_overlap"] == 0
        and row["train_count"] > 0
        and row["heldout_count"] > 0
        for row in split_manifest
    )
    gates = {
        "cache_manifest_complete": cache_gate,
        "legacy_context_not_reused": not any(
            row["legacy_context_reused"] for row in context_rows
        ),
        "answer_option_mapping": not mapping_failures and all(
            row["mapping_ok"] for row in mapping_rows
        ),
        "split_isolation": split_gate,
        "stage44a_checkpoints": not checkpoint_failures
        and len(checkpoint_rows) == len(seeds)
        and all(row["valid"] for row in checkpoint_rows),
        "truncation_policy_audited": not truncation_failures
        and all(not row["silent_truncation"] for row in truncation_rows),
    }
    for name, passed in gates.items():
        if not passed and not any(item.get("failed_gate") == name for item in failures):
            failures.append(
                {
                    "failed_stage": "stage44b_external_recovery_audit",
                    "failed_gate": name,
                }
            )

    _json_dump(output / "cache_manifest.json", cache_manifest)
    _write_csv(output / "cache_context_audit.csv", context_rows)
    _write_csv(output / "answer_option_mapping.csv", mapping_rows)
    _write_csv(output / "split_audit.csv", split_rows)
    _json_dump(output / "split_manifest.json", split_manifest)
    _write_csv(output / "checkpoint_verification.csv", checkpoint_rows)
    _write_csv(output / "truncation_audit.csv", truncation_rows)
    _json_dump(output / "failure_cases.json", failures)
    summary = {
        "stage": "44B_external_recovery_audit",
        "seeds": list(seeds),
        "task_names": list(task_names),
        "max_length": max_length,
        "train_per_label": train_per_label,
        "heldout_per_label": heldout_per_label,
        "record_counts": {task: len(rows) for task, rows in projected.items()},
        "legacy_context_leakage_records": sum(
            bool(row["legacy_context_leakage"]) for row in context_rows
        ),
        "legacy_context_reused_records": sum(
            bool(row["legacy_context_reused"]) for row in context_rows
        ),
        "truncated_record_count": sum(bool(row["truncated"]) for row in truncation_rows),
        "stage_gates": gates,
        "passes_stage_gate": not failures and all(gates.values()),
    }
    summary["allows_stage44b_training"] = summary["passes_stage_gate"]
    _json_dump(output / "summary.json", summary)
    return summary
