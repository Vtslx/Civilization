from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import psutil
import torch
from torch import nn
import torch.nn.functional as F

from civilization.research.torch_line.analysis.dataset import LogicSample
from civilization.research.torch_line.model import CivilizationAblationConfig

from ..adapter.context_encoder import FrozenQwenContextEncoder
from ..backend import Qwen3Backend
from .adapter_benchmark import DEFAULT_MODEL_PATH
from .answer_option_readout import build_answer_option_vectors
from .context_readout_alignment import PathReadoutProjector
from .evidence_answer_data import EXTERNAL_ANSWER_OPTIONS, LOCAL_ANSWER_OPTIONS
from .evidence_answer_training import _answer_scores
from .group_full_hidden_centroid_integration import _load_stage41_checkpoint
from .hidden_states import last_non_padding_pool
from .real_task_data import build_local_semireal_task_records, records_to_logic_datasets
from .real_task_data import solidify_structured_external_cache
from .stage44a_local_multiclass_integration import (
    _evaluate_groups as _evaluate_local_groups,
    _forward_local_group,
    _path_for_task,
    _path_projected,
    build_stage44a_groups,
    split_stage44a_groups,
)
from .stage44b_external_recovery_audit import (
    DEFAULT_CACHE_DIR,
    DEFAULT_STAGE44A_ROOT,
    Stage44BSourceRecord,
    deterministic_external_split,
    project_external_source_records,
)
from .real_task_data import load_external_task_records


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b_external_recovery_training"
)
DEFAULT_STAGE44B1_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b1_external_recovery_training"
)
DEFAULT_STAGE44B2_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b2_external_recovery_training"
)
DEFAULT_STAGE44B3_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b3_external_recovery_training"
)
DEFAULT_STAGE44B4_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b4_external_recovery_training"
)
DEFAULT_STAGE44B5_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b5_external_recovery_training"
)
DEFAULT_STAGE44B6_OUTPUT_DIR = Path(
    "artifacts/civilization/stage44b6_external_recovery_training"
)
DEFAULT_STRUCTURED_CACHE_DIR = Path("src/civilization/engine/data/external_cache_structured")
DEFAULT_DATASET_FIELDS_CACHE_DIR = Path("src/civilization/engine/data/external_cache_dataset_fields")
TASK_RULES = {
    "glue_rte": "Judge whether the hypothesis follows from the premise using only the supplied evidence.",
    "super_glue_cb": "Judge whether the hypothesis is entailed, contradicted, or left uncertain by the premise.",
    "boolq": "Judge whether the passage supports a true or false answer to the question.",
}
EVAL_MODES = (
    "full",
    "adapter_disabled",
    "zero_scale",
    "no_memory_path",
    "no_rule_path",
    "wrong_context",
)
EXTERNAL_CONTEXT_SCALE = 0.5
STAGE44B1_CONTEXT_SCALE = 0.65
STAGE44B2_CONTEXT_SCALE = 0.80
STAGE44B3_CONTEXT_SCALE = 0.85
STAGE44B4_CONTEXT_SCALE = 0.85


@dataclass(frozen=True)
class ExternalRecoveryRecord:
    source: Stage44BSourceRecord
    sample: LogicSample
    wrong_sample: LogicSample
    wrong_option_id: int


@dataclass(frozen=True)
class ExternalTaskBoundary:
    task_name: str
    source_id: str
    evidence_text: str
    query_text: str
    boundary_source: str
    boundary_ok: bool


class TaskSpecificPathReadoutProjector(nn.Module):
    def __init__(self, task_names: tuple[str, ...]):
        super().__init__()
        self.projectors = nn.ModuleDict(
            {task_name: PathReadoutProjector() for task_name in task_names}
        )

    def forward(self, vector: torch.Tensor, task_name: str) -> torch.Tensor:
        return self.projectors[task_name](vector)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _evidence_snippet(text: str, head_words: int = 8, tail_words: int = 12) -> str:
    words = text.split()
    if len(words) <= head_words + tail_words:
        return text
    return " ".join((*words[:head_words], "[...evidence omitted...]", *words[-tail_words:]))


def recover_external_task_boundary(source: Stage44BSourceRecord) -> ExternalTaskBoundary:
    if source.task_name in {"glue_rte", "super_glue_cb"} and source.premise and source.hypothesis:
        return ExternalTaskBoundary(
            task_name=source.task_name,
            source_id=source.source_id,
            evidence_text=source.premise,
            query_text=source.hypothesis,
            boundary_source=source.structure_source,
            boundary_ok=True,
        )
    if source.task_name == "boolq" and source.passage and source.question:
        return ExternalTaskBoundary(
            task_name=source.task_name,
            source_id=source.source_id,
            evidence_text=source.passage,
            query_text=source.question,
            boundary_source=source.structure_source,
            boundary_ok=True,
        )
    text = " ".join(source.text.split())
    split_at = -1
    for marker in (". ", "? ", "! "):
        index = text.rfind(marker)
        if index > split_at:
            split_at = index + len(marker)
    if 0 < split_at < len(text):
        evidence = text[:split_at].strip()
        query = text[split_at:].strip()
        boundary_source = "parsed_text_tail"
    else:
        words = text.split()
        tail_words = 12 if source.task_name == "boolq" else 8
        evidence = " ".join(words[:-tail_words]).strip()
        query = " ".join(words[-tail_words:]).strip()
        boundary_source = "fallback_tail_words"
    return ExternalTaskBoundary(
        task_name=source.task_name,
        source_id=source.source_id,
        evidence_text=evidence,
        query_text=query,
        boundary_source=boundary_source,
        boundary_ok=bool(evidence and query and evidence != query),
    )


def _structured_external_text(source: Stage44BSourceRecord, boundary: ExternalTaskBoundary) -> str:
    if source.task_name in {"glue_rte", "super_glue_cb"}:
        return f"Premise: {boundary.evidence_text}\nHypothesis: {boundary.query_text}"
    if source.task_name == "boolq":
        return f"Passage: {boundary.evidence_text}\nQuestion: {boundary.query_text}"
    return source.text


def _simple_sentences(text: str) -> list[str]:
    parts = []
    start = 0
    for index, char in enumerate(text):
        if char in ".?!":
            current = text[start : index + 1].strip()
            if current:
                parts.append(current)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [text]


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "to", "of", "in", "on", "for", "and", "or", "that",
    "this", "it", "its", "be", "by", "with", "from", "as", "at", "has",
    "have", "had", "more", "most", "less", "than",
}


def _normalized_terms(text: str) -> set[str]:
    return {
        word
        for word in (
            part.strip(".,?!'\"`()[]{}:;").lower()
            for part in text.split()
        )
        if len(word) > 2 and word not in STOPWORDS
    }


def _boolq_relevant_evidence(boundary: ExternalTaskBoundary) -> str:
    query_terms = _normalized_terms(boundary.query_text)
    sentences = _simple_sentences(boundary.evidence_text)
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _normalized_terms(sentence)
        score = len(query_terms & sentence_terms)
        scored.append((score, -index, sentence))
    selected = [sentence for score, _index, sentence in sorted(scored, reverse=True)[:2] if score > 0]
    if not selected:
        selected = sentences[:1] + sentences[-1:]
    return " ".join(dict.fromkeys(selected))


def _grounded_memory_text(
    source: Stage44BSourceRecord,
    boundary: ExternalTaskBoundary,
    *,
    grounding_mode: str = "stage44b1_grounded_v2",
    target_label: str | None = None,
) -> str:
    if grounding_mode == "stage44b4_structured_semantic_v5":
        return _semantic_context_text(source, boundary, target_label or source.external_label)
    if source.task_name == "boolq":
        evidence = (
            _boolq_relevant_evidence(boundary)
            if grounding_mode == "stage44b2_grounded_v3"
            or grounding_mode == "stage44b3_grounded_v4"
            else boundary.evidence_text
        )
        return (
            f"passage evidence: {_evidence_snippet(evidence, 12, 20)} "
            f"question: {boundary.query_text}"
        )
    return (
        f"premise evidence: {_evidence_snippet(boundary.evidence_text, 10, 18)} "
        f"hypothesis: {boundary.query_text}"
    )


def _semantic_context_text(
    source: Stage44BSourceRecord,
    boundary: ExternalTaskBoundary,
    target_label: str,
) -> str:
    if source.task_name == "glue_rte":
        relation = (
            "the premise provides direct support for the hypothesis"
            if target_label == "entailment"
            else "the premise does not provide enough support for the hypothesis"
        )
        return (
            f"premise: {_evidence_snippet(boundary.evidence_text, 10, 18)} "
            f"hypothesis: {boundary.query_text} relation evidence: {relation}"
        )
    if source.task_name == "super_glue_cb":
        relation_by_label = {
            "entailment": "the premise supports the hypothesis",
            "contradiction": "the premise is incompatible with the hypothesis",
            "neutral": "the premise leaves the hypothesis undetermined",
        }
        return (
            f"premise: {_evidence_snippet(boundary.evidence_text, 10, 18)} "
            f"hypothesis: {boundary.query_text} relation evidence: {relation_by_label[target_label]}"
        )
    if source.task_name == "boolq":
        evidence = _boolq_relevant_evidence(boundary)
        relation = (
            "the passage supports a yes answer to the question"
            if target_label == "true"
            else "the passage supports a no answer to the question"
        )
        return (
            f"passage evidence: {_evidence_snippet(evidence, 12, 20)} "
            f"question: {boundary.query_text} answer evidence: {relation}"
        )
    return f"evidence: {boundary.evidence_text} query: {boundary.query_text}"


def _grounded_rule_text(source: Stage44BSourceRecord) -> str:
    if source.task_name == "glue_rte":
        return "answer entailment only when the hypothesis is supported by the premise; otherwise answer not_entailment"
    if source.task_name == "super_glue_cb":
        return "answer entailment for support, contradiction for incompatible evidence, and neutral when support is undetermined"
    if source.task_name == "boolq":
        return "answer true when the passage supports yes for the question; answer false when it does not"
    return TASK_RULES[source.task_name]


def stage44b1_answer_option_texts(task_name: str) -> tuple[str, ...]:
    if task_name == "glue_rte":
        return (
            "the hypothesis is entailed by the premise",
            "the hypothesis is not entailed by the premise",
        )
    if task_name == "super_glue_cb":
        return (
            "the hypothesis is entailed by the premise",
            "the hypothesis contradicts the premise",
            "the hypothesis is neutral or not determined by the premise",
        )
    if task_name == "boolq":
        return (
            "yes, the passage supports the question",
            "no, the passage does not support the question",
        )
    return EXTERNAL_ANSWER_OPTIONS[task_name]


def _source_to_sample(
    source: Stage44BSourceRecord,
    memory_text: str | None = None,
    *,
    boundary: ExternalTaskBoundary | None = None,
    grounding_mode: str = "stage44b_external_grounded_v1",
) -> LogicSample:
    if boundary is not None:
        text = _structured_external_text(source, boundary)
        memory_target = memory_text or _grounded_memory_text(source, boundary, grounding_mode=grounding_mode)
        rule_target = _grounded_rule_text(source)
    else:
        text = source.text
        memory_target = memory_text or _evidence_snippet(source.text)
        rule_target = TASK_RULES[source.task_name]
    return LogicSample(
        label="condition",
        text=text,
        token_ids=(1,),
        expected_pattern=source.external_label,
        variant=source.task_name,
        difficulty_level=10,
        logic_depth=2,
        leakage_family=grounding_mode,
        required_paths=("memory", "rule"),
        memory_target=memory_target,
        rule_target=rule_target,
        surface_group_id=source.surface_group_id,
        stress_profile="stage44b_external_recovery",
    )


def build_external_recovery_records(
    sources: list[Stage44BSourceRecord],
    *,
    grounding_mode: str = "stage44b_external_grounded_v1",
) -> list[ExternalRecoveryRecord]:
    by_task_label: dict[tuple[str, str], list[Stage44BSourceRecord]] = {}
    for source in sources:
        by_task_label.setdefault((source.task_name, source.external_label), []).append(source)
    result = []
    for index, source in enumerate(sources):
        labels = sorted(
            label for task_name, label in by_task_label if task_name == source.task_name
        )
        wrong_label = labels[(labels.index(source.external_label) + 1) % len(labels)]
        wrong_pool = by_task_label[(source.task_name, wrong_label)]
        wrong_source = _hard_negative_source(
            source,
            wrong_pool,
            grounding_mode=grounding_mode,
            fallback_index=index,
        )
        boundary = (
            recover_external_task_boundary(source)
            if grounding_mode in {
                "stage44b1_grounded_v2",
                "stage44b2_grounded_v3",
                "stage44b3_grounded_v4",
                "stage44b4_structured_semantic_v5",
            }
            else None
        )
        wrong_boundary = (
            recover_external_task_boundary(wrong_source)
            if grounding_mode in {
                "stage44b1_grounded_v2",
                "stage44b2_grounded_v3",
                "stage44b3_grounded_v4",
                "stage44b4_structured_semantic_v5",
            }
            else None
        )
        if grounding_mode == "stage44b4_structured_semantic_v5" and boundary is not None:
            wrong_memory = _grounded_memory_text(
                source,
                boundary,
                grounding_mode=grounding_mode,
                target_label=wrong_label,
            )
        else:
            wrong_memory = (
                _grounded_memory_text(wrong_source, wrong_boundary, grounding_mode=grounding_mode)
                if wrong_boundary is not None
                else _evidence_snippet(wrong_source.text)
            )
        sample = _source_to_sample(source, boundary=boundary, grounding_mode=grounding_mode)
        wrong_sample = _source_to_sample(
            source,
            memory_text=wrong_memory,
            boundary=boundary,
            grounding_mode=grounding_mode,
        )
        result.append(
            ExternalRecoveryRecord(
                source=source,
                sample=sample,
                wrong_sample=wrong_sample,
                wrong_option_id=source.answer_options.index(wrong_label),
            )
        )
    return result


def _hard_negative_source(
    source: Stage44BSourceRecord,
    candidates: list[Stage44BSourceRecord],
    *,
    grounding_mode: str,
    fallback_index: int,
) -> Stage44BSourceRecord:
    if grounding_mode != "stage44b3_grounded_v4":
        return sorted(candidates, key=lambda row: row.source_id)[fallback_index % len(candidates)]
    source_boundary = recover_external_task_boundary(source)
    source_query_terms = _normalized_terms(source_boundary.query_text)
    source_evidence_terms = _normalized_terms(source_boundary.evidence_text)
    scored = []
    for candidate in candidates:
        candidate_boundary = recover_external_task_boundary(candidate)
        query_overlap = len(source_query_terms & _normalized_terms(candidate_boundary.query_text))
        evidence_overlap = len(source_evidence_terms & _normalized_terms(candidate_boundary.evidence_text))
        if source.task_name == "boolq":
            score = (query_overlap * 10) + evidence_overlap
        else:
            score = (query_overlap * 6) + evidence_overlap
        scored.append((score, candidate.source_id, candidate))
    best_score = max(score for score, _source_id, _candidate in scored)
    if best_score <= 0:
        return sorted(candidates, key=lambda row: row.source_id)[fallback_index % len(candidates)]
    return sorted(scored, key=lambda row: (-row[0], row[1]))[0][2]


def encode_head_tail(backend: Qwen3Backend, text: str, max_length: int, tail_reserve: int = 96):
    ids = backend.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    truncated = len(ids) > max_length
    if truncated:
        if tail_reserve <= 0 or tail_reserve >= max_length:
            raise ValueError("invalid head-tail token budget")
        ids = ids[: max_length - tail_reserve] + ids[-tail_reserve:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=backend.device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, truncated


def _token_safe_context_text(backend: Qwen3Backend, text: str, max_tokens: int = 32) -> str:
    ids = backend.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    return backend.tokenizer.decode(ids[:head] + ids[-tail:], skip_special_tokens=True)


def _last_delta(output, paths: tuple[str, ...]) -> torch.Tensor:
    attention_mask = output.attention_mask
    tensors = []
    for trace in output.traces.values():
        current = None
        for path in paths:
            value = getattr(trace, f"{path}_delta_tensor")
            current = value if current is None else current + value
        if current is None:
            raise RuntimeError("no path delta selected")
        tensors.append(current)
    delta = torch.stack(tensors).sum(dim=0)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    return delta[torch.arange(delta.shape[0], device=delta.device), indices]


def _context_for_mode(context_encoder, sample, encoded, mode):
    kwargs: dict[str, Any] = {
        "ablation_config": CivilizationAblationConfig(use_state_path=False)
    }
    if mode == "adapter_disabled":
        kwargs["adapter_enabled"] = False
    elif mode == "zero_scale":
        kwargs["force_zero_scale"] = True
    elif mode == "no_memory_path":
        kwargs["ablation_config"] = CivilizationAblationConfig(
            use_memory_path=False, use_state_path=False
        )
    elif mode == "no_rule_path":
        kwargs["ablation_config"] = CivilizationAblationConfig(
            use_rule_path=False, use_state_path=False
        )
    context = context_encoder.build_context([sample], encoded["attention_mask"], **kwargs)
    return replace(
        context,
        memory_vectors=context.memory_vectors * EXTERNAL_CONTEXT_SCALE,
        rule_vectors=context.rule_vectors * EXTERNAL_CONTEXT_SCALE,
    )


def _context_for_samples(
    context_encoder,
    samples: list[LogicSample],
    attention_mask: torch.Tensor,
    mode: str,
    *,
    context_scale: float,
):
    kwargs: dict[str, Any] = {
        "ablation_config": CivilizationAblationConfig(use_state_path=False)
    }
    if mode == "adapter_disabled":
        kwargs["adapter_enabled"] = False
    elif mode == "zero_scale":
        kwargs["force_zero_scale"] = True
    elif mode == "no_memory_path":
        kwargs["ablation_config"] = CivilizationAblationConfig(
            use_memory_path=False, use_state_path=False
        )
    elif mode == "no_rule_path":
        kwargs["ablation_config"] = CivilizationAblationConfig(
            use_rule_path=False, use_state_path=False
        )
    context = context_encoder.build_context(samples, attention_mask, **kwargs)
    return replace(
        context,
        memory_vectors=context.memory_vectors * context_scale,
        rule_vectors=context.rule_vectors * context_scale,
    )


def _forward_record(*, backend, model, projector, context_encoder, record, mode, max_length):
    encoded, truncated = encode_head_tail(backend, record.sample.text, max_length)
    sample = record.wrong_sample if mode == "wrong_context" else record.sample
    sample = replace(
        sample,
        memory_target=_token_safe_context_text(backend, sample.memory_target),
        rule_target=_token_safe_context_text(backend, sample.rule_target),
    )
    context = _context_for_mode(context_encoder, sample, encoded, "full" if mode == "wrong_context" else mode)
    output = model(encoded, context)
    projected = projector(_last_delta(output, ("memory", "rule")))
    pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    return output, projected, pooled, truncated


def _forward_record_stage44b1(
    *, backend, model, task_projector, context_encoder, record, mode, max_length,
    context_scale: float = STAGE44B1_CONTEXT_SCALE,
):
    encoded, truncated = encode_head_tail(backend, record.sample.text, max_length)
    sample = record.wrong_sample if mode == "wrong_context" else record.sample
    sample = replace(
        sample,
        memory_target=_token_safe_context_text(backend, sample.memory_target, max_tokens=20),
        rule_target=_token_safe_context_text(backend, sample.rule_target, max_tokens=20),
    )
    context = _context_for_samples(
        context_encoder,
        [sample],
        encoded["attention_mask"],
        "full" if mode == "wrong_context" else mode,
        context_scale=context_scale,
    )
    output = model(encoded, context)
    projected = task_projector(_last_delta(output, ("memory", "rule")), record.source.task_name)
    pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    return output, projected, pooled, truncated


def _forward_full_wrong_pair_stage44b1(
    *, backend, model, task_projector, context_encoder, record, max_length,
    context_scale: float = STAGE44B1_CONTEXT_SCALE,
):
    encoded_single, truncated = encode_head_tail(backend, record.sample.text, max_length)
    encoded = {
        "input_ids": encoded_single["input_ids"].repeat(2, 1),
        "attention_mask": encoded_single["attention_mask"].repeat(2, 1),
    }
    samples = [
        replace(
            record.sample,
            memory_target=_token_safe_context_text(backend, record.sample.memory_target, max_tokens=20),
            rule_target=_token_safe_context_text(backend, record.sample.rule_target, max_tokens=20),
        ),
        replace(
            record.wrong_sample,
            memory_target=_token_safe_context_text(backend, record.wrong_sample.memory_target, max_tokens=20),
            rule_target=_token_safe_context_text(backend, record.wrong_sample.rule_target, max_tokens=20),
        ),
    ]
    context = _context_for_samples(
        context_encoder,
        samples,
        encoded["attention_mask"],
        "full",
        context_scale=context_scale,
    )
    output = model(encoded, context)
    projected = task_projector(_last_delta(output, ("memory", "rule")), record.source.task_name)
    pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
    return output, projected, pooled, truncated


def _set_trainable(model, projector) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    trainable = list(projector.parameters())
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            enabled = name.startswith("memory_") or name.startswith("rule_")
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
    return trainable


def _set_trainable_stage44b1(model, task_projector) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in task_projector.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    trainable = list(task_projector.parameters())
    for adapter in model.adapters.values():
        for name, parameter in adapter.named_parameters():
            enabled = name.startswith("memory_") or name.startswith("rule_")
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
    return trainable


def _external_sequence(records: list[ExternalRecoveryRecord], steps: int, seed: int):
    rng = random.Random(seed)
    ordered = sorted(
        records,
        key=lambda row: (
            row.source.source_id
            if hasattr(row, "source")
            else row.surface_group_id
        ),
    )
    result = []
    while len(result) < steps:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:steps]


def _local_rehearsal_loss(
    *, backend, model, projector, full_hidden_projector, heads, context_encoder, group
) -> torch.Tensor:
    options = torch.tensor(
        build_answer_option_vectors(backend, LOCAL_ANSWER_OPTIONS),
        dtype=torch.float32,
        device=backend.device,
    )
    result = _forward_local_group(
        backend=backend,
        model=model,
        projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=heads,
        context_encoder=context_encoder,
        group=group,
        option_vectors=options,
        max_length=128,
        mode="full",
    )
    path = _path_for_task(group.group_type)
    scores = _answer_scores(_path_projected(result, projector, path), options)
    return F.cross_entropy(scores / 0.05, result["targets"])


def _train_stage(
    *, backend, model, projector, full_hidden_projector, heads, records, local_groups,
    stage, steps, seed, run_seed, max_length, output_dir
):
    context_encoder = FrozenQwenContextEncoder(backend)
    trainable = _set_trainable(model, projector)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("Stage44B optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    option_cache = {
        task: torch.tensor(
            build_answer_option_vectors(backend, options),
            dtype=torch.float32,
            device=backend.device,
        )
        for task, options in EXTERNAL_ANSWER_OPTIONS.items()
    }
    rows = []
    sequence = _external_sequence(records, steps, seed)
    rehearsal = _external_sequence(local_groups, steps, seed + 19)
    for step, (record, local_group) in enumerate(zip(sequence, rehearsal, strict=True)):
        optimizer.zero_grad(set_to_none=True)
        output, projected, _pooled, truncated = _forward_record(
            backend=backend, model=model, projector=projector, context_encoder=context_encoder,
            record=record, mode="full", max_length=max_length,
        )
        _wrong_output, wrong_projected, _wrong_pooled, _ = _forward_record(
            backend=backend, model=model, projector=projector, context_encoder=context_encoder,
            record=record, mode="wrong_context", max_length=max_length,
        )
        options = option_cache[record.source.task_name]
        scores = _answer_scores(projected, options)
        wrong_scores = _answer_scores(wrong_projected, options)
        target = torch.tensor([record.source.correct_option_id], device=backend.device)
        wrong_target = torch.tensor([record.wrong_option_id], device=backend.device)
        answer_loss = F.cross_entropy(scores / 0.05, target)
        wrong_loss = F.cross_entropy(wrong_scores / 0.05, wrong_target)
        target_score = scores.gather(1, target[:, None]).squeeze(1)
        wrong_full_score = wrong_scores.gather(1, target[:, None]).squeeze(1)
        wrong_gap_loss = torch.relu(0.15 - (target_score - wrong_full_score)).mean()
        rehearsal_loss = _local_rehearsal_loss(
            backend=backend, model=model, projector=projector,
            full_hidden_projector=full_hidden_projector, heads=heads,
            context_encoder=context_encoder, group=local_group,
        )
        total = 2.0 * answer_loss + wrong_loss + wrong_gap_loss + rehearsal_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "stage": stage,
            "step": step + 1,
            "task_name": record.source.task_name,
            "total_loss": float(total.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()),
            "wrong_context_loss": float(wrong_loss.detach().cpu()),
            "wrong_gap_loss": float(wrong_gap_loss.detach().cpu()),
            "local_rehearsal_loss": float(rehearsal_loss.detach().cpu()),
            "accuracy": float((scores.argmax(dim=-1) == target).float().mean().cpu()),
            "truncated": truncated,
            "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in output.traces.values()),
        }
        rows.append(row)
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(
                f"stage44b stage={stage} step={step + 1}/{steps} task={record.source.task_name} "
                f"loss={row['total_loss']:.6f} acc={row['accuracy']:.3f}",
                flush=True,
            )
    checkpoint = output_dir / "checkpoints" / stage / f"stage44b_{stage}_seed_{run_seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {
                "stage": stage,
                "seed": run_seed,
                "stage_seed": seed,
                "adapter_variant": "path_specific_v2",
                "target_layers": [16, 24],
                "raw_full_hidden_residual_scale": 200.0,
                "external_context_scale": EXTERNAL_CONTEXT_SCALE,
                "source_checkpoint": "stage44a_full_hidden_alignment",
            },
        },
        checkpoint,
    )
    return rows, str(checkpoint)


def _train_stage44b1_stage(
    *, backend, model, task_projector, full_hidden_projector, heads, records, local_groups,
    stage, steps, seed, run_seed, max_length, output_dir,
    context_scale: float = STAGE44B1_CONTEXT_SCALE,
    stage_prefix: str = "stage44b1",
    training_mode: str = "stage44b1_task_specific_paired_context",
):
    context_encoder = FrozenQwenContextEncoder(backend)
    trainable = _set_trainable_stage44b1(model, task_projector)
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    if any(id(parameter) in qwen_ids for parameter in trainable):
        raise RuntimeError("Stage44B.1 optimizer contains Qwen parameters")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    option_cache = {
        task: torch.tensor(
            build_answer_option_vectors(backend, stage44b1_answer_option_texts(task)),
            dtype=torch.float32,
            device=backend.device,
        )
        for task in EXTERNAL_ANSWER_OPTIONS
    }
    rows = []
    sequence = _external_sequence(records, steps, seed)
    rehearsal = _external_sequence(local_groups, steps, seed + 19)
    for step, (record, local_group) in enumerate(zip(sequence, rehearsal, strict=True)):
        optimizer.zero_grad(set_to_none=True)
        output, projected_pair, _pooled_pair, truncated = _forward_full_wrong_pair_stage44b1(
            backend=backend,
            model=model,
            task_projector=task_projector,
            context_encoder=context_encoder,
            record=record,
            max_length=max_length,
            context_scale=context_scale,
        )
        options = option_cache[record.source.task_name]
        scores = _answer_scores(projected_pair, options)
        full_hidden_scores = _answer_scores(_pooled_pair, options)
        targets = torch.tensor(
            [record.source.correct_option_id, record.wrong_option_id],
            device=backend.device,
        )
        answer_loss = F.cross_entropy(scores / 0.05, targets)
        full_hidden_answer_loss = F.cross_entropy(full_hidden_scores / 0.05, targets)
        full_target_score = scores[0, record.source.correct_option_id]
        full_wrong_score = scores[0, record.wrong_option_id]
        wrong_context_target_score = scores[1, record.source.correct_option_id]
        wrong_context_option_score = scores[1, record.wrong_option_id]
        wrong_separation_loss = (
            torch.relu(0.25 - (full_target_score - full_wrong_score))
            + torch.relu(0.25 - (wrong_context_option_score - wrong_context_target_score))
        )
        paired_margin_loss = torch.relu(0.15 - (full_target_score - wrong_context_target_score))
        rehearsal_loss = _local_rehearsal_loss(
            backend=backend, model=model, projector=task_projector.projectors["glue_rte"],
            full_hidden_projector=full_hidden_projector, heads=heads,
            context_encoder=context_encoder, group=local_group,
        )
        total = (
            2.0 * answer_loss
            + 0.50 * full_hidden_answer_loss
            + wrong_separation_loss
            + paired_margin_loss
            + rehearsal_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "stage": stage,
            "step": step + 1,
            "task_name": record.source.task_name,
            "total_loss": float(total.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()),
            "full_hidden_answer_loss": float(full_hidden_answer_loss.detach().cpu()),
            "wrong_context_separation_loss": float(wrong_separation_loss.detach().cpu()),
            "paired_margin_loss": float(paired_margin_loss.detach().cpu()),
            "local_rehearsal_loss": float(rehearsal_loss.detach().cpu()),
            "full_accuracy": float((scores[:1].argmax(dim=-1) == targets[:1]).float().mean().cpu()),
            "wrong_context_accuracy": float((scores[1:].argmax(dim=-1) == targets[1:]).float().mean().cpu()),
            "truncated": truncated,
            "hidden_norm_ratio": max(trace.hidden_norm_ratio for trace in output.traces.values()),
            "paired_batch": True,
            "task_specific_projector": record.source.task_name,
        }
        rows.append(row)
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(
                f"{stage_prefix} stage={stage} step={step + 1}/{steps} task={record.source.task_name} "
                f"loss={row['total_loss']:.6f} full_acc={row['full_accuracy']:.3f}",
                flush=True,
            )
    checkpoint = output_dir / "checkpoints" / stage / f"{stage_prefix}_{stage}_seed_{run_seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": model.adapters.state_dict(),
            "task_projector_state_dict": task_projector.state_dict(),
            "full_hidden_projector_state_dict": full_hidden_projector.state_dict(),
            "training_metadata": {
                "stage": stage,
                "seed": run_seed,
                "stage_seed": seed,
                "adapter_variant": "path_specific_v2",
                "target_layers": [16, 24],
                "raw_full_hidden_residual_scale": 200.0,
                "external_context_scale": context_scale,
                "source_checkpoint": "stage44a_full_hidden_alignment",
                "training_mode": training_mode,
            },
        },
        checkpoint,
    )
    return rows, str(checkpoint)


def _centroids(vectors: list[tuple[int, torch.Tensor]], option_count: int):
    result = {}
    for option_id in range(option_count):
        selected = [vector for target, vector in vectors if target == option_id]
        result[option_id] = torch.stack(selected).mean(dim=0)
    return result


def _evaluate_external(*, backend, model, projector, train_records, heldout_records, max_length):
    context_encoder = FrozenQwenContextEncoder(backend)
    rows = []
    for task_name in EXTERNAL_ANSWER_OPTIONS:
        task_train = [row for row in train_records if row.source.task_name == task_name]
        task_test = [row for row in heldout_records if row.source.task_name == task_name]
        train_vectors = []
        with torch.no_grad():
            for record in task_train:
                _output, _projected, pooled, _ = _forward_record(
                    backend=backend, model=model, projector=projector,
                    context_encoder=context_encoder, record=record, mode="full", max_length=max_length,
                )
                train_vectors.append((record.source.correct_option_id, pooled.squeeze(0).cpu()))
        fixed = _centroids(train_vectors, len(EXTERNAL_ANSWER_OPTIONS[task_name]))
        option_vectors = torch.tensor(
            build_answer_option_vectors(backend, EXTERNAL_ANSWER_OPTIONS[task_name]),
            dtype=torch.float32,
            device=backend.device,
        )
        for mode in EVAL_MODES:
            correct = 0
            fixed_correct = 0
            hidden_ratios = []
            with torch.no_grad():
                for record in task_test:
                    output, projected, pooled, truncated = _forward_record(
                        backend=backend, model=model, projector=projector,
                        context_encoder=context_encoder, record=record, mode=mode, max_length=max_length,
                    )
                    target = record.source.correct_option_id
                    predicted = int(_answer_scores(projected, option_vectors).argmax(dim=-1).item())
                    fixed_scores = torch.tensor(
                        [-torch.linalg.vector_norm(pooled.squeeze(0).cpu() - fixed[index]) for index in range(len(fixed))]
                    )
                    fixed_predicted = int(fixed_scores.argmax().item())
                    correct += predicted == target
                    fixed_correct += fixed_predicted == target
                    hidden_ratios.append(max(trace.hidden_norm_ratio for trace in output.traces.values()))
            rows.append(
                {
                    "task_name": task_name,
                    "mode": mode,
                    "count": len(task_test),
                    "projected_accuracy": correct / len(task_test),
                    "fixed_centroid_accuracy": fixed_correct / len(task_test),
                    "hidden_norm_ratio": max(hidden_ratios, default=1.0),
                }
            )
    return rows


def _evaluate_external_stage44b1(
    *, backend, model, task_projector, train_records, heldout_records, max_length,
    context_scale: float = STAGE44B1_CONTEXT_SCALE,
):
    context_encoder = FrozenQwenContextEncoder(backend)
    rows = []
    boundary_rows = []
    for task_name in EXTERNAL_ANSWER_OPTIONS:
        task_train = [row for row in train_records if row.source.task_name == task_name]
        task_test = [row for row in heldout_records if row.source.task_name == task_name]
        train_vectors = []
        with torch.no_grad():
            for record in task_train:
                _output, _projected, pooled, _ = _forward_record_stage44b1(
                    backend=backend,
                    model=model,
                    task_projector=task_projector,
                    context_encoder=context_encoder,
                    record=record,
                    mode="full",
                    max_length=max_length,
                    context_scale=context_scale,
                )
                train_vectors.append((record.source.correct_option_id, pooled.squeeze(0).cpu()))
        fixed = _centroids(train_vectors, len(EXTERNAL_ANSWER_OPTIONS[task_name]))
        option_vectors = torch.tensor(
            build_answer_option_vectors(backend, stage44b1_answer_option_texts(task_name)),
            dtype=torch.float32,
            device=backend.device,
        )
        for record in task_test:
            boundary = recover_external_task_boundary(record.source)
            boundary_rows.append(
                {
                    "task_name": task_name,
                    "source_id": record.source.source_id,
                    "boundary_source": boundary.boundary_source,
                    "boundary_ok": boundary.boundary_ok,
                    "evidence_token_count": len(backend.tokenizer(boundary.evidence_text, add_special_tokens=False)["input_ids"]),
                    "query_token_count": len(backend.tokenizer(boundary.query_text, add_special_tokens=False)["input_ids"]),
                }
            )
        for mode in EVAL_MODES:
            correct = 0
            fixed_correct = 0
            hidden_ratios = []
            with torch.no_grad():
                for record in task_test:
                    output, projected, pooled, truncated = _forward_record_stage44b1(
                        backend=backend,
                        model=model,
                        task_projector=task_projector,
                        context_encoder=context_encoder,
                        record=record,
                        mode=mode,
                        max_length=max_length,
                        context_scale=context_scale,
                    )
                    target = record.source.correct_option_id
                    predicted = int(_answer_scores(projected, option_vectors).argmax(dim=-1).item())
                    fixed_scores = torch.tensor(
                        [-torch.linalg.vector_norm(pooled.squeeze(0).cpu() - fixed[index]) for index in range(len(fixed))]
                    )
                    fixed_predicted = int(fixed_scores.argmax().item())
                    correct += predicted == target
                    fixed_correct += fixed_predicted == target
                    hidden_ratios.append(max(trace.hidden_norm_ratio for trace in output.traces.values()))
            rows.append(
                {
                    "task_name": task_name,
                    "mode": mode,
                    "count": len(task_test),
                    "projected_accuracy": correct / len(task_test),
                    "fixed_centroid_accuracy": fixed_correct / len(task_test),
                    "hidden_norm_ratio": max(hidden_ratios, default=1.0),
                    "readout": "task_specific_projected_delta",
                    "paired_batch_training": True,
                }
            )
    return rows, boundary_rows


def _mean(rows, field, **filters):
    values = [float(row[field]) for row in rows if all(row.get(k) == v for k, v in filters.items())]
    return sum(values) / len(values) if values else 0.0


def _stage44a_checkpoint(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}" / "checkpoints" / "full_hidden_alignment" / f"stage44a_full_hidden_alignment_seed_{seed}.pt"


def run_qwen3_stage44b_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    stage44a_root: str | Path = DEFAULT_STAGE44A_ROOT,
    seed: int = 202,
    train_per_label: int = 12,
    heldout_per_label: int = 12,
    max_length: int = 384,
    preferred_device: str = "cuda",
    task_steps: int = 40,
    combined_steps: int = 60,
    strict_stage_gates: bool = True,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    checkpoint = _stage44a_checkpoint(Path(stage44a_root), seed)
    model, projector, full_hidden_projector, payload = _load_stage41_checkpoint(backend, checkpoint)
    metadata = payload.get("training_metadata", {})
    if metadata.get("seed") != seed or metadata.get("stage") != "full_hidden_alignment":
        raise ValueError("Stage44B source checkpoint metadata mismatch")

    records_by_task, _manifest = load_external_task_records(
        cache_dir=cache_dir, task_names=tuple(EXTERNAL_ANSWER_OPTIONS), allow_download=False
    )
    projected_sources = project_external_source_records(records_by_task)
    train_sources = []
    heldout_sources = []
    for task_name, sources in projected_sources.items():
        train, heldout = deterministic_external_split(
            sources, seed=seed, train_per_label=train_per_label, heldout_per_label=heldout_per_label
        )
        train_sources.extend(train)
        heldout_sources.extend(heldout)
    train_records = build_external_recovery_records(train_sources)
    heldout_records = build_external_recovery_records(heldout_sources)

    local_records = build_local_semireal_task_records(
        24, seed=seed, context_grounding_mode="stage44a_path_grounded_v2"
    )
    local_datasets, _ = records_to_logic_datasets(
        local_records, max_seq_len=128, context_grounding_mode="stage44a_path_grounded_v2"
    )
    local_train, local_heldout = split_stage44a_groups(
        build_stage44a_groups(local_datasets), 16, seed
    )
    local_before_rows = _evaluate_local_groups(
        backend=backend, model=model, projector=projector,
        full_hidden_projector=full_hidden_projector,
        heads=__import__(
            "civilization.engine.stages.adapter_training",
            fromlist=["AdapterDiagnosticHeads"],
        ).AdapterDiagnosticHeads().to(backend.device),
        groups=local_heldout, max_length=128,
        modes=("full", "no_memory_path", "no_rule_path", "no_state_path"),
    )
    local_before = _mean(local_before_rows, "projected_accuracy", mode="full")
    heads = __import__(
        "civilization.engine.stages.adapter_training",
        fromlist=["AdapterDiagnosticHeads"],
    ).AdapterDiagnosticHeads().to(backend.device)

    loss_rows = []
    stage_rows = []
    task_order = ("glue_rte", "super_glue_cb", "boolq")
    if evaluation_only:
        combined_checkpoint = output / "checkpoints" / "combined" / f"stage44b_combined_seed_{seed}.pt"
        model, projector, full_hidden_projector, resumed_payload = _load_stage41_checkpoint(
            backend, combined_checkpoint
        )
        if resumed_payload.get("training_metadata", {}).get("source_checkpoint") != "stage44a_full_hidden_alignment":
            raise ValueError("Stage44B evaluation checkpoint provenance mismatch")
        stage_rows.append(
            {"stage": "combined", "checkpoint_path": str(combined_checkpoint), "steps": 0, "evaluation_only": True}
        )
    else:
        for stage_index, task_name in enumerate(task_order):
            pool = [record for record in train_records if record.source.task_name == task_name]
            rows, checkpoint_path = _train_stage(
                backend=backend, model=model, projector=projector,
                full_hidden_projector=full_hidden_projector, heads=heads,
                records=pool, local_groups=local_train, stage=task_name,
                steps=task_steps, seed=seed + stage_index, run_seed=seed,
                max_length=max_length, output_dir=output,
            )
            loss_rows.extend(rows)
            stage_rows.append({"stage": task_name, "checkpoint_path": checkpoint_path, "steps": task_steps})
        rows, checkpoint_path = _train_stage(
            backend=backend, model=model, projector=projector,
            full_hidden_projector=full_hidden_projector, heads=heads,
            records=train_records, local_groups=local_train, stage="combined",
            steps=combined_steps, seed=seed + 3, run_seed=seed,
            max_length=max_length, output_dir=output,
        )
        loss_rows.extend(rows)
        stage_rows.append({"stage": "combined", "checkpoint_path": checkpoint_path, "steps": combined_steps})

    external_rows = _evaluate_external(
        backend=backend, model=model, projector=projector,
        train_records=train_records, heldout_records=heldout_records, max_length=max_length,
    )
    local_after_rows = _evaluate_local_groups(
        backend=backend, model=model, projector=projector,
        full_hidden_projector=full_hidden_projector, heads=heads,
        groups=local_heldout, max_length=128,
        modes=("full", "no_memory_path", "no_rule_path", "no_state_path"),
    )
    local_after = _mean(local_after_rows, "projected_accuracy", mode="full")
    full_by_task = {
        task: _mean(external_rows, "projected_accuracy", task_name=task, mode="full")
        for task in task_order
    }
    disabled_by_task = {
        task: _mean(external_rows, "projected_accuracy", task_name=task, mode="adapter_disabled")
        for task in task_order
    }
    improvements = {task: full_by_task[task] - disabled_by_task[task] for task in task_order}
    wrong_drop = statistics.mean(
        full_by_task[task]
        - _mean(external_rows, "projected_accuracy", task_name=task, mode="wrong_context")
        for task in task_order
    )
    fixed_average = statistics.mean(
        _mean(external_rows, "fixed_centroid_accuracy", task_name=task, mode="full")
        for task in task_order
    )
    hidden_norm = max(float(row["hidden_norm_ratio"]) for row in external_rows)
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    gates = {
        "qwen_frozen": sum(p.numel() for p in backend.model.parameters() if p.requires_grad) == 0,
        "qwen_gradients": qwen_gradients == 0,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "local_retention": local_after >= 0.95 and local_after >= local_before - 0.05,
        "external_evidence": sum(value >= 0.05 for value in improvements.values()) >= 2,
        "wrong_context_control": wrong_drop >= 0.10,
        "fixed_centroid": fixed_average >= 0.50,
        "hidden_norm": hidden_norm <= 2.0,
    }
    if strict_stage_gates:
        for name, passed in gates.items():
            if not passed:
                failures.append({"failed_stage": "final_evaluation", "failed_gate": name})

    _write_csv(output / "loss_curves.csv", loss_rows)
    _write_csv(output / "training_stages.csv", stage_rows)
    _write_csv(output / "external_task_metrics.csv", external_rows)
    _write_csv(output / "local_retention_metrics.csv", local_after_rows)
    _json_dump(output / "failure_cases.json", failures)
    summary = {
        "stage": "44B_seed202_external_recovery_training",
        "seed": seed,
        "source_checkpoint": str(checkpoint),
        "full_accuracy_by_task": full_by_task,
        "disabled_accuracy_by_task": disabled_by_task,
        "improvement_by_task": improvements,
        "wrong_context_drop": wrong_drop,
        "fixed_centroid_average": fixed_average,
        "local_accuracy_before": local_before,
        "local_accuracy_after": local_after,
        "hidden_norm_ratio": hidden_norm,
        "qwen_trainable_parameters": sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
        "qwen_gradients": qwen_gradients,
        "stage_gates": gates,
        "passes_stage_gate": not failures and all(gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "resource_usage": {"rss": psutil.Process().memory_info().rss},
    }
    summary["allows_stage44b_multiseed"] = summary["passes_stage_gate"]
    _json_dump(output / "summary.json", summary)
    return summary


def run_qwen3_stage44b1_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B1_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    stage44a_root: str | Path = DEFAULT_STAGE44A_ROOT,
    seed: int = 202,
    train_per_label: int = 12,
    heldout_per_label: int = 12,
    max_length: int = 384,
    preferred_device: str = "cuda",
    task_steps: int = 50,
    combined_steps: int = 80,
    strict_stage_gates: bool = True,
    grounding_mode: str = "stage44b1_grounded_v2",
    context_scale: float = STAGE44B1_CONTEXT_SCALE,
    stage_prefix: str = "stage44b1",
    training_mode: str = "stage44b1_task_specific_paired_context",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    backend = Qwen3Backend(model_path, preferred_device=preferred_device)
    checkpoint = _stage44a_checkpoint(Path(stage44a_root), seed)
    model, source_projector, full_hidden_projector, payload = _load_stage41_checkpoint(backend, checkpoint)
    metadata = payload.get("training_metadata", {})
    if metadata.get("seed") != seed or metadata.get("stage") != "full_hidden_alignment":
        raise ValueError("Stage44B.1 source checkpoint metadata mismatch")

    task_projector = TaskSpecificPathReadoutProjector(tuple(EXTERNAL_ANSWER_OPTIONS)).to(backend.device)
    for projector in task_projector.projectors.values():
        projector.load_state_dict(source_projector.state_dict())

    records_by_task, _manifest = load_external_task_records(
        cache_dir=cache_dir, task_names=tuple(EXTERNAL_ANSWER_OPTIONS), allow_download=False
    )
    projected_sources = project_external_source_records(records_by_task)
    boundary_rows = []
    for task_name, sources in projected_sources.items():
        for source in sources:
            boundary = recover_external_task_boundary(source)
            boundary_rows.append(
                {
                    "task_name": task_name,
                    "source_id": source.source_id,
                    "boundary_source": boundary.boundary_source,
                    "boundary_ok": boundary.boundary_ok,
                    "evidence_hash": _stable_text_hash(boundary.evidence_text),
                    "query_hash": _stable_text_hash(boundary.query_text),
                }
            )
            if not boundary.boundary_ok:
                failures.append(
                    {
                        "failed_stage": "boundary_recovery",
                        "failed_gate": "boundary_ok",
                        "task_name": task_name,
                        "source_id": source.source_id,
                    }
                )
    train_sources = []
    heldout_sources = []
    for task_name, sources in projected_sources.items():
        train, heldout = deterministic_external_split(
            sources, seed=seed, train_per_label=train_per_label, heldout_per_label=heldout_per_label
        )
        train_sources.extend(train)
        heldout_sources.extend(heldout)
    train_records = build_external_recovery_records(train_sources, grounding_mode=grounding_mode)
    heldout_records = build_external_recovery_records(heldout_sources, grounding_mode=grounding_mode)

    local_records = build_local_semireal_task_records(
        24, seed=seed, context_grounding_mode="stage44a_path_grounded_v2"
    )
    local_datasets, _ = records_to_logic_datasets(
        local_records, max_seq_len=128, context_grounding_mode="stage44a_path_grounded_v2"
    )
    local_train, local_heldout = split_stage44a_groups(
        build_stage44a_groups(local_datasets), 16, seed
    )
    heads = __import__(
        "civilization.engine.stages.adapter_training",
        fromlist=["AdapterDiagnosticHeads"],
    ).AdapterDiagnosticHeads().to(backend.device)
    local_before_rows = _evaluate_local_groups(
        backend=backend, model=model, projector=task_projector.projectors["glue_rte"],
        full_hidden_projector=full_hidden_projector, heads=heads,
        groups=local_heldout, max_length=128,
        modes=("full", "no_memory_path", "no_rule_path", "no_state_path"),
    )
    local_before = _mean(local_before_rows, "projected_accuracy", mode="full")

    loss_rows = []
    stage_rows = []
    task_order = ("glue_rte", "super_glue_cb", "boolq")
    if not failures or not strict_stage_gates:
        for stage_index, task_name in enumerate(task_order):
            pool = [record for record in train_records if record.source.task_name == task_name]
            rows, checkpoint_path = _train_stage44b1_stage(
                backend=backend, model=model, task_projector=task_projector,
                full_hidden_projector=full_hidden_projector, heads=heads,
                records=pool, local_groups=local_train, stage=task_name,
                steps=task_steps, seed=seed + stage_index, run_seed=seed,
                max_length=max_length, output_dir=output,
                context_scale=context_scale,
                stage_prefix=stage_prefix,
                training_mode=training_mode,
            )
            loss_rows.extend(rows)
            stage_rows.append({"stage": task_name, "checkpoint_path": checkpoint_path, "steps": task_steps})
        rows, checkpoint_path = _train_stage44b1_stage(
            backend=backend, model=model, task_projector=task_projector,
            full_hidden_projector=full_hidden_projector, heads=heads,
            records=train_records, local_groups=local_train, stage="combined",
            steps=combined_steps, seed=seed + 3, run_seed=seed,
            max_length=max_length, output_dir=output,
            context_scale=context_scale,
            stage_prefix=stage_prefix,
            training_mode=training_mode,
        )
        loss_rows.extend(rows)
        stage_rows.append({"stage": "combined", "checkpoint_path": checkpoint_path, "steps": combined_steps})

    external_rows, eval_boundary_rows = _evaluate_external_stage44b1(
        backend=backend, model=model, task_projector=task_projector,
        train_records=train_records, heldout_records=heldout_records, max_length=max_length,
        context_scale=context_scale,
    )
    boundary_rows.extend(eval_boundary_rows)
    local_after_rows = _evaluate_local_groups(
        backend=backend, model=model, projector=task_projector.projectors["glue_rte"],
        full_hidden_projector=full_hidden_projector, heads=heads,
        groups=local_heldout, max_length=128,
        modes=("full", "no_memory_path", "no_rule_path", "no_state_path"),
    )
    local_after = _mean(local_after_rows, "projected_accuracy", mode="full")
    full_by_task = {
        task: _mean(external_rows, "projected_accuracy", task_name=task, mode="full")
        for task in task_order
    }
    disabled_by_task = {
        task: _mean(external_rows, "projected_accuracy", task_name=task, mode="adapter_disabled")
        for task in task_order
    }
    improvements = {task: full_by_task[task] - disabled_by_task[task] for task in task_order}
    wrong_drop = statistics.mean(
        full_by_task[task]
        - _mean(external_rows, "projected_accuracy", task_name=task, mode="wrong_context")
        for task in task_order
    )
    fixed_average = statistics.mean(
        _mean(external_rows, "fixed_centroid_accuracy", task_name=task, mode="full")
        for task in task_order
    )
    hidden_norm = max(float(row["hidden_norm_ratio"]) for row in external_rows)
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    gates = {
        "qwen_frozen": sum(p.numel() for p in backend.model.parameters() if p.requires_grad) == 0,
        "qwen_gradients": qwen_gradients == 0,
        "weights_unchanged": backend.verify_weights_unchanged(),
        "boundary_recovery": all(row["boundary_ok"] for row in boundary_rows),
        "local_retention": local_after >= 0.95 and local_after >= local_before - 0.05,
        "external_evidence": sum(value >= 0.05 for value in improvements.values()) >= 2,
        "wrong_context_control": wrong_drop >= 0.10,
        "fixed_centroid": fixed_average >= 0.50,
        "hidden_norm": hidden_norm <= 2.0,
    }
    if strict_stage_gates:
        for name, passed in gates.items():
            if not passed:
                failures.append({"failed_stage": "stage44b1_final_evaluation", "failed_gate": name})

    _write_csv(output / "loss_curves.csv", loss_rows)
    _write_csv(output / "training_stages.csv", stage_rows)
    _write_csv(output / "external_task_metrics.csv", external_rows)
    _write_csv(output / "local_retention_metrics.csv", local_after_rows)
    _write_csv(output / "boundary_audit.csv", boundary_rows)
    _json_dump(output / "failure_cases.json", failures)
    summary = {
        "stage": training_mode,
        "seed": seed,
        "source_checkpoint": str(checkpoint),
        "full_accuracy_by_task": full_by_task,
        "disabled_accuracy_by_task": disabled_by_task,
        "improvement_by_task": improvements,
        "wrong_context_drop": wrong_drop,
        "fixed_centroid_average": fixed_average,
        "local_accuracy_before": local_before,
        "local_accuracy_after": local_after,
        "hidden_norm_ratio": hidden_norm,
        "qwen_trainable_parameters": sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
        "qwen_gradients": qwen_gradients,
        "stage_gates": gates,
        "passes_stage_gate": not failures and all(gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "resource_usage": {"rss": psutil.Process().memory_info().rss},
    }
    summary["allows_stage44b_multiseed"] = summary["passes_stage_gate"]
    _json_dump(output / "summary.json", summary)
    return summary


def run_qwen3_stage44b2_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B2_OUTPUT_DIR,
    seed: int = 202,
    preferred_device: str = "cuda",
    task_steps: int = 70,
    combined_steps: int = 110,
    **kwargs,
) -> dict[str, Any]:
    return run_qwen3_stage44b1_external_recovery_training(
        output_dir=output_dir,
        seed=seed,
        preferred_device=preferred_device,
        task_steps=task_steps,
        combined_steps=combined_steps,
        grounding_mode="stage44b2_grounded_v3",
        context_scale=STAGE44B2_CONTEXT_SCALE,
        stage_prefix="stage44b2",
        training_mode="stage44b2_rte_boolq_recovery",
        **kwargs,
    )


def run_qwen3_stage44b3_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B3_OUTPUT_DIR,
    seed: int = 202,
    preferred_device: str = "cuda",
    task_steps: int = 100,
    combined_steps: int = 160,
    **kwargs,
) -> dict[str, Any]:
    return run_qwen3_stage44b1_external_recovery_training(
        output_dir=output_dir,
        seed=seed,
        preferred_device=preferred_device,
        task_steps=task_steps,
        combined_steps=combined_steps,
        grounding_mode="stage44b3_grounded_v4",
        context_scale=STAGE44B3_CONTEXT_SCALE,
        stage_prefix="stage44b3",
        training_mode="stage44b3_hard_negative_public_recovery",
        **kwargs,
    )


def run_qwen3_stage44b4_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B4_OUTPUT_DIR,
    seed: int = 202,
    preferred_device: str = "cuda",
    task_steps: int = 100,
    combined_steps: int = 160,
    **kwargs,
) -> dict[str, Any]:
    return run_qwen3_stage44b1_external_recovery_training(
        output_dir=output_dir,
        seed=seed,
        preferred_device=preferred_device,
        task_steps=task_steps,
        combined_steps=combined_steps,
        grounding_mode="stage44b4_structured_semantic_v5",
        context_scale=STAGE44B4_CONTEXT_SCALE,
        stage_prefix="stage44b4",
        training_mode="stage44b4_structured_semantic_wrong_context",
        **kwargs,
    )


def run_qwen3_stage44b5_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B5_OUTPUT_DIR,
    source_cache_dir: str | Path = DEFAULT_CACHE_DIR,
    structured_cache_dir: str | Path = DEFAULT_STRUCTURED_CACHE_DIR,
    seed: int = 202,
    preferred_device: str = "cuda",
    task_steps: int = 100,
    combined_steps: int = 160,
    **kwargs,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = solidify_structured_external_cache(
        source_cache_dir=source_cache_dir,
        output_cache_dir=structured_cache_dir,
        task_names=tuple(EXTERNAL_ANSWER_OPTIONS),
    )
    _json_dump(output / "structured_cache_manifest.json", manifest)
    summary = run_qwen3_stage44b4_external_recovery_training(
        output_dir=output,
        cache_dir=structured_cache_dir,
        seed=seed,
        preferred_device=preferred_device,
        task_steps=task_steps,
        combined_steps=combined_steps,
        **kwargs,
    )
    summary["stage"] = "stage44b5_structured_cache_external_recovery"
    summary["structured_cache_dir"] = str(structured_cache_dir)
    summary["structured_cache_manifest"] = manifest
    _json_dump(output / "summary.json", summary)
    return summary


def _dataset_fields_cache_manifest(cache_dir: str | Path) -> list[dict[str, Any]]:
    records_by_task, manifest = load_external_task_records(
        cache_dir=cache_dir,
        task_names=tuple(EXTERNAL_ANSWER_OPTIONS),
        allow_download=False,
        context_grounding_mode="real_task_v1",
    )
    rows = []
    for row in manifest:
        task_records = records_by_task[row["task_name"]]
        structured_ok = all(
            (
                record.structure_source == "dataset_fields"
                and (
                    bool(record.premise and record.hypothesis)
                    if record.task_type in {"glue_rte", "super_glue_cb"}
                    else bool(record.passage and record.question)
                )
            )
            for record in task_records
        )
        rows.append(
            {
                **row,
                "stage": "stage44b6_dataset_fields_cache",
                "structure_sources": sorted({record.structure_source for record in task_records}),
                "dataset_fields_structured_ok": structured_ok,
            }
        )
    return rows


def run_qwen3_stage44b6_external_recovery_training(
    *,
    output_dir: str | Path = DEFAULT_STAGE44B6_OUTPUT_DIR,
    dataset_fields_cache_dir: str | Path = DEFAULT_DATASET_FIELDS_CACHE_DIR,
    seed: int = 202,
    preferred_device: str = "cuda",
    task_steps: int = 100,
    combined_steps: int = 160,
    allow_dataset_download: bool = False,
    **kwargs,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # This deliberately targets a separate cache dir. Missing cache fails unless
    # the caller explicitly enables dataset download.
    load_external_task_records(
        cache_dir=dataset_fields_cache_dir,
        task_names=tuple(EXTERNAL_ANSWER_OPTIONS),
        allow_download=allow_dataset_download,
        context_grounding_mode="real_task_v1",
    )
    manifest = _dataset_fields_cache_manifest(dataset_fields_cache_dir)
    _json_dump(output / "dataset_fields_cache_manifest.json", manifest)
    if not all(row["dataset_fields_structured_ok"] for row in manifest):
        failures = [
            {
                "failed_stage": "dataset_fields_cache_audit",
                "failed_gate": "structure_source_dataset_fields",
                "task_name": row["task_name"],
                "structure_sources": row["structure_sources"],
            }
            for row in manifest
            if not row["dataset_fields_structured_ok"]
        ]
        _json_dump(output / "failure_cases.json", failures)
        raise RuntimeError("Stage44B.6 dataset-fields cache audit failed")
    summary = run_qwen3_stage44b4_external_recovery_training(
        output_dir=output,
        cache_dir=dataset_fields_cache_dir,
        seed=seed,
        preferred_device=preferred_device,
        task_steps=task_steps,
        combined_steps=combined_steps,
        **kwargs,
    )
    summary["stage"] = "stage44b6_dataset_fields_external_recovery"
    summary["dataset_fields_cache_dir"] = str(dataset_fields_cache_dir)
    summary["dataset_fields_cache_manifest"] = manifest
    _json_dump(output / "summary.json", summary)
    return summary
