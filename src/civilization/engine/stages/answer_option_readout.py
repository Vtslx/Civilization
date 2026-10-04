from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from civilization.research.torch_line.analysis.dataset import LogicSample

from ..backend import Qwen3Backend


EXTERNAL_ANSWER_OPTIONS: dict[str, tuple[str, ...]] = {
    "glue_rte": ("entailment", "not_entailment"),
    "super_glue_cb": ("entailment", "contradiction", "neutral"),
    "boolq": ("true", "false"),
}


def answer_options_for_samples(samples: list[LogicSample]) -> tuple[str, ...]:
    if not samples:
        raise ValueError("answer option readout requires samples")
    variants = {sample.variant for sample in samples}
    if len(variants) != 1:
        raise ValueError("answer option readout requires one task variant")
    variant = next(iter(variants))
    if variant in EXTERNAL_ANSWER_OPTIONS:
        return EXTERNAL_ANSWER_OPTIONS[variant]
    options = tuple(sorted({sample.expected_pattern for sample in samples if sample.expected_pattern}))
    if not options:
        raise ValueError(f"no answer options for variant {variant}")
    return options


def _masked_mean_embeddings(
    backend: Qwen3Backend,
    texts: list[str],
    max_length: int = 32,
) -> torch.Tensor:
    encoded, truncations = backend.encode(texts, max_length=max_length)
    if truncations:
        raise ValueError("answer option text was truncated")
    input_ids = encoded["input_ids"].to(backend.device)
    attention_mask = encoded["attention_mask"].to(backend.device)
    with torch.no_grad():
        embeddings = backend.model.model.embed_tokens(input_ids)
        token_mask = attention_mask.to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
    return pooled.float().cpu()


def build_answer_option_vectors(
    backend: Qwen3Backend,
    options: tuple[str, ...],
) -> np.ndarray:
    if len(set(options)) != len(options):
        raise ValueError("answer options must be unique")
    vectors = _masked_mean_embeddings(backend, list(options))
    return vectors.numpy()


@dataclass(frozen=True)
class AnswerOptionMetrics:
    accuracy: float
    macro_accuracy: float
    mean_margin: float
    rows: list[dict[str, Any]]


def score_answer_options(
    sample_vectors: np.ndarray,
    samples: list[LogicSample],
    option_vectors: np.ndarray,
    options: tuple[str, ...],
) -> AnswerOptionMetrics:
    if sample_vectors.ndim != 2 or option_vectors.ndim != 2:
        raise ValueError("sample_vectors and option_vectors must be rank-2")
    if sample_vectors.shape[1] != option_vectors.shape[1]:
        raise ValueError("sample and option vectors must have the same hidden size")
    if len(samples) != sample_vectors.shape[0]:
        raise ValueError("sample count does not match vectors")
    sample_norm = sample_vectors / np.clip(np.linalg.norm(sample_vectors, axis=1, keepdims=True), 1e-8, None)
    option_norm = option_vectors / np.clip(np.linalg.norm(option_vectors, axis=1, keepdims=True), 1e-8, None)
    scores = sample_norm @ option_norm.T
    if not np.isfinite(scores).all():
        raise ValueError("answer option scores contain NaN/Inf")
    rows: list[dict[str, Any]] = []
    correct_flags: list[bool] = []
    margins: list[float] = []
    for index, sample in enumerate(samples):
        expected = sample.expected_pattern
        if expected not in options:
            raise ValueError(f"expected answer {expected!r} not in options")
        expected_index = options.index(expected)
        sorted_indices = np.argsort(scores[index])[::-1]
        predicted_index = int(sorted_indices[0])
        runner_up = int(sorted_indices[1]) if len(sorted_indices) > 1 else predicted_index
        margin = float(scores[index, expected_index] - scores[index, runner_up])
        correct = predicted_index == expected_index
        correct_flags.append(correct)
        margins.append(margin)
        rows.append(
            {
                "sample_id": sample.surface_group_id,
                "task_type": sample.variant,
                "true_answer": expected,
                "predicted_answer": options[predicted_index],
                "correct": correct,
                "margin": margin,
                "score": float(scores[index, predicted_index]),
            }
        )
    per_answer = []
    for option in options:
        values = [
            row["correct"]
            for row in rows
            if row["true_answer"] == option
        ]
        if values:
            per_answer.append(sum(values) / len(values))
    return AnswerOptionMetrics(
        accuracy=sum(correct_flags) / len(correct_flags) if correct_flags else 0.0,
        macro_accuracy=float(np.mean(per_answer)) if per_answer else 0.0,
        mean_margin=float(np.mean(margins)) if margins else 0.0,
        rows=rows,
    )
