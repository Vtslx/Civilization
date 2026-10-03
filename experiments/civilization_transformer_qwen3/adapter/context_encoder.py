from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS, LogicSample
from experiments.civilization_transformer_torch.analysis.path_dependency_training import dependency_state_for_label
from experiments.civilization_transformer_torch.memory import MemoryItem
from experiments.civilization_transformer_torch.rules import RuleItem

from ..backend import Qwen3Backend
from .civilization_adapter import CivilizationAdapterContext


@dataclass(frozen=True)
class SampleContextItems:
    memories: tuple[MemoryItem, ...]
    rules: tuple[RuleItem, ...]
    state_values: tuple[float, float, float]


class FrozenQwenContextEncoder:
    def __init__(self, backend: Qwen3Backend):
        self.backend = backend

    def encode_text_groups(self, groups: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(groups)
        max_items = max((len(group) for group in groups), default=0)
        if max_items == 0:
            return (
                torch.zeros((batch_size, 0, 1024), dtype=self.backend.dtype, device=self.backend.device),
                torch.zeros((batch_size, 0), dtype=torch.bool, device=self.backend.device),
            )
        vectors = torch.zeros((batch_size, max_items, 1024), dtype=self.backend.dtype, device=self.backend.device)
        mask = torch.zeros((batch_size, max_items), dtype=torch.bool, device=self.backend.device)
        flat_texts: list[str] = []
        positions: list[tuple[int, int]] = []
        for batch_index, group in enumerate(groups):
            for item_index, text in enumerate(group):
                flat_texts.append(text)
                positions.append((batch_index, item_index))
        encoded, truncations = self.backend.encode(flat_texts, max_length=64)
        if truncations:
            raise ValueError("context encoding truncated a memory or rule item")
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        with torch.no_grad():
            embeddings = self.backend.model.model.embed_tokens(input_ids)
            token_mask = attention_mask.to(embeddings.dtype).unsqueeze(-1)
            pooled = (embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        for vector, (batch_index, item_index) in zip(pooled, positions, strict=True):
            vectors[batch_index, item_index] = vector
            mask[batch_index, item_index] = True
        return vectors, mask

    def build_context(
        self,
        samples: list[LogicSample],
        attention_mask: torch.Tensor,
        ablation_config=None,
        adapter_enabled: bool = True,
        force_zero_scale: bool = False,
        context_mode: str = "full",
    ) -> CivilizationAdapterContext:
        item_groups = [context_items_for_sample(sample, context_mode=context_mode) for sample in samples]
        memory_texts = [
            [f"{item.summary} {item.content} {item.relation_type}" for item in group.memories]
            for group in item_groups
        ]
        rule_texts = [
            [f"{item.type} {item.condition} {item.effect} {item.source}" for item in group.rules]
            for group in item_groups
        ]
        memory_vectors, memory_mask = self.encode_text_groups(memory_texts)
        rule_vectors, rule_mask = self.encode_text_groups(rule_texts)
        state_values = torch.tensor(
            [group.state_values for group in item_groups],
            dtype=self.backend.dtype,
            device=self.backend.device,
        )
        return CivilizationAdapterContext(
            memory_vectors=memory_vectors,
            memory_mask=memory_mask,
            state_values=state_values,
            rule_vectors=rule_vectors,
            rule_mask=rule_mask,
            attention_mask=attention_mask.to(self.backend.device),
            ablation_config=ablation_config,
            adapter_enabled=adapter_enabled,
            force_zero_scale=force_zero_scale,
        )


def qwen_text_for_sample(sample: LogicSample) -> str:
    return re.sub(
        r"\b[a-z0-9_]+_surface_\d{4}\b",
        f"{sample.variant}_shared_surface",
        sample.text,
    )


def context_items_for_sample(sample: LogicSample, context_mode: str = "full") -> SampleContextItems:
    if context_mode not in {"full", "empty", "wrong"}:
        raise ValueError(f"unsupported context_mode: {context_mode}")
    neutral_state = (0.5, 0.5, 0.5)
    state = dependency_state_for_label(sample.label) if "state" in sample.required_paths else None
    if sample.leakage_family.startswith("rule_conditioned_conflict_v2"):
        state = None
        fixed_rule_arbitration_state = (0.85, 0.15, 0.85)
    else:
        fixed_rule_arbitration_state = neutral_state
    if context_mode == "empty":
        return SampleContextItems((), (), (0.5, 0.5, 0.5))
    if sample.leakage_family.endswith("_grounded_v1") or "stage44a_path_grounded_v2" in sample.leakage_family:
        if context_mode == "wrong":
            wrong_label = LOGIC_LABELS[(LOGIC_LABELS.index(sample.label) + 1) % len(LOGIC_LABELS)]
            if "stage44a_path_grounded_v2" in sample.leakage_family:
                from ..analysis.evidence_answer_data import LABEL_TO_LOCAL_OPTION

                wrong_option = LABEL_TO_LOCAL_OPTION[wrong_label]
                memory_target = f"the neighboring evidence supports the action: {wrong_option}"
                rule_target = f"the neighboring controlling policy requires the action: {wrong_option}"
            else:
                memory_target = "neighboring evidence belongs to a different record and should not decide this case"
                rule_target = "wrong-context rule: prefer a neighboring record, which should reduce confidence"
            state = dependency_state_for_label(wrong_label) if "state" in sample.required_paths else None
        else:
            memory_target = sample.memory_target
            rule_target = sample.rule_target
        memories = []
        if "memory" in sample.required_paths:
            memories.append(
                MemoryItem(
                    id=f"memory_{sample.surface_group_id}",
                    summary="grounded task evidence",
                    content=memory_target,
                    relation_type="grounded_context",
                    priority=1.0,
                    confidence=1.0,
                )
            )
            for index in range(sample.context_noise_count):
                memories.append(
                    MemoryItem(
                        id=f"noise_memory_{index}",
                        summary=f"irrelevant memory {index}",
                        content=f"noise context {index} does not decide the result",
                        relation_type="noise",
                        priority=0.2,
                        confidence=0.3,
                    )
                )
        rules = []
        if "rule" in sample.required_paths:
            rules.append(
                RuleItem(
                    id=f"rule_{sample.surface_group_id}",
                    type="soft",
                    condition="grounded evidence review",
                    effect=rule_target,
                    priority=1.0,
                    source="qwen3_grounded_v1",
                )
            )
        return SampleContextItems(
            memories=tuple(memories),
            rules=tuple(rules),
            state_values=(
                fixed_rule_arbitration_state
                if sample.leakage_family.startswith("rule_conditioned_conflict_v2") and "state" in sample.required_paths
                else
                (state.rigor, state.creativity, state.defensiveness)
                if state is not None
                else neutral_state
            ),
        )
    if context_mode == "wrong":
        wrong_label = LOGIC_LABELS[(LOGIC_LABELS.index(sample.label) + 1) % len(LOGIC_LABELS)]
        state = dependency_state_for_label(wrong_label) if "state" in sample.required_paths else None
        memory_target = f"memory_selects_{wrong_label}"
        rule_target = f"rule_selects_{wrong_label}"
    else:
        memory_target = sample.memory_target
        rule_target = sample.rule_target

    memories: list[MemoryItem] = []
    if "memory" in sample.required_paths:
        memories.append(
            MemoryItem(
                id=f"memory_{sample.surface_group_id}",
                summary=f"context memory {memory_target}",
                content=f"{memory_target} resolves the hidden relation",
                relation_type=memory_target,
                priority=1.0,
                confidence=1.0,
            )
        )
        for index in range(sample.context_noise_count):
            memories.append(
                MemoryItem(
                    id=f"noise_memory_{index}",
                    summary=f"irrelevant memory {index}",
                    content=f"noise context {index} does not decide the result",
                    relation_type="noise",
                    priority=0.2,
                    confidence=0.3,
                )
            )
    rules: list[RuleItem] = []
    if "rule" in sample.required_paths:
        rules.append(
            RuleItem(
                id=f"rule_{sample.surface_group_id}",
                type="soft",
                condition="context",
                effect=f"{rule_target} selects the controlling result",
                priority=1.0,
                source="qwen3_adapter",
            )
        )
        for index in range(sample.context_noise_count):
            rules.append(
                RuleItem(
                    id=f"noise_rule_{index}",
                    type="soft",
                    condition="noise",
                    effect=f"irrelevant preference {index}",
                    priority=0.2,
                    source="qwen3_adapter_noise",
                )
            )
        for index in range(sample.conflict_context_count):
            rules.append(
                RuleItem(
                    id=f"conflict_rule_{index}",
                    type="conflict",
                    condition="context|conflict",
                    effect=f"conflicting alternative {index}",
                    priority=0.35,
                    source="qwen3_adapter_conflict",
                )
            )
    return SampleContextItems(
        memories=tuple(memories),
        rules=tuple(rules),
        state_values=(
            (state.rigor, state.creativity, state.defensiveness)
            if state is not None
            else neutral_state
        ),
    )
