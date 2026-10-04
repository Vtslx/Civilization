from __future__ import annotations

import pytest

from civilization.engine.stages.evidence_answer_data import build_evidence_answer_samples
from civilization.engine.stages.matched_baseline_benchmark import (
    CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
    MLP_BOTTLENECK,
    _serialize_sample_context,
    plain_adapter_parameter_count,
)
from civilization.engine.stages.necessity_alignment_data import build_necessity_pairs
from civilization.engine.stages.real_task_data import build_local_semireal_task_records, records_to_logic_datasets


def _sample():
    records = build_local_semireal_task_records(2, seed=202, context_grounding_mode="grounded_v1")
    datasets, _ = records_to_logic_datasets(records, max_seq_len=64, context_grounding_mode="grounded_v1")
    samples = next(iter(datasets.values()))
    return build_necessity_pairs(build_evidence_answer_samples(samples, "local_semireal"))[0].full_sample


def test_plain_adapter_parameter_match_is_below_one_percent():
    count = plain_adapter_parameter_count(MLP_BOTTLENECK)
    assert abs(count - CIVILIZATION_OPTIMIZED_PARAMETER_COUNT) / CIVILIZATION_OPTIMIZED_PARAMETER_COUNT < 0.01


def test_prompt_only_excludes_context_and_token_context_contains_it():
    sample = _sample()
    prompt = _serialize_sample_context(sample, "full", include_context=False)
    context = _serialize_sample_context(sample, "full", include_context=True)
    assert "[TASK]" in prompt
    assert "[MEMORY]" not in prompt
    assert "[MEMORY]" in context
    assert "[RULE]" in context
    assert "[STATE]" in context


def test_token_context_ablations_remove_only_the_named_channel():
    sample = _sample()
    full = _serialize_sample_context(sample, "full", include_context=True)
    no_memory = _serialize_sample_context(sample, "no_memory", include_context=True)
    no_rule = _serialize_sample_context(sample, "no_rule", include_context=True)
    assert sample.memory_target in full
    assert sample.memory_target not in no_memory
    assert sample.rule_target in full
    assert sample.rule_target not in no_rule
    assert "[TASK]" in no_memory and "[TASK]" in no_rule


def test_unknown_context_mode_fails_closed():
    with pytest.raises(ValueError, match="unsupported token-context mode"):
        _serialize_sample_context(_sample(), "invented", include_context=True)
