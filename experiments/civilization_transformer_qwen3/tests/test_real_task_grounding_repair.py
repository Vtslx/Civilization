from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from experiments.civilization_transformer_qwen3.analysis.answer_option_readout import (
    answer_options_for_samples,
    score_answer_options,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_data import (
    build_local_semireal_task_records,
    records_to_logic_datasets,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_grounding_repair import (
    run_qwen3_real_task_grounding_repair,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_data import _normalize_external_row
from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS
import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_grounded_context_is_deterministic_and_avoids_logic_label_names() -> None:
    first = build_local_semireal_task_records(2, seed=202, context_grounding_mode="grounded_v1")
    second = build_local_semireal_task_records(2, seed=202, context_grounding_mode="grounded_v1")
    assert first == second
    datasets, _ = records_to_logic_datasets(first, max_seq_len=64, context_grounding_mode="grounded_v1")
    forbidden = set(LOGIC_LABELS)
    for samples in datasets.values():
        for sample in samples:
            context_text = f"{sample.memory_target} {sample.rule_target}".lower()
            assert not (forbidden & set(context_text.replace("_", " ").split()))
            assert sample.leakage_family.endswith("_grounded_v1")


def test_answer_option_readout_scores_expected_option() -> None:
    records = build_local_semireal_task_records(1, context_grounding_mode="grounded_v1")
    datasets, _ = records_to_logic_datasets(records, max_seq_len=64, context_grounding_mode="grounded_v1")
    samples = datasets["operation_decision"][:2]
    options = answer_options_for_samples(samples)
    option_vectors = np.eye(len(options), dtype=np.float32)
    sample_vectors = np.stack([option_vectors[options.index(sample.expected_pattern)] for sample in samples])
    metrics = score_answer_options(sample_vectors, samples, option_vectors, options)
    assert metrics.accuracy == 1.0
    assert metrics.rows


def test_grounding_repair_smoke_writes_required_artifacts(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rows = [
        {"sentence1": "A starts.", "sentence2": "A runs.", "label": 0},
        {"sentence1": "B starts.", "sentence2": "B is denied.", "label": 1},
        {"sentence1": "C starts.", "sentence2": "C runs.", "label": 0},
        {"sentence1": "D starts.", "sentence2": "D is denied.", "label": 1},
    ]
    with (cache_dir / "glue_rte.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = _normalize_external_row("glue_rte", row, index)
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")
    summary = run_qwen3_real_task_grounding_repair(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=2,
        local_train_groups=1,
        external_task_names=("glue_rte",),
        external_cache_dir=cache_dir,
        external_train_per_label=1,
        external_heldout_per_label=1,
        training_steps=1,
        preferred_device="cpu",
        evaluation_batch_size=20,
        evaluation_modes=("full", "adapter_disabled", "wrong_context"),
    )
    required = {
        "summary.json",
        "grounded_context_manifest.json",
        "answer_option_metrics.csv",
        "fixed_centroid_metrics.csv",
        "ablation_drop.csv",
        "wrong_context.csv",
        "truncation_cases.json",
        "failure_cases.json",
    }
    assert required.issubset({path.name for path in (tmp_path / "out").iterdir()})
    assert summary["context_grounding_mode"] == "grounded_v1"
    assert "stage_gates" in summary
