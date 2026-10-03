import json

import pytest

from experiments.civilization_transformer_qwen3.analysis.real_task_data import (
    EXTERNAL_DATASET_SPECS,
    build_local_semireal_task_records,
    load_external_task_records,
    records_to_logic_datasets,
    split_by_surface_group,
)
from experiments.civilization_transformer_qwen3.analysis.real_task_migration import (
    run_qwen3_real_task_migration,
)
from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_local_semireal_records_are_deterministic_and_complete() -> None:
    first = build_local_semireal_task_records(samples_per_label=3, seed=202)
    second = build_local_semireal_task_records(samples_per_label=3, seed=202)
    assert first == second
    assert set(first) == {
        "operation_decision",
        "rule_conflict",
        "causal_trace",
        "condition_check",
        "priority_selection",
        "negation_constraint",
    }
    for rows in first.values():
        assert {record.label for record in rows} == set(LOGIC_LABELS)
        assert all(record.source_type == "local_semireal" for record in rows)
        assert all(record.memory_items for record in rows)
        assert all(record.rule_items for record in rows)
        assert all(record.expected_answer for record in rows)


def test_local_semireal_split_keeps_surface_groups_isolated() -> None:
    records = build_local_semireal_task_records(samples_per_label=4, seed=202)
    datasets, _ = records_to_logic_datasets(records, max_seq_len=64)
    train, test = split_by_surface_group(datasets["operation_decision"], train_groups=2, seed=202)
    assert train and test
    assert {sample.surface_group_id for sample in train}.isdisjoint(
        {sample.surface_group_id for sample in test}
    )


def test_external_cache_required_unless_download_allowed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="ALLOW_DATASET_DOWNLOAD"):
        load_external_task_records(cache_dir=tmp_path, task_names=("glue_rte",), allow_download=False)


def test_external_cache_manifest_and_label_mapping(tmp_path) -> None:
    cache = tmp_path / "glue_rte.jsonl"
    rows = [
        {"sentence1": "A system starts.", "sentence2": "The system runs.", "label": 0},
        {"sentence1": "A system starts.", "sentence2": "The system is denied.", "label": 1},
    ]
    with cache.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            # Use the same normalization path as a downloaded cache would use.
            from experiments.civilization_transformer_qwen3.analysis.real_task_data import _normalize_external_row
            from dataclasses import asdict

            record = _normalize_external_row("glue_rte", row, index)
            handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")
    datasets, manifest = load_external_task_records(
        cache_dir=tmp_path,
        task_names=("glue_rte",),
        allow_download=False,
    )
    assert set(datasets) == {"glue_rte"}
    assert {record.label for record in datasets["glue_rte"]} == {"condition", "negation"}
    assert manifest[0]["dataset_id"] == EXTERNAL_DATASET_SPECS["glue_rte"]["dataset_id"]
    assert manifest[0]["source"] == "cache"
    assert manifest[0]["fingerprint"]


def test_real_task_migration_smoke_writes_artifacts(tmp_path) -> None:
    summary = run_qwen3_real_task_migration(
        output_dir=tmp_path,
        seeds=(202,),
        local_samples_per_label=2,
        local_train_groups=1,
        external_task_names=(),
        training_steps=1,
        preferred_device="cpu",
        evaluation_batch_size=20,
        evaluation_modes=("full", "adapter_disabled", "zero_scale"),
    )
    required = {
        "summary.json",
        "dataset_manifest.json",
        "training_runs.json",
        "loss_curves.csv",
        "local_task_metrics.csv",
        "external_task_metrics.csv",
        "ablation_drop.csv",
        "wrong_context.csv",
        "language_preservation.csv",
        "resource_usage.json",
        "failure_cases.json",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    assert summary["external_training_default"] == "evaluation_only"
    assert summary["stage_gates"]["external_baselines_recorded"]
    assert summary["stage_gates"]["qwen_frozen"]

