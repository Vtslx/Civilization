from __future__ import annotations

import json

from civilization.engine.stages.stage54_long_running_service_stress import (
    Stage54LongRunConfig,
    run_stage54_fake_long_running_stress,
)


def _fast_config() -> Stage54LongRunConfig:
    return Stage54LongRunConfig(
        duration_seconds=0.8,
        request_interval_seconds=0.1,
        reload_interval_seconds=0.25,
        snapshot_interval_seconds=0.2,
        controls=("full", "adapter_disabled"),
        min_reload_count=1,
    )


def test_stage54_fake_long_run_passes_and_writes_artifacts(tmp_path) -> None:
    summary = run_stage54_fake_long_running_stress(output_dir=tmp_path, config=_fast_config())
    assert summary["passes_stage_gate"]
    assert summary["request_count"] > 1
    assert summary["reload_count"] >= 1
    assert summary["snapshot_count"] >= 1
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "request_results.jsonl").exists()
    assert (tmp_path / "reload_events.jsonl").exists()
    assert (tmp_path / "metrics_snapshots.jsonl").exists()
    assert (tmp_path / "metrics_snapshots.csv").exists()


def test_stage54_no_half_switch_versions(tmp_path) -> None:
    summary = run_stage54_fake_long_running_stress(output_dir=tmp_path, config=_fast_config())
    assert summary["stage_gates"]["no_half_switch_versions"]
    assert not summary["invalid_versions"]


def test_stage54_metrics_snapshots_are_jsonl(tmp_path) -> None:
    run_stage54_fake_long_running_stress(output_dir=tmp_path, config=_fast_config())
    rows = [json.loads(line) for line in (tmp_path / "metrics_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all("rss_mb" in row for row in rows)
