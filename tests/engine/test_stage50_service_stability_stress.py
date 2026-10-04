from __future__ import annotations

import json

from civilization.engine.stages.stage50_service_stability_stress import (
    Stage50StressConfig,
    run_stage50_fake_stress,
)


def test_stage50_fake_stress_writes_required_artifacts(tmp_path) -> None:
    summary = run_stage50_fake_stress(
        output_dir=tmp_path,
        config=Stage50StressConfig(total_requests=6, concurrency=2, batch_size=2, controls=("full", "no_memory_path")),
    )
    assert summary["passes_stage_gate"]
    assert summary["total_prediction_rows"] == 12
    assert summary["control_audit_failures"] == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "request_results.jsonl").exists()
    assert (tmp_path / "latency_metrics.csv").exists()
    assert (tmp_path / "resource_usage.json").exists()


def test_stage50_stress_records_resource_and_latency_metrics(tmp_path) -> None:
    summary = run_stage50_fake_stress(
        output_dir=tmp_path,
        config=Stage50StressConfig(total_requests=4, concurrency=1, batch_size=1, controls=("full",)),
    )
    assert summary["latency_metrics"]["count"] == 4
    assert summary["latency_metrics"]["max_seconds"] >= 0.0
    assert "rss_growth_mb" in summary["resource_usage"]
    rows = [json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["failure_count"] == 0 for row in rows)


def test_stage50_failure_gate_fails_when_request_limit_is_exceeded(tmp_path) -> None:
    summary = run_stage50_fake_stress(
        output_dir=tmp_path,
        config=Stage50StressConfig(total_requests=3, concurrency=1, batch_size=3, max_batch_size=1),
    )
    assert not summary["passes_stage_gate"]
    assert summary["failure_count"] > 0
    assert not summary["stage_gates"]["failure_rate"]
