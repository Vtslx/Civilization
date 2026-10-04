from __future__ import annotations

import json

from civilization.engine.stages.stage53_concurrent_reload_stress import (
    Stage53ConcurrentReloadConfig,
    run_stage53_fake_concurrent_reload_stress,
)


def test_stage53_fake_concurrent_reload_passes_and_writes_artifacts(tmp_path) -> None:
    summary = run_stage53_fake_concurrent_reload_stress(
        output_dir=tmp_path,
        config=Stage53ConcurrentReloadConfig(
            pre_reload_requests=1,
            concurrent_requests=4,
            post_reload_requests=1,
            concurrency=2,
            controls=("full", "adapter_disabled"),
        ),
    )
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["no_half_switch_versions"]
    assert summary["stage_gates"]["only_draining_errors"]
    assert summary["stage_gates"]["post_reload_version_seen"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "request_results.jsonl").exists()
    assert (tmp_path / "resource_usage.json").exists()


def test_stage53_versions_are_old_or_new_only(tmp_path) -> None:
    summary = run_stage53_fake_concurrent_reload_stress(
        output_dir=tmp_path,
        config=Stage53ConcurrentReloadConfig(pre_reload_requests=2, concurrent_requests=3, post_reload_requests=2, concurrency=3),
    )
    allowed = {summary["old_version"], summary["new_version"]}
    assert set(summary["successful_versions"]).issubset(allowed)
    assert not summary["invalid_versions"]


def test_stage53_request_results_are_jsonl(tmp_path) -> None:
    run_stage53_fake_concurrent_reload_stress(
        output_dir=tmp_path,
        config=Stage53ConcurrentReloadConfig(pre_reload_requests=1, concurrent_requests=1, post_reload_requests=1, concurrency=1),
    )
    rows = [json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all("request_index" in row for row in rows)
